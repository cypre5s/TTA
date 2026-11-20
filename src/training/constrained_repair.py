"""Rank copy-preserving, one-tag OTSL edits with the trained encoder-decoder."""

import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch

from scripts.baseline_score import load
from scripts.evaluate import read_split
from scripts.ground_truth import immutable, jbytes, sha
from src.otsl.grid import TOKENS, parse
from src.training.repair import PAD, TAGS, RepairModel, corrupt, encode

CHECKPOINT = Path("models/otsl_repair/server_v1/best.pt")
BASELINE = Path("experiments/baseline/original_v1/B0")
ROOT = Path("experiments/ablation")
DATA = Path("data/splits/ground_truth_v1")
POLICY = dict(
    eligible="strict parse failure",
    max_candidates=64,
    max_tags=320,
    edits="one inserted, deleted or replaced OTSL tag; preserve complete text payload",
    ranking="maximum mean target log probability under validation-selected transformer",
)


def valid(text):
    try:
        parse(text)
        return True
    except ValueError:
        return False


def candidates(raw):
    tags = list(TOKENS.finditer(raw))
    if not tags or len(tags) > POLICY["max_tags"]:
        return []
    choices = {}

    def check(s):
        if s not in choices and valid(s):
            choices[s] = None

    for i, match in enumerate(tags):
        start, end = match.span()
        next_start = tags[i + 1].start() if i + 1 < len(tags) else len(raw)
        payload = raw[end:next_start].strip()
        if not payload:
            check(raw[:start] + raw[end:])
        for tag in ["<fcel>"] if payload else TAGS:
            if tag != match[0]:
                check(raw[:start] + tag + raw[end:])
        for tag in ("<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>"):
            check(raw[:start] + tag + raw[start:])
        if len(choices) >= POLICY["max_candidates"]:
            break
    for tag in ("<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>"):
        check(raw + tag)
    return list(choices)[: POLICY["max_candidates"]]


class Ranker:
    def __init__(self, path, device):
        state = torch.load(path, map_location=device, weights_only=True)
        self.device = device
        self.model = RepairModel(state["limit"]).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.checkpoint_sha256 = sha(path.read_bytes())

    @torch.inference_mode()
    def repair(self, raw):
        if valid(raw):
            return raw, "valid_input", 0, None
        source = encode(raw)
        if len(source) > self.model.limit:
            return raw, "over_length", 0, None
        options = [(s, encode(s)) for s in candidates(raw)]
        options = [(s, ids) for s, ids in options if len(ids) <= self.model.limit]
        if not options:
            return raw, "no_valid_edit", 0, None
        x = torch.tensor([source], device=self.device)
        memory = self.model.core.encoder(self.model.embedding(x))
        scores = []
        for start in range(0, len(options), 8):
            batch = options[start : start + 8]
            targets = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for _, ids in batch],
                batch_first=True,
                padding_value=PAD,
            ).to(self.device)
            tgt, labels = targets[:, :-1], targets[:, 1:]
            mask = torch.ones(
                tgt.shape[1], tgt.shape[1], device=self.device, dtype=torch.bool
            ).triu(1)
            output = self.model.core.decoder(
                self.model.embedding(tgt),
                memory.expand(len(batch), -1, -1),
                tgt_mask=mask,
                tgt_key_padding_mask=tgt.eq(PAD),
            )
            nll = torch.nn.functional.cross_entropy(
                self.model.head(output).transpose(1, 2),
                labels,
                reduction="none",
                ignore_index=PAD,
            )
            scores.extend((-nll.sum(1) / labels.ne(PAD).sum(1)).tolist())
        i = max(range(len(scores)), key=scores.__getitem__)
        return options[i][0], "model_ranked", len(options), scores[i]


def validation(ranker):
    rng = random.Random(43)
    counts = {}
    for rec in read_split(DATA, "validation"):
        gt = load(Path(rec["gt"]))
        good = gt["corrected_otsl"]
        raw = Path("data/otsl/raw", rec["table_id"] + ".otsl").read_text()
        synthetic, error = corrupt(good, rng)
        for name, source in [("raw", raw), ("synthetic", synthetic)]:
            best, reason, _, _ = ranker.repair(source)
            c = counts.setdefault(name, Counter())
            c.update(
                {
                    "total": 1,
                    reason: 1,
                    "exact_before": int(source == good),
                    "exact_after": int(best == good),
                    "valid_before": int(valid(source)),
                    "valid_after": int(valid(best)),
                    "changed": int(source != best),
                }
            )
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", choices=["validation", "B0", "fine_original", "fine_lora"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ranker = Ranker(CHECKPOINT, args.device)
    if args.input == "validation":
        out = dict(
            policy=POLICY,
            checkpoint_sha256=ranker.checkpoint_sha256,
            counts=validation(ranker),
        )
        immutable(ROOT / "repair_validation.json", jbytes(out))
        print(out, flush=True)
        return
    src = BASELINE if args.input == "B0" else ROOT / args.input / "predictions"
    files = sorted(src.glob("*.json"))
    if args.input == "B0" and len(files) != len(read_split(DATA, "test")):
        raise ValueError("Incomplete B0 predictions")
    for p in files:
        out = ROOT / "repaired" / args.input / p.name
        if out.exists():
            continue
        raw = load(p)["raw_otsl"]
        start = time.monotonic()
        repaired, reason, count, score = ranker.repair(raw)
        try:
            parse(raw)
            error_type = None
        except ValueError as exc:
            error_type = str(exc)
        immutable(
            out,
            jbytes(
                dict(
                    table_id=p.stem,
                    raw_sha256=sha(raw.encode()),
                    source_sha256=sha(p.read_bytes()),
                    repaired_otsl=repaired,
                    decision=reason,
                    candidate_count=count,
                    mean_log_probability=score,
                    repair_confidence=math.exp(score) if score is not None else None,
                    confidence_type="geometric mean target token probability; uncalibrated",
                    error_type=error_type,
                    checkpoint_sha256=ranker.checkpoint_sha256,
                    seconds=time.monotonic() - start,
                )
            ),
        )
    print(args.input, len(files), flush=True)


if __name__ == "__main__":
    main()
