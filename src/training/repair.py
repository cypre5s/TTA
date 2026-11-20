"""Byte and OTSL tag encoder-decoder, trained on reviewed project splits only."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import torch
from torch import nn
from src.otsl.grid import TOKENS, parse
from scripts.ground_truth import sha, immutable, jbytes
from scripts.training_reference import permits_machine_training, training_reference

TAGS = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>")
PAD, BOS, EOS = 0, 1, 2
OFFSET = 3 + len(TAGS)
VOCAB = OFFSET + 256


def encode(text):
    out = [BOS]
    last = 0
    for m in TOKENS.finditer(text):
        out.extend(OFFSET + b for b in text[last : m.start()].encode())
        out.append(3 + TAGS.index(m[0]))
        last = m.end()
    return out + [OFFSET + b for b in text[last:].encode()] + [EOS]


def decode(ids):
    out = bytearray()
    for n in ids:
        if n == EOS:
            break
        if 3 <= n < OFFSET:
            out.extend(TAGS[n - 3].encode())
        elif n >= OFFSET:
            out.append(n - OFFSET)
    return out.decode("utf-8", errors="replace")


def corrupt(raw, rng):
    ms = list(TOKENS.finditer(raw))
    cells = [m for m in ms if m[1] != "nl"]
    if not cells:
        raise ValueError("No cells")
    m = rng.choice(cells)
    end = next((t.start() for t in ms if t.start() > m.start()), len(raw))
    kind = rng.choice(["missing_cell", "extra_cell", "wrong_span", "wrong_row"])
    if kind == "missing_cell":
        bad = raw[: m.start()] + raw[end:]
    elif kind == "extra_cell":
        bad = raw[: m.start()] + "<ecel>" + raw[m.start() :]
    elif kind == "wrong_span":
        repl = rng.choice([t for t in ("<lcel>", "<ucel>", "<xcel>") if t != m[0]])
        bad = raw[: m.start()] + repl + raw[end:]
    else:
        bad = raw[: m.start()] + "<nl>" + raw[m.start() :]
    return bad, kind


class RepairModel(nn.Module):
    def __init__(self, limit=1536, width=96):
        super().__init__()
        self.limit = limit
        self.tokens = nn.Embedding(VOCAB, width, padding_idx=PAD)
        self.pos = nn.Embedding(limit, width)
        self.core = nn.Transformer(
            d_model=width,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=width * 3,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Linear(width, VOCAB)

    def embedding(self, ids):
        return self.tokens(ids) + self.pos(
            torch.arange(ids.shape[1], device=ids.device)
        )

    def forward(self, x, y):
        mask = torch.ones(
            (y.shape[1], y.shape[1]), device=y.device, dtype=torch.bool
        ).triu(1)
        return self.head(
            self.core(
                self.embedding(x),
                self.embedding(y),
                tgt_mask=mask,
                src_key_padding_mask=x.eq(PAD),
                memory_key_padding_mask=x.eq(PAD),
                tgt_key_padding_mask=y.eq(PAD),
            )
        )


def samples(split, name, seed, limit):
    from scripts.evaluate import read_split

    rng = random.Random(seed)
    pairs = []
    stats = Counter()
    allow_machine = permits_machine_training(split)
    for row in read_split(split, name):
        gt = training_reference(row, allow_machine)
        good = gt["corrected_otsl"]
        parse(good)
        synthetic, error = corrupt(good, rng)
        examples = [(synthetic, "synthetic", error)]
        predicted = Path("data/otsl/raw") / (gt["table_id"] + ".otsl")
        if predicted.exists():
            original = predicted.read_text()
            if original != good:
                examples.append((original, "real", "reviewer_corrected"))
        for bad, origin, error in examples:
            inp, target = encode(bad), encode(good)
            if max(len(inp), len(target)) > limit:
                stats["excluded_over_length_" + origin] += 1
                continue
            pairs.append((inp, target, origin, error))
            stats[origin] += 1
    if not pairs:
        raise ValueError(f"No eligible {name} repair pairs")
    return pairs, stats


def loss(model, pair, device):
    x, y = [torch.tensor([v], device=device) for v in pair[:2]]
    return nn.functional.cross_entropy(
        model(x, y[:, :-1]).flatten(0, 1), y[:, 1:].reshape(-1)
    )


def train(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tr, train_counts = samples(args.dataset, "train", args.seed, args.limit)
    val, val_counts = samples(args.dataset, "validation", args.seed + 1, args.limit)
    args.output.mkdir(parents=True, exist_ok=False)
    model = RepairModel(args.limit).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    best = float("inf")
    epochs = []
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(tr)
        for pair in tr:
            optim.zero_grad(set_to_none=True)
            v = loss(model, pair, args.device)
            v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        by = {}
        model.eval()
        with torch.inference_mode():
            for pair in val:
                by.setdefault(pair[2], []).append(loss(model, pair, args.device).item())
        means = {k: sum(v) / len(v) for k, v in by.items()}
        score = sum(sum(v) for v in by.values()) / sum(map(len, by.values()))
        epochs.append(dict(epoch=epoch + 1, validation_loss=means))
        state = dict(model=model.state_dict(), limit=args.limit, epoch=epoch + 1)

        def checkpoint(name):
            p = args.output / name
            tmp = p.with_suffix(".tmp")
            torch.save(state, tmp)
            tmp.replace(p)

        checkpoint("last.pt")
        if score < best:
            checkpoint("best.pt")
            best = score
    immutable(
        args.output / "training.json",
        jbytes(
            dict(
                seed=args.seed,
                split=sha((args.dataset / "manifest.json").read_bytes()),
                train_counts=train_counts,
                validation_counts=val_counts,
                epochs=epochs,
                device=args.device,
            )
        ),
    )


@torch.inference_mode()
def predict(args):
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    model = RepairModel(checkpoint["limit"]).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Preserve the raw input")
    original = args.input.read_text()
    encoded = encode(original)
    if len(encoded) > model.limit:
        raise ValueError("Input exceeds trained length")
    x = torch.tensor([encoded], device=args.device)
    ids = [BOS]
    conf = []
    for _ in range(model.limit - 1):
        y = torch.tensor([ids], device=args.device)
        probs = model(x, y)[0, -1].softmax(-1)
        token = int(probs.argmax())
        ids.append(token)
        conf.append(float(probs[token]))
        if token == EOS:
            break
    raw = decode(ids)
    try:
        parse(raw)
        valid = ids[-1] == EOS
    except ValueError:
        valid = False
    immutable(
        args.output,
        jbytes(
            dict(
                table_id=args.input.stem,
                raw_sha256=sha(original.encode()),
                repaired_otsl=raw,
                structure_valid=valid,
                repair_confidence=sum(conf) / len(conf),
                confidence_type="mean selected token probability; uncalibrated",
                review_status="pending",
            )
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="op", required=True)
    a = sub.add_parser("train")
    a.add_argument("--dataset", type=Path, required=True)
    a.add_argument("--epochs", type=int, default=3)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--limit", type=int, default=1536)
    b = sub.add_parser("predict")
    b.add_argument("--checkpoint", type=Path, required=True)
    b.add_argument("--input", type=Path, required=True)
    for q in [a, b]:
        q.add_argument("--device", default="cpu")
        q.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    train(args) if args.op == "train" else predict(args)
