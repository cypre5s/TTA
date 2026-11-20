"""Paired full generations against the deployed model with per-image accuracy evidence."""

import argparse
import hashlib
import json
import re
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from scripts.bench_xiaomi import cfg, fit_image, out
from scripts.cloud_ocr import generate, save
from src.xiaomi_ocr.fast_decode import variant
from src.xiaomi_ocr.attention_graph import graph_hybrid_decode
from src.xiaomi_ocr.decode_kernels import accelerated_decode
from src.xiaomi_ocr.graph_decode import graph_linear_layers, graph_linear_stacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "deployed",
            "norm",
            "projections",
            "norm_projections",
            "norm_projections_recurrent",
            "deployed",
        ],
    )
    parser.add_argument("--inputs", type=Path, default=out / "xiaomi_inputs.json")
    parser.add_argument("--output", type=Path, default=out / "xiaomi_deep_decode.json")
    args = parser.parse_args()
    paths = [Path(p) for p in json.loads(args.inputs.read_text())]
    processor = AutoProcessor.from_pretrained(cfg["checkpoint"], local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = (
        AutoModelForImageTextToText.from_pretrained(
            cfg["checkpoint"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    result = dict(
        gpu=torch.cuda.get_device_name(), inputs=list(map(str, paths)), trials=[]
    )
    expected = None
    for mode in args.modes:
        trial = dict(mode=mode)
        try:
            with ExitStack() as stack:
                kernels = (
                    mode.removesuffix("_graph")
                    .removesuffix("_stack")
                    .removesuffix("_hybrid")
                )
                plain = re.fullmatch(r"original(4|8|12|16)", kernels)
                batch_size = int(plain.group(1)) if plain else 8
                if not plain:
                    stack.enter_context(variant(model, "packed_conv"))
                    if kernels != "deployed":
                        stack.enter_context(accelerated_decode(model, kernels))
                if mode.endswith("_graph"):
                    stack.enter_context(graph_linear_layers(model))
                elif mode.endswith("_stack"):
                    stack.enter_context(graph_linear_stacks(model))
                elif mode.endswith("_hybrid"):
                    stack.enter_context(graph_hybrid_decode(model))
                generate(
                    model,
                    processor,
                    paths[:1],
                    {**cfg, "max_new_tokens": 4},
                    fit_image,
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                hashes = []
                outputs = []
                lengths = []
                for i in range(0, len(paths), batch_size):
                    rows = generate(
                        model, processor, paths[i : i + batch_size], cfg, fit_image
                    )
                    for row in rows:
                        raw = processor.decode(
                            row,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        outputs.append(raw)
                        hashes.append(hashlib.sha256(raw.encode()).hexdigest())
                        lengths.append(
                            int((row != processor.tokenizer.pad_token_id).sum())
                        )
                elapsed = time.perf_counter() - start
                if expected is None:
                    if mode not in {"deployed", "original4", "original8"}:
                        raise ValueError("Start with a baseline")
                    expected = hashes
                evidence = args.output.parent / (args.output.stem + "_raw")
                evidence.mkdir(exist_ok=True)
                save(
                    evidence / (mode + ".json"),
                    dict(zip((p.stem for p in paths), outputs)),
                )
                trial.update(
                    seconds=elapsed,
                    tokens=sum(lengths),
                    hashes=hashes,
                    token_lengths=lengths,
                    peak_bytes=torch.cuda.max_memory_allocated(),
                    matches=sum(a == b for a, b in zip(hashes, expected)),
                    changed=[
                        str(p) for p, a, b in zip(paths, hashes, expected) if a != b
                    ],
                )
        except (
            ValueError,
            RuntimeError,
            TypeError,
            AttributeError,
            ImportError,
        ) as exc:
            trial["error"] = repr(exc)
            torch.cuda.empty_cache()
        result["trials"].append(trial)
        save(args.output, result)
        print(
            mode,
            {
                k: v
                for k, v in trial.items()
                if k not in ("changed", "hashes", "token_lengths")
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
