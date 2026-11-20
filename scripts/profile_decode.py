"""Count CUDA kernels on six steady decode steps of eight real images."""

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from transformers import AutoModelForImageTextToText, AutoProcessor
from scripts.bench_xiaomi import cfg, out, fit_image
from scripts.cloud_ocr import generate, save
from src.xiaomi_ocr.fast_decode import variant
from src.xiaomi_ocr.decode_kernels import accelerated_decode


def main():
    paths = [Path(p) for p in json.loads((out / "xiaomi_inputs.json").read_text())]
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
    proc = AutoProcessor.from_pretrained(cfg["checkpoint"], local_files_only=True)
    proc.tokenizer.padding_side = "left"
    result = {}
    for mode in ("deployed", "norm_projections_recurrent"):
        with ExitStack() as stack:
            stack.enter_context(variant(model, "packed_conv"))
            if mode != "deployed":
                stack.enter_context(accelerated_decode(model, mode))
            generate(model, proc, paths, {**cfg, "max_new_tokens": 6}, fit_image)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=3, warmup=2, active=6, repeat=1),
            ) as prof:
                hook = model.register_forward_hook(lambda *args: prof.step())
                try:
                    generate(
                        model, proc, paths, {**cfg, "max_new_tokens": 15}, fit_image
                    )
                finally:
                    hook.remove()
            events = [
                e
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            counts = Counter(e.name for e in events)
            times = Counter()
            for e in events:
                times[e.name] += e.time_range.elapsed_us()
            result[mode] = dict(
                steps=6,
                cuda_events=len(events),
                events_per_step=len(events) / 6,
                profiled_gpu_ms_sum=sum(times.values()) / 1000,
                top_by_count=counts.most_common(10),
                top_by_gpu_us=times.most_common(8),
            )
            save(out / "decode_profile.json", result)
            print(mode, len(events), flush=True)


if __name__ == "__main__":
    main()
