"""Small local tests of fused delta rule and prompt lookup on the frozen cohort."""

from pathlib import Path
import time, json, hashlib
from contextlib import contextmanager
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from scripts.bench_xiaomi import cfg, fit_image, out
from scripts.cloud_ocr import generate, save
from PIL import Image


@contextmanager
def fused_delta():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    from fla.ops.gated_delta_rule import (
        fused_recurrent_gated_delta_rule,
        chunk_gated_delta_rule,
    )

    old = (q.torch_recurrent_gated_delta_rule, q.torch_chunk_gated_delta_rule)

    def wrap(kernel):
        def run(
            query,
            key,
            value,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
            **kwargs,
        ):
            if use_qk_l2norm_in_kernel:
                query, key = q.l2norm(query), q.l2norm(key)
            return kernel(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )

        return run

    try:
        q.torch_recurrent_gated_delta_rule = wrap(fused_recurrent_gated_delta_rule)
        q.torch_chunk_gated_delta_rule = wrap(chunk_gated_delta_rule)
        yield
    finally:
        q.torch_recurrent_gated_delta_rule, q.torch_chunk_gated_delta_rule = old


def main():
    prior = json.loads((out / "xiaomi_local_comparison.json").read_text())
    paths = list(map(Path, prior["inputs"]))
    proc = AutoProcessor.from_pretrained(cfg["checkpoint"], local_files_only=True)
    proc.tokenizer.padding_side = "left"
    model = (
        AutoModelForImageTextToText.from_pretrained(
            cfg["checkpoint"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    details = dict(inputs=prior["inputs"])
    with Image.open(paths[0]) as f:
        img = fit_image(f.convert("RGB"))
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": cfg["prompt"]},
                {"type": "image", "image": img},
            ],
        }
    ]
    start = time.perf_counter()
    inp = proc.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda", torch.bfloat16)
    torch.cuda.synchronize()
    details["prepare_first_seconds"] = time.perf_counter() - start
    with torch.inference_mode():
        try:
            start = time.perf_counter()
            ids = model.generate(
                **inp,
                do_sample=False,
                max_new_tokens=192,
                prompt_lookup_num_tokens=3,
                use_cache=True,
            )
            candidate = ids[:, inp.input_ids.shape[1] :].cpu()
            elapsed = time.perf_counter() - start
            start = time.perf_counter()
            gold = model.generate(
                **inp, do_sample=False, max_new_tokens=192, use_cache=True
            )[:, inp.input_ids.shape[1] :].cpu()
            details["prompt_lookup"] = dict(
                seconds=elapsed,
                baseline_seconds=time.perf_counter() - start,
                equal=torch.equal(candidate, gold),
                tokens=candidate.numel(),
            )
        except (
            ValueError,
            TypeError,
            RuntimeError,
            NotImplementedError,
            AttributeError,
        ) as exc:
            details["prompt_lookup"] = dict(error=f"{type(exc).__name__}: {exc}")
    save(out / "xiaomi_extra.json", details)
    try:
        with fused_delta():
            generate(model, proc, paths[:1], {**cfg, "max_new_tokens": 4}, fit_image)
            start = time.perf_counter()
            seq = generate(model, proc, paths, cfg, fit_image)
            elapsed = time.perf_counter() - start
            hashes = [
                hashlib.sha256(
                    proc.decode(
                        row,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).encode()
                ).hexdigest()
                for row in seq
            ]
            details["fused_delta"] = dict(
                seconds=elapsed,
                raw_sha256=hashes,
                exact_matches=sum(
                    a == b for a, b in zip(hashes, prior["variants"][0]["raw_sha256"])
                ),
            )
    except (RuntimeError, TypeError, ImportError, AttributeError, ValueError) as exc:
        details["fused_delta"] = dict(error=f"{type(exc).__name__}: {exc}")
    save(out / "xiaomi_extra.json", details)


if __name__ == "__main__":
    main()
