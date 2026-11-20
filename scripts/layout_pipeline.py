"""Overlap page decode and artifact encoding with the single local Paddle predictor."""

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from hashlib import sha256
from io import BytesIO
import time

import cv2
from PIL import Image, ImageDraw


def load(page):
    image = cv2.imread(page["image"], cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {page['image']}")
    return image


def encode(page, image, boxes):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    crops = []
    for idx, box in enumerate(boxes):
        coords = box["coordinate"]
        box["read_order"] = idx
        draw.rectangle(
            coords, outline="#ef2939" if box["label"] == "table" else "#3065cc", width=3
        )
        draw.text(
            (coords[0] + 3, coords[1] + 3),
            f"{idx} {box['label']}",
            fill="#a42020",
            stroke_width=1,
            stroke_fill="white",
        )
        if box["label"] == "table":
            crop = image.crop(tuple(map(int, coords)))
            if min(crop.size) >= 2:
                data = BytesIO()
                crop.save(data, format="PNG")
                payload = data.getvalue()
                crops.append((idx, sha256(payload).hexdigest(), payload))
    canvas.thumbnail((900, 1250))
    jpeg = BytesIO()
    canvas.save(jpeg, format="JPEG", quality=82)
    return image.size, boxes, crops, jpeg.getvalue()


def encode_bgr(page, bgr, boxes):
    return encode(page, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), boxes)


def predict_pages(model, pages, pipelined):
    if not pipelined:
        for p in pages:
            with Image.open(p["image"]) as raw:
                im = raw.convert("RGB")
            start = time.perf_counter()
            boxes = next(iter(model.predict(p["image"]))).json["res"]["boxes"]
            elapsed = time.perf_counter() - start
            yield p, elapsed, encode(p, im, boxes)
        return
    # At most two prefetched inputs and three encodes; commit in input order.
    with ThreadPoolExecutor(2) as reader, ThreadPoolExecutor(3) as writer:
        source = iter(pages)
        reads = deque(
            (p, reader.submit(load, p))
            for p in [next(source, None), next(source, None)]
            if p is not None
        )
        writes = deque()
        while reads:
            p, future = reads.popleft()
            nxt = next(source, None)
            if nxt is not None:
                reads.append((nxt, reader.submit(load, nxt)))
            bgr = future.result()
            start = time.perf_counter()
            boxes = next(iter(model.predict(bgr))).json["res"]["boxes"]
            elapsed = time.perf_counter() - start
            writes.append((p, elapsed, writer.submit(encode_bgr, p, bgr, boxes)))
            if len(writes) >= 3:
                page, dt, task = writes.popleft()
                yield page, dt, task.result()
        while writes:
            page, dt, task = writes.popleft()
            yield page, dt, task.result()
