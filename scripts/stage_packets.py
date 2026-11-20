"""Continuously publish only the currently unlocked visual review stage."""

import time
from review import packets, raw, verify, freeze

for target, previous in [(10, 0), (100, 10), (1000, 100)]:
    if len(list(packets.glob("*.json"))) >= target:
        continue
    while previous:
        if len(list(raw.glob("*.json"))) >= previous:
            try:
                if set(range(1, previous + 1)) <= {x["ordinal"] for x in verify()}:
                    break
            except (ValueError, OSError):
                pass
        time.sleep(12)
    while freeze(target) < target:
        time.sleep(12)
    print("packet stage ready", target, flush=True)
