#!/usr/bin/env bash
# Resume durable stages in this checkout. The visual review workers are Codex agents.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-resume}
case "$mode" in collect|process|annotate|resume) ;; *) exit 2 ;; esac
py=.venv/bin/python
layout_py=${PADDLE_PYTHON:-scripts/local_paddle.sh}
start() {
    local name=$1
    shift
    if "$py" - "$PWD" "$@" <<'PY'
import pathlib,sys,os
root=pathlib.Path(sys.argv[1])
# Match running interpreter + script/args, ignoring Python -u.
args=[x for x in sys.argv[2:] if x!='-u'][1:]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        if (proc/'cwd').resolve()!=root:continue
        cmd=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
        if [x for x in cmd[1:] if x!='-u']==args:sys.exit(0)
    except (OSError,UnicodeError):pass
sys.exit(1)
PY
    then return; fi
    if command -v tmux >/dev/null; then
        # Detached sessions survive closing the launching terminal.
        local session="worker-$name" launch logfile
        if tmux -L tenderpilot has-session -t "$session" 2>/dev/null; then return; fi
        printf -v launch '%q ' "$@"
        printf -v logfile '%q' "$PWD/logs/local/$name.log"
        tmux -L tenderpilot new-session -d -s "$session" -c "$PWD" "exec $launch >>$logfile 2>&1"
        echo "$name started in $session"
    else
        nohup "$@" >>"logs/local/$name.log" 2>&1 </dev/null &
        echo "$name PID $!"
    fi
}
exec 9>logs/local/scheduler.lock
flock -n 9 || exit 1
# Child workers must not inherit the scheduling lock.
if [[ $mode == collect || $mode == resume ]]; then
    start collect "$py" -u scripts/collect.py 9>&-
    start collect_culr "$py" -u scripts/collect.py --config configs/crawler_culr.yaml 9>&-
    start collect_beijing "$py" -u scripts/collect.py --config configs/crawler_beijing.yaml 9>&-
fi
if [[ $mode == process || $mode == resume ]]; then
    start process "$py" scripts/process.py --watch 9>&-
fi
if [[ $mode == annotate || $mode == resume ]]; then
    test -x "$layout_py"
    start layout "$layout_py" scripts/layout.py --watch 9>&-
    start queue "$py" scripts/local_queue.py --watch 9>&-
    xiaomi_py=${XIAOMI_PYTHON:-../TDA/.venv/bin/python}
    start xiaomi_4070 "$xiaomi_py" -u scripts/cloud_ocr.py --config configs/xiaomi_local.yaml --ready data/local_ocr/ready --output data/tables/predictions 9>&-
    start packets "$py" scripts/stage_packets.py 9>&-
fi
