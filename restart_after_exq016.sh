#!/bin/bash
# Waits for EXQ-016 to leave 'running' status, then restarts the runner
# so EXQ-017 and EXQ-018 get the fixed RE_RUN_DONE regex.
STATUS=/Users/dgolden/Documents/GitHub/REE_assembly/evidence/experiments/runner_status.json
RUNNER_DIR=/Users/dgolden/Documents/GitHub/ree-v2

echo "[watcher] Waiting for EXQ-016 to complete..."
while true; do
    current=$(python3 -c "import json; s=json.load(open('$STATUS')); print((s.get('current') or {}).get('queue_id','idle'))" 2>/dev/null)
    if [ "$current" != "EXQ-016" ]; then
        echo "[watcher] EXQ-016 done (current=$current). Restarting runner..."
        kill $(cat $RUNNER_DIR/runner.pid) 2>/dev/null
        sleep 2
        cd $RUNNER_DIR
        nohup python3 experiment_runner.py > runner.log 2>&1 &
        echo $! > runner.pid
        echo "[watcher] Runner restarted as PID $(cat runner.pid). Fix active."
        break
    fi
    sleep 15
done
