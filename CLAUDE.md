# Claude Code Notes

When launching long-running training or experiment jobs in this repo:

- Always run them in the background with tmux so they continue if the terminal dies.
- Always send logs, summaries, and other run artifacts to the `result/` directory.
- Always activate the local virtual environment first with `source .venv/bin/activate`.

Recommended pattern:

```bash
source .venv/bin/activate
nohup bash -lc '<command>' > result/<run>.log 2>&1 < /dev/null &
```

Sweep scripts should live in the repo root, not under `result/`.
