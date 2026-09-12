# Jettank — local VLM ↔ cloud LLM loop

Perception and reasoning for a Yahboom Jettank on a **Jetson Orin Nano Super (8 GB)**.

## Design

Two rates, deliberately decoupled:

| Loop | Runs | Rate | Blocking? |
|---|---|---|---|
| **fast** | camera → local VLM → observation | ~1.5 s, on-device | never waits on network |
| **slow** | observations → cloud LLM → plan | ≥6 s, background task | best-effort; failure is a no-op |

The fast loop owns the robot. The cloud only ever *suggests* — if it is slow,
offline, or returns nonsense, the robot keeps perceiving and stays responsive.
Every network path returns `None` on failure rather than raising.

## Working directories

Two trees, deliberately separate:

| Path | Role |
|---|---|
| `~/jettank` | **operating tree** — what runs, what deploys to the robot. Edit here. |
| `~/git/jettank` | **git repo** — what gets published. Never edit directly; it is overwritten. |

The operating tree accumulates things that must never be published: the venv,
61 MB Piper voice models, the face database, logs. Keeping the repo a separate
directory means a stray file cannot be swept in by an over-broad `git add`.

```bash
./scripts/sync_git.sh      # operating tree -> git repo (refuses if it finds a secret)
./scripts/deploy.sh        # operating tree -> the robot
./scripts/deploy.sh --test # ...and run the test suite there
```

`sync_git.sh` greps the source for credential-*shaped* strings (`sk-ant-…`,
`ghp_…`, `BEGIN … PRIVATE KEY`) and aborts before copying anything. It matches
the shape of real credentials, not the words "key" or "password", so config
field names and documentation do not trip it.

**Edit in the operating tree, never in the repo.** `sync_git.sh` uses
`rsync --delete`, so a change made only in `~/git/jettank` is silently
reverted on the next sync.

## Secrets

Nothing secret is in either tree:

| What | Where | Mode |
|---|---|---|
| Cloud API key | `~/.jettank.env` | 600 |
| GitHub / GitLab PATs, Wi-Fi password | `~/git/.env` | 600 |
| SSH key for the robot | `~/.ssh/jetson_ed25519` | 600 |
| Face embeddings | `~/.jettank_faces.json` **on the robot** | 600 |

All sit outside every git repo. `~/.jettank.env` is sourced by `run.sh` at
runtime and is never deployed — the robot keeps its own copy. Face embeddings
never leave the device; the cloud only ever sees names and confidences.

## Layout

    jettank/config.py   env-driven config, no secrets at import
    jettank/camera.py   threaded capture, always the newest frame
    jettank/vlm.py      local VLM over an Ollama-compatible HTTP API
    jettank/cloud.py    provider-agnostic cloud client (anthropic | openai | none)
    jettank/robot.py    actuator interface + NullRobot until the Yahboom SDK is wired
    jettank/loop.py     the orchestrator
    scripts/            provisioning + first-boot recovery
    tests/              logic tests using a stubbed transport

## Model sizing

The Orin Nano Super shares 8 GB LPDDR5 between CPU and GPU, so **RAM is the
binding constraint, not disk**. That comfortably fits ~3B-class VLMs
(`qwen2.5vl:3b`, VILA-1.5-3B, Gemma-3-4B). Run headless
(`systemctl set-default multi-user.target`) — the desktop session costs roughly
0.8–1.5 GB, which is the difference between a 3B model fitting comfortably and
fighting the OOM killer.

## Setup

    ./scripts/setup_jetson.sh
    ./.venv/bin/python -m jettank.loop --once -v     # smoke test
    ./.venv/bin/python -m jettank.loop               # run the loop

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `JETTANK_CAMERA` | `0` | V4L2 index or device path |
| `JETTANK_VLM_URL` | `http://127.0.0.1:11434` | local model server |
| `JETTANK_VLM_MODEL` | `qwen2.5vl:3b` | local VLM |
| `JETTANK_VLM_INTERVAL` | `1.5` | seconds between local inferences |
| `JETTANK_CLOUD_PROVIDER` | `none` | `anthropic`, `openai`, or `none` |
| `JETTANK_CLOUD_URL` | provider default | base URL (any OpenAI-compatible gateway) |
| `JETTANK_CLOUD_MODEL` | `claude-opus-5` | cloud model id |
| `JETTANK_CLOUD_KEY_ENV` | `ANTHROPIC_API_KEY` | env var holding the key |
| `JETTANK_CLOUD_MIN_INTERVAL` | `6.0` | escalation rate limit |
| `JETTANK_CLOUD_SEND_FRAMES` | `true` | send the camera frame to the cloud model, not just text |

`provider=none` runs fully local — useful before the cloud data-path question is
settled, since camera frames leaving the device is a policy decision.

### What the cloud model receives

By default the cloud model gets **both** the local VLM's text observations and
the current camera frame (`JETTANK_CLOUD_SEND_FRAMES=true`). Sending the frame
measurably improves its reasoning — it picks up spatial detail the local model
never wrote down.

Set `JETTANK_CLOUD_SEND_FRAMES=false` to send text only, which keeps all imagery
on-device. That is the right setting if camera frames leaving the robot is a
policy problem; the loop works either way.

Frames are sent as base64 with the media type sniffed from the image header —
`image/png` or `image/jpeg` as appropriate.

## Tests

    python3 tests/test_loop.py

Runs without `httpx` installed by stubbing the transport.

## Running locally (no Jetson, no root)

This box has no `pip`/`venv` and no passwordless `sudo`, so dependencies were
fetched with `apt-get download` + `dpkg-deb -x` (see the `wsl-no-root-tooling`
note). `./run.sh` sets the resulting `PYTHONPATH` for you.

    # start the local model server
    ~/.local/ollama/bin/ollama serve &
    ~/.local/ollama/bin/ollama pull qwen2.5vl:3b

    # one pass against a synthetic scene, no camera needed
    ./run.sh --once -v --image testdata/scene.png

    # continuous loop for 60s
    ./run.sh -v --image testdata/scene.png --duration 60

To enable the cloud half:

    export ANTHROPIC_API_KEY=sk-...
    export JETTANK_CLOUD_PROVIDER=anthropic
    ./run.sh --once -v --image testdata/scene.png

Any OpenAI-compatible gateway works instead:

    export JETTANK_CLOUD_PROVIDER=openai
    export JETTANK_CLOUD_URL=https://your-gateway
    export JETTANK_CLOUD_KEY_ENV=YOUR_KEY_VAR

## Status

- [x] Loop, clients and config — 35/35 logic tests pass
- [x] Local VLM live — `qwen2.5vl:3b` real inference (~2-5 s warm on laptop CPU)
- [x] Cloud LLM live — `claude-opus-5` over `api.anthropic.com`, plans parsed
- [x] Full loop cycling continuously, with cloud reasoning over temporal context
- [x] Multimodal — camera frames sent to the cloud model
- [ ] Jetson reachable (blocked: `nv-oobe` first-boot wizard needs a power-cycle)
- [ ] Local VLM benchmarked against measured free RAM on-device
- [ ] Yahboom SDK identified; `NullRobot` replaced with the real driver
