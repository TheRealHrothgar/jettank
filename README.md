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

`provider=none` runs fully local — useful before the cloud data-path question is
settled, since camera frames leaving the device is a policy decision.

Note the cloud model only ever receives **text observations** from the local
VLM, never raw images. That keeps imagery on-device by default.

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

- [x] Loop, clients and config — 18/18 logic tests pass
- [ ] Jetson reachable (blocked: `nv-oobe` first-boot wizard needs a display)
- [ ] Local VLM pulled and benchmarked against measured free RAM
- [ ] Yahboom SDK identified; `NullRobot` replaced with the real driver
- [ ] Cloud provider chosen
