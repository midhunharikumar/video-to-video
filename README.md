# V2V Release

3D camera-path visualizer + depth estimation + render-asset generation for video-to-video workflows.

## Environment (single `uv` env)

This repo targets **Python 3.10** so the same environment can support:
- visualizer UI (`viser`)
- depth estimation (`estimate_depth.py`)
- render asset generation (`render_from_cam_info.py`)
- Uni3C GPU point-rasterizer backend (`pytorch3d`)

### Prerequisites

- Python 3.10
- CUDA toolkit (matching your PyTorch build, e.g. CUDA 12.1 for `torch>=2.4`)
- `uv` package manager

### Setup

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync --group full
./scripts/install_pytorch3d.sh /path/to/Uni3C/pytorch3d   # preferred if you have Uni3C checkout
# or:
./scripts/install_pytorch3d.sh                              # builds from upstream source tag (~10–30 min)
```

`full` includes all optional groups (`depth`, `render`, `generation`, `dev`).

## Web Visualizer (React Three Fiber)

An alternative browser UI is included under `frontend/` (React Three Fiber) with a Python backend
under `backend/` (FastAPI). It reuses the same camera/export/render math as the `viser` tool and
keeps the `cam_info.json` contract compatible with `render_from_cam_info.py`.

### Features (current)

- Point-cloud playback + scrubber (looping)
- Camera keyframe placement from current view
- Keyframe transform gizmos (toggleable)
- Path interpolation preview (line + frustums)
- Framing guide HUD overlay (center box + subtle outer shading)
- `cam_info.json` export
- Render-asset generation (same outputs as `render_from_cam_info.py`)

### Run (backend + frontend)

Terminal 1 (Python backend):

```bash
source .venv/bin/activate
./run_web_backend.sh --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 (frontend):

```bash
./run_web_frontend.sh
```

Open `http://localhost:5173`.

If `npm` is missing, `run_web_frontend.sh` can bootstrap a local project Node.js + npm runtime
using `uv` + `nodeenv` (stored in `.nodeenv/`), so you can stay in a uv-only workflow.
The script defaults to a pinned Node version (`20.11.1`) to avoid `nodeenv` major-version URL issues.

Manual uv-only bootstrap (optional):

```bash
uv run --with nodeenv python -m nodeenv --prebuilt --node=20.11.1 .nodeenv
PATH="$PWD/.nodeenv/bin:$PATH" ./run_web_frontend.sh
```

If bootstrap fails (e.g. restricted access to `nodejs.org`), try a different version:

```bash
V2V_NODE_VERSION=20.18.0 ./run_web_frontend.sh
```

If your backend runs on a different host/port, start the frontend with:

```bash
VITE_API_BASE=http://localhost:8000 npm run dev
```

### One-command launch (preloaded like the `viser` app)

You can also launch backend + frontend together and preload the UI with paths:

```bash
./run_web_visualizer.sh \
  --video visualizer/test_data/test_video.mp4 \
  --depth visualizer/test_data/test_depths.npz
```

Real-data example:

```bash
./run_web_visualizer.sh \
  --video /path/to/your/input.mp4 \
  --depth /path/to/your/depths.npz
```

Useful options:
- `--max-frames 81`
- `--target-fps 16`
- `--point-subsample 4`
- `--backend-reload`
- `--open` (local desktop only; avoid on headless/SSH servers)

## Viser Auto-Depth Launcher (fallback-safe copy)

The original `visualizer/app.py` is unchanged. A new launcher module was added:
`visualizer/app_autodepth.py`.

Use it via:

```bash
./run_visualizer_autodepth.sh --video /path/to/input.mp4
```

This will:
1. estimate depth automatically (or reuse cached depth),
2. start the existing Viser point-cloud app.

UI launcher mode (no CLI video):

```bash
./run_visualizer_autodepth.sh
```

Then in the browser you can either:
- enter a **remote path** (file on the server machine), or
- **upload a local file** from your browser.

After selection, click **Estimate Depth + Start Point Cloud App**.

## Prompt Extension (optional)

`inference_wan22_v2v_local.py` and `generate.py` support optional LLM-based prompt expansion
via `--use_prompt_extend`. Two backends are available:

| Backend | Flag | Requirements |
|---|---|---|
| Local Qwen model (default) | `--prompt_extend_method local_qwen` | ~14 GB VRAM; model auto-downloaded from HuggingFace |
| DashScope API | `--prompt_extend_method dashscope` | DashScope API key |

**DashScope setup:**
1. Create an account and get an API key at https://dashscope.console.aliyun.com/
2. Export the key: `export DASH_API_KEY=<your-key>`

**Example:**
```bash
python inference_wan22_v2v_local.py \
  --use_prompt_extend \
  --prompt_extend_method dashscope \
  --prompt_extend_target_lang en \
  ...
```

## PyTorch3D / GPU Rendering Notes

PyTorch3D wheels are not available for every platform/CUDA combination, and source builds need
`torch` available first. For that reason, `pytorch3d` is installed as a second step into the same
`uv` environment (after `uv sync --group full`).

If you already have a Uni3C checkout with a working PyTorch3D source build (for example
`/path/to/Uni3C/pytorch3d`), you can install it into this environment:

```bash
uv pip install --no-build-isolation -e /path/to/Uni3C/pytorch3d
```

The renderer also checks `--uni3c-root/pytorch3d` on import, which can help when using a local
Uni3C checkout during development.
