# Wh0

Wh0 learns manipulation policies from synthetic hand-centric video data. The repo has two main components:

[Project page](https://chenyt31.github.io/wh0.github.io/) | [Paper](https://arxiv.org/abs/2606.22136)

If you are using a code agent, read the workflow below first, then ask the agent to install/validate the environment and run the default workflow: 10 synthetic videos, 100-step fine-tuning with a saved checkpoint, whole-episode render-hand visualizations, prediction-vs-GT videos, and a final list of output paths.

| Component | Directory | Role |
|-----------|-----------|------|
| WM-H | [`WM-H/`](WM-H/) | Synthetic manipulation video generation |
| Policy | [`vitra-wh0/`](vitra-wh0/) | VITRA-based training, inference, and evaluation |

![Wh0 teaser](assets/images/teaser.png)

## End-to-End Workflow

```text
WM-H instructions
  -> scene image edit
  -> Qwen-VL prompt augment
  -> Wan I2V video generation
  -> HaWoR annotation
  -> optional robot-hand video edit
  -> VITRA training tree + indices
  -> visualization / training / evaluation
```

Typical stage order:

1. Configure environment, weights, and paths in `configs/project_request.yaml`.
2. Generate WM-H videos with `scripts/run_wmh.sh` or `scripts/run_all.sh --stage wmh`.
3. Annotate generated videos with HaWoR using `scripts/run_annotate.sh`.
4. Optionally edit videos with robot hands using Qwen-Image-Edit Lightning LoRA. This is an independent stage after video generation; the default edits every 4 frames and does not add a second hand for single-hand tasks.
5. Build a VITRA `WM-H` training tree; by default 20% of available clips link to edited videos.
6. Visualize WM-H or G1 samples for quick inspection.
7. Build dataset indices and run VITRA training or evaluation.

Generated WM-H runs live under `WM-H/database/wm-h/instr_first/streaming_runs/<run_id>/`. Multi-GPU WM-H outputs are automatically merged into `<run_id>_merged/` before annotation and training. A prepared training tree is written to `<run_id>/vitra_training_data/`.

The G1 dataset mixed into policy training is collected with Unitree's [`xr_teleoperate`](https://github.com/unitreerobotics/xr_teleoperate) stack on a G1 + Inspire hand platform; deployment uses the same checkout under `vitra-wh0/thirdparty/xr_teleoperate`.

## Quick Start

Requirements: Linux, CUDA, [uv](https://docs.astral.sh/uv/), configured model weights, and at least one 80GB NVIDIA GPU for the main generation/training workflows.

Clone the repository, enter the repo root, then run the guided config flow:

```bash
git clone https://github.com/chenyt31/Wh0.git
cd Wh0
```

Use the guided config flow:

```bash
bash scripts/agent_run.sh --reconfigure
bash scripts/agent_run.sh --run quick
bash scripts/agent_run.sh --run all
```

The default local config runs 10 WM-H videos, uses the verified FP8/vLLM environment for Qwen3-VL, keeps Qwen3-VL FP8 and Wan resident on 80GB GPUs, releases the scene-image editor after each edit batch, edits every 4th frame for robot-hand variants, selects edited clips with 20% probability when building WM-H training links, mixes in G1 via `WM-H_50k`, and fine-tunes for 100 steps with a saved checkpoint. The run-local training config is written to `<run_id>/vitra_training_config.json`.

`--run all` performs setup, weight linking/download, and the workflow. If the environment and weights are already ready, use `bash scripts/agent_run.sh --run run` to rerun only the configured workflow.
Set `paths.input_path` to an existing WM-H run directory to resume the full pipeline from annotation without regenerating videos.

Or run manually:

```bash
$EDITOR configs/project_request.yaml
bash scripts/setup.sh
bash scripts/run_from_config.sh weights
bash scripts/run_from_config.sh run
```

The root `pyproject.toml` is the canonical Python environment. The tested policy-training stack is `torch==2.6.0+cu124`. WM-H Qwen3-VL FP8 generation can use a separate vLLM environment; set `WMH_PYTHON` and prepend that environment `bin/` directory to `PATH` when you use one.

## Repository Layout

```text
Wh0/
├── scripts/              # User-facing shell entrypoints
├── tools/                # Python implementations, validation, maintenance
├── libs/hand_recon/      # Shared HaWoR + MoGe + MANO pipeline
├── libs/dataset_index/   # G1 and episodic index builders
├── assets/debug_eval/    # Small validation assets
├── WM-H/                 # Synthetic data generation
├── vitra-wh0/            # VITRA-based policy code
└── weights/              # External weights, symlinks, and local assets
```

`scripts/` is intentionally kept small. Put reusable Python tools, validation, dataset utilities, and maintenance code under `tools/`.

## Configuration and Weights

Edit `configs/project_request.yaml`. It controls:

- package source and CUDA settings
- optional dependency groups
- weight download/link settings
- WM-H profile and instruction count
- annotation parallelism
- hand-edit interval and robot-hand sampling probability
- training task, config path, and optional smoke-step overrides
- dataset index behavior

For open-source or fresh-machine use, start from `configs/project_request.example.yaml` and leave machine-local `weights.items.*.local_path` blank.

Download or link enabled weights:

```bash
bash scripts/download_weights.sh
```

All external assets are organized under `weights/`:

```text
weights/
├── external/       # DROID-SLAM and related files
├── hawor/          # HaWoR detector/checkpoints/config
├── MANO_RIGHT.pkl
├── mano/
├── checkpoints/    # VITRA checkpoints
└── models/         # MoGe, PaliGemma, Qwen, Wan, etc.
```

`tools/weights/manage_weights.py sync` also refreshes compatibility symlinks for `WM-H/` and `vitra-wh0/`.

## Common Commands

Validate the repo and debug assets:

```bash
bash tools/validation/validate_environment.sh quick
bash tools/validation/validate_environment.sh runtime
```

Run one WM-H sample on one 80GB GPU. The debug desktop image source is `assets/debug_eval/desktop` by default; override it with `IMAGE_DIR=...` when needed.

```bash
PATH=/path/to/wmh-vllm-env/bin:$PATH \
WMH_PYTHON=/path/to/wmh-vllm-env/bin/python \
CUDA_VISIBLE_DEVICES=0 PROFILE=default TOTAL_INSTRUCTIONS=1 BATCH_SIZE=1 \
bash scripts/run_wmh.sh
```

Annotate a generated run:

```bash
RUN_DIR=WM-H/database/wm-h/instr_first/streaming_runs/<run_id>
INPUT_PATH="$RUN_DIR" PARALLEL_K=0 WH0_DATASET_NAME=WM-H \
bash scripts/run_annotate.sh
```

Build training links with robot-hand edits selected at 20% probability:

```bash
bash scripts/run_all.sh --stage hand_edit --input-path "$RUN_DIR/videos" --hand-edit-every-n 4
bash scripts/run_all.sh --stage prepare_data --input-path "$RUN_DIR" --robot-prob 0.2
```

Visualize training/test data:

```bash
RENDER_HAND=1 MAX_EPISODES=1 bash scripts/run_all.sh --stage visualize \
  --input-path "$RUN_DIR/vitra_training_data" \
  --output-path "$RUN_DIR/visualize_wmh_render_hand"

RENDER_HAND=1 WHOLE_EPISODE=1 NUM_SAMPLES=1 bash scripts/run_all.sh --stage visualize \
  --input-path "$(pwd)/assets/debug_eval/g1_dataset" \
  --output-path "$(pwd)/validation_outputs/g1_render_hand_episode"
```

Run a standalone G1-only smoke test:

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
CONFIG=vitra/configs/robot_finetune_debug.json \
bash scripts/run_train.sh \
  --max_steps 100 \
  --batch_size 1 \
  --total_batch_size 1 \
  --num_workers 0
```

Run model inference with a side-by-side prediction/GT video:

```bash
CKPT="$(pwd)/outputs/vitra/vitra_wmh_finetune/checkpoints/<run>/checkpoints/<ckpt>/weights.pt"
CFG="$(pwd)/outputs/vitra/vitra_wmh_finetune/checkpoints/<run>/config.json"
RUN_DIR="$(pwd)/WM-H/database/wm-h/instr_first/streaming_runs/<run_id>"

bash scripts/run_eval_pipeline.sh annotation_default \
  --mode annotation \
  --video-path "$RUN_DIR/videos/<video>.mp4" \
  --annotation-npy "$RUN_DIR/episodic_annotations/<annotation>.npy" \
  --config "$CFG" \
  --model-path "$CKPT" \
  --statistics-path "$(pwd)/weights/statistics/dataset_statistics.json" \
  --output-dir "$(pwd)/validation_outputs/pred_vs_gt_wmh" \
  --output-videos-dir "$(pwd)/validation_outputs/pred_vs_gt_wmh"

bash scripts/run_eval_pipeline.sh annotation_default \
  --mode dataset \
  --root-dir "$(pwd)/assets/debug_eval/g1_dataset" \
  --sample-idx 0 \
  --config "$CFG" \
  --model-path "$CKPT" \
  --statistics-path "$(pwd)/assets/debug_eval/g1_dataset/g1_dataset_angle_statistics.json" \
  --output-dir "$(pwd)/validation_outputs/pred_vs_gt_g1" \
  --output-videos-dir "$(pwd)/validation_outputs/pred_vs_gt_g1"
```

## Script Entrypoints

| Command | Purpose |
|---------|---------|
| `scripts/setup.sh` | Install/sync the repo environment and third-party checkouts |
| `scripts/download_weights.sh` | Download or link configured weights |
| `scripts/run_from_config.sh` | Run setup/weights/workflow from `project_request.yaml` |
| `scripts/run_all.sh` | Unified stage runner |
| `scripts/run_wmh.sh` | WM-H generation wrapper |
| `scripts/run_annotate.sh` | HaWoR annotation for WM-H runs |
| `scripts/run_visualize_training_data.sh` | WM-H/G1 visualization wrapper |
| `scripts/run_train.sh` | VITRA training/eval wrapper |
| `scripts/run_eval_pipeline.sh` | Debug/eval wrapper |
| `scripts/run_human_inference.sh` | Single-image human-hand prediction wrapper |

## Notes and Troubleshooting

- WM-H generation supports one or more 80GB GPUs. Use `wmh.profile: default` for both single-card and multi-card runs; it uses all visible GPUs and keeps Qwen3-VL FP8 plus Wan resident by default.
- Use `wmh.profile: single_gpu` only as a conservative fallback when you need lower VL memory limits or explicit one-card debugging. If memory is tight, set `WM-H/configs/video.single_gpu.yaml` `settings.offload_between_stages: true`.
- Qwen3-VL FP8 is expected to run through vLLM. If a Transformers FP8 loader rejects your GPU, use a compatible vLLM environment for WM-H.
- `vllm==0.8.5` is not reliable for current Qwen3-VL in this repo; use a current vLLM environment validated for your Qwen3-VL checkpoint.
- Install a `torch-scatter` wheel matching the active Torch/CUDA stack for annotation.
- Use `RENDER_HAND=1` for full-episode hand mesh visualization. G1 visualization defaults to episode rendering only; action/state comparison is opt-in with `COMPARE_ACTION=1`.
- If training fails on missing indices, run `tools/dataset/build_g1_index.py` or `tools/dataset/build_episode_index.py`.

## Component Docs

- [WM-H/README.md](WM-H/README.md): synthetic data generation, GPU profiles, and training-tree preparation.
- [vitra-wh0/README.md](vitra-wh0/README.md): VITRA-based training, inference, and evaluation.

## Acknowledgements

Wh0 builds on and integrates several open-source projects and released models:

- [VITRA](https://github.com/microsoft/VITRA): VLA architecture, dataset format, checkpoints, and training code.
- [HaWoR](https://github.com/ThunderVVV/HaWoR): world-space hand reconstruction.
- [MoGe](https://github.com/microsoft/MoGe): camera and geometry estimation used by hand reconstruction.
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM): SLAM dependency used through HaWoR.
- [MANO](https://mano.is.tue.mpg.de/): parametric hand model.
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): video/image generation pipeline utilities.
- Qwen models, including Qwen-Image-Edit and Qwen3-VL, for scene editing and prompt augmentation.
- Wan 2.2 I2V and Wan Lightning LoRA for video generation.
- PaliGemma / Gemma weights used by the policy stack where configured.

Research use only. Respect upstream licenses and model terms.
