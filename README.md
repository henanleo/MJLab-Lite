# MJLab-Lite

## Introduction

This project is a streamlined version of [MJLab](https://github.com/mujocolab/mjlab), with W&B experiment tracking, ONNX export, multi-GPU distributed training, and other advanced features removed. **It focuses solely on the core Motion Imitation functionality**, making it ideal for learning and rapid experimentation.

## Features
-  **Minimal Dependencies** - Removed unnecessary dependencies for easier deployment
-  **Core Focus** - ~60 lines training script, ~70 lines simulation script

## Installation

### Requirements

- Python 3.10 - 3.13
- CUDA 11.8+ (12.x recommended)
- NVIDIA GPU

### Using uv (Recommended)

```bash
git clone https://github.com/your-username/mjlab-lite.git
cd mjlab-lite
uv sync
```

### Using pip

```bash
git clone https://github.com/your-username/mjlab-lite.git
cd mjlab-lite
pip install -e .
```

## Quick Start

### 1. Training

```bash
uv run train -m your_motion.npz
```

Training parameters:

```bash
uv run train --help

# Example
uv run train -m motion.npz -d cuda:0 -l logs/exp1
```

| Parameter       | Short | Description      | Default        |
| --------------- | ----- | ---------------- | -------------- |
| `--motion-file` | `-m`  | Motion file path | (required)     |
| `--device`      | `-d`  | Compute device   | `cuda:0`       |
| `--log-dir`     | `-l`  | Log directory    | auto-generated |

### 2. Simulation Playback

```bash
uv run sim -m your_motion.npz -c logs/xxx/model_xxx.pt
```

Simulation parameters:

```bash
uv run sim --help

# Example
uv run sim -m motion.npz -c model.pt  
```

| Parameter       | Short | Description      | Default    |
| --------------- | ----- | ---------------- | ---------- |
| `--motion-file` | `-m`  | Motion file path | (required) |
| `--checkpoint`  | `-c`  | Model file path  | (required) |
| `--device`      | `-d`  | Compute device   | `cuda:0`   |
| `--num-envs`    | `-n`  | Number of envs   | `1`        |
| `--viewer`      | `-v`  | Viewer type      | `auto`     |

Viewer types:

- `auto`: Auto-detect (uses native if display available, otherwise viser)
- `native`: Local GLFW window
- `viser`: Web viewer (server-friendly, opens http://localhost:8080)


### CSV to NPZ Conversion

If your motion data is in CSV format, use the conversion script:

```bash
uv run python -m mjlab.scripts.csv_to_npz \
    --input-file motion.csv \
    --output-name output \
    --input-fps 30 \
    --output-fps 50
```

## Acknowledgements

- [MJLab](https://github.com/mujocolab/mjlab) - Original project

