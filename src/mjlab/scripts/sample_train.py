"""
Minimal RSL-RL training script (single GPU / CPU)
保留 train.py 的核心训练功能，移除多GPU、视频录制等高级功能
"""

import os
from pathlib import Path
from dataclasses import asdict

from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


def train(
    task_id: str,
    motion_file: str | None = None,
    device: str = "cuda:0",
    log_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
    enable_nan_guard: bool = False,
):
    """
    精简的训练函数，保留核心功能：
    - ✓ 任务配置加载
    - ✓ Motion文件处理（tracking任务）
    - ✓ 环境创建和VecEnv包装
    - ✓ Runner创建和训练
    - ✓ 模型自动保存（runner.learn()自动处理，每save_interval个iteration保存一次）
    - ✓ Checkpoint恢复
    - ✓ 配置文件保存
    - ✓ Git仓库记录
    - ✓ NaN guard（可选）
    
    移除的功能：
    - ✗ 多GPU支持（torchrunx）
    - ✗ 视频录制
    - ✗ W&B checkpoint加载
    - ✗ 复杂的命令行参数解析
    
    模型保存说明：
    - 模型会在训练过程中自动保存到 log_dir 目录
    - 保存间隔由 agent_cfg.save_interval 控制（默认50个iteration）
    - 保存文件名格式: model_<iteration>.pt
    - 例如: model_50.pt, model_100.pt, model_150.pt 等
    """
    # --------------------------------------------------
    # 1. 基础环境配置
    # --------------------------------------------------
    configure_torch_backends()
    
    # 设置MuJoCo环境变量（重要！）
    os.environ["MUJOCO_GL"] = "egl"
    
    # 设置CUDA设备
    if device.startswith("cuda"):
        gpu_id = device.split(":")[-1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        device = f"cuda:0"  # 重新映射为cuda:0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)

    seed = agent_cfg.seed
    env_cfg.seed = seed
    agent_cfg.seed = seed
    
    print(f"[INFO] Training with: device={device}, seed={seed}")

    # --------------------------------------------------
    # 2. Tracking任务：绑定本地motion文件
    # --------------------------------------------------
    is_tracking_task = (
        env_cfg.commands is not None
        and "motion" in env_cfg.commands
        and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    )
    
    if is_tracking_task:
        if motion_file is None:
            raise ValueError("Must provide motion_file for tracking tasks.")

        motion_path = Path(motion_file)
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        assert env_cfg.commands is not None
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.motion_file = str(motion_path.resolve())
        print(f"[INFO]: Using motion file from local path: {motion_file}")

    # --------------------------------------------------
    # 3. NaN guard（可选）
    # --------------------------------------------------
    if enable_nan_guard:
        env_cfg.sim.nan_guard.enabled = True
        print(f"[INFO] NaN guard enabled, output dir: {env_cfg.sim.nan_guard.output_dir}")

    # --------------------------------------------------
    # 4. 创建日志目录
    # --------------------------------------------------
    if log_dir is None:
        from datetime import datetime
        log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
        log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir_name += f"_{agent_cfg.run_name}"
        log_dir_path = log_root / log_dir_name
    else:
        log_dir_path = Path(log_dir)
    
    log_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir_path}")

    # --------------------------------------------------
    # 5. 创建环境
    # --------------------------------------------------
    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode=None,  # 精简版不支持视频录制
    )

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    # --------------------------------------------------
    # 6. Runner创建
    # --------------------------------------------------
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = OnPolicyRunner

    agent_cfg_dict = asdict(agent_cfg)
    env_cfg_dict = asdict(env_cfg)

    runner = runner_cls(
        env,
        agent_cfg_dict,
        str(log_dir_path),
        device,
    )

    # Git仓库记录（重要：用于复现）
    runner.add_git_repo_to_log(__file__)
    
    # --------------------------------------------------
    # 7. Checkpoint恢复（如果提供）
    # --------------------------------------------------
    if resume_path is not None:
        resume_path_obj = Path(resume_path)
        if not resume_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path_obj))

    # --------------------------------------------------
    # 8. 保存配置文件（重要：用于复现）
    # --------------------------------------------------
    dump_yaml(log_dir_path / "params" / "env.yaml", env_cfg_dict)
    dump_yaml(log_dir_path / "params" / "agent.yaml", agent_cfg_dict)

    # --------------------------------------------------
    # 9. 开始训练（模型会自动保存）
    # --------------------------------------------------
    # runner.learn() 会根据 agent_cfg.save_interval 自动保存模型checkpoint
    # 默认每50个iteration保存一次，保存为 model_<iteration>.pt
    # 保存路径: log_dir_path / "model_<iteration>.pt"
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    print(f"[INFO] Training completed! Models saved in: {log_dir_path}")
    print(f"[INFO] Checkpoint files: model_*.pt (saved every {agent_cfg.save_interval} iterations)")


def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="精简的训练脚本 - 单GPU/CPU训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # Velocity任务（不需要motion文件）
  uv run python -m mjlab.scripts.sample_train Mjlab-Velocity-Flat-Unitree-G1
  
  # Tracking任务（需要motion文件）
  uv run python -m mjlab.scripts.sample_train Mjlab-Tracking-Flat-Unitree-G1 --motion-file assets/motions/walk.npz
  
  # 指定设备
  uv run python -m mjlab.scripts.sample_train Mjlab-Velocity-Flat-Unitree-G1 --device cuda:0
  
  # 从checkpoint恢复训练
  uv run python -m mjlab.scripts.sample_train Mjlab-Velocity-Flat-Unitree-G1 --resume-path logs/rsl_rl/exp/model_1000.pt
        """
    )
    
    parser.add_argument(
        "task_id",
        type=str,
        help="任务ID，例如: Mjlab-Velocity-Flat-Unitree-G1"
    )
    
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Motion文件路径（tracking任务必需）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="训练设备，例如: cuda:0, cuda:1, cpu (默认: cuda:0)"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="日志目录路径（默认自动生成）"
    )
    
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Checkpoint文件路径（用于恢复训练）"
    )
    
    parser.add_argument(
        "--enable-nan-guard",
        action="store_true",
        help="启用NaN guard"
    )
    
    args = parser.parse_args()
    
    # 导入任务注册表
    import mjlab.tasks  # noqa: F401
    
    train(
        task_id=args.task_id,
        motion_file=args.motion_file,
        device=args.device,
        log_dir=args.log_dir,
        resume_path=args.resume_path,
        enable_nan_guard=args.enable_nan_guard,
    )


if __name__ == "__main__":
    main()
