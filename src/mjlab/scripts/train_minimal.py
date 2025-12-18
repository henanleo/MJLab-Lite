"""最精简的 Motion Imitation 训练脚本"""
import argparse
import os
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.os import dump_yaml


def train(
    motion_file: str,
    device: str = "cuda:0",
    log_dir: str | None = None,
    num_envs: int = 4096, 
):
    import mjlab.tasks  # noqa: F401
    
    configure_torch_backends()
    
    # 设置环境变量
    os.environ["MUJOCO_GL"] = "egl"
    if device.startswith("cuda"):
        gpu_id = device.split(":")[-1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        device = "cuda:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # 加载配置
    task_id = "Mjlab-Tracking-Flat-Unitree-G1"
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    
    env_cfg.scene.num_envs = num_envs
    print(f"[INFO] Training with {num_envs} parallel environments")
    
    seed = agent_cfg.seed
    env_cfg.seed = seed
    agent_cfg.seed = seed
    
    # 设置motion文件
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(Path(motion_file).resolve())
    print(f"[INFO] Using motion file: {motion_file}")
    
    # 创建日志目录
    if log_dir is None:
        log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
        log_dir_path = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir_path}")
    
    # 创建环境
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = OnPolicyRunner
    
    agent_cfg_dict = asdict(agent_cfg)
    env_cfg_dict = asdict(env_cfg)
    
    runner = runner_cls(env, agent_cfg_dict, str(log_dir_path), device)
    runner.add_git_repo_to_log(__file__)
    
    # 保存配置文件
    dump_yaml(log_dir_path / "params" / "env.yaml", env_cfg_dict)
    dump_yaml(log_dir_path / "params" / "agent.yaml", agent_cfg_dict)
    
    # 训练
    print(f"[INFO] Training with device={device}, seed={seed}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    env.close()
    print(f"[INFO] 训练完成! 模型保存在: {log_dir_path}")


def main():
    parser = argparse.ArgumentParser(description="Motion Imitation 训练")
    parser.add_argument("--motion-file", "-m", type=str, required=True, help="Motion文件路径 (.npz)")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="设备 (默认: cuda:0)")
    parser.add_argument("--log-dir", "-l", type=str, default=None, help="日志目录 (默认: 自动生成)")
    parser.add_argument("--num-envs", "-n", type=int, default=4096, help="并行环境数量 (默认: 4096)")
    args = parser.parse_args()
    
    train(
        motion_file=args.motion_file,
        device=args.device,
        log_dir=args.log_dir,
        num_envs=args.num_envs,
    )


if __name__ == "__main__":
    main()
