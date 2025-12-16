"""最精简的 Motion Imitation 训练脚本"""
import argparse
import os
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg


def train(motion_file: str, device: str = "cuda:0", log_dir: str | None = None):
    os.environ["MUJOCO_GL"] = "egl"
    
    import mjlab.tasks  # noqa: F401
    
    # 加载配置
    env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1")
    agent_cfg = load_rl_cfg("Mjlab-Tracking-Flat-Unitree-G1")
    
    # 设置motion文件
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(Path(motion_file).resolve())
    
    # 创建日志目录
    if log_dir is None:
        log_dir_path = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 创建环境
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # 创建Runner并训练
    runner = OnPolicyRunner(env, asdict(agent_cfg), str(log_dir_path), device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    env.close()
    print(f"训练完成! 模型保存在: {log_dir_path}")


def main():
    parser = argparse.ArgumentParser(description="Motion Imitation 训练")
    parser.add_argument("--motion-file", "-m", type=str, required=True, help="Motion文件路径 (.npz)")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="设备 (默认: cuda:0)")
    parser.add_argument("--log-dir", "-l", type=str, default=None, help="日志目录 (默认: 自动生成)")
    args = parser.parse_args()
    
    train(motion_file=args.motion_file, device=args.device, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
