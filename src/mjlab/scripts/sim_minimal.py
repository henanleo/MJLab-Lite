"""最精简的 Motion Imitation 仿真/播放脚本"""
import argparse
import os
from pathlib import Path
from dataclasses import asdict

from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def play(motion_file: str, checkpoint: str, device: str = "cuda:0", num_envs: int = 1, viewer: str = "auto"):
    import mjlab.tasks  # noqa: F401
    
    # 加载配置
    env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=True)
    agent_cfg = load_rl_cfg("Mjlab-Tracking-Flat-Unitree-G1")
    env_cfg.scene.num_envs = num_envs
    
    # 设置motion文件
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(Path(motion_file).resolve())
    
    # 创建环境
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # 加载模型
    runner = OnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)
    
    # 选择查看器
    if viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer = "native" if has_display else "viser"
    
    if viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        print("[INFO] 使用 Viser Web 查看器，请在浏览器打开: http://localhost:8080")
        ViserPlayViewer(env, policy).run()
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Motion Imitation 仿真播放")
    parser.add_argument("--motion-file", "-m", type=str, required=True, help="Motion文件路径 (.npz)")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="模型文件路径 (.pt)")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="设备 (默认: cuda:0)")
    parser.add_argument("--num-envs", "-n", type=int, default=1, help="环境数量 (默认: 1)")
    parser.add_argument("--viewer", "-v", type=str, default="auto", choices=["auto", "native", "viser"],
                        help="查看器类型: auto(自动检测), native(本地窗口), viser(Web) (默认: auto)")
    args = parser.parse_args()
    
    play(motion_file=args.motion_file, checkpoint=args.checkpoint, device=args.device, 
         num_envs=args.num_envs, viewer=args.viewer)


if __name__ == "__main__":
    main()
