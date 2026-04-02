"""render_policy.py – visualise a saved PPO or SAC policy with a live pygame window.

Modes
-----
single  : play one episode live in a pygame window; optionally save a GIF.
eval    : headless evaluation of all runs in a directory tree; prints average returns.

Examples
--------
# Watch a policy live (closes when the episode ends):
python notebooks/render_policy.py single \\
    --run-dir output/Hopper-v4/ppo/20260401_223554_seed1

# Same, but also save a GIF:
python notebooks/render_policy.py single \\
    --run-dir output/Hopper-v4/ppo/20260401_223554_seed1 \\
    --save-gif hopper_ppo.gif --fps 30

# Evaluate every run under an output subtree:
python notebooks/render_policy.py eval \\
    --output-dir output/Hopper-v4/ppo \\
    --trials 10
"""

import argparse
import glob
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Fix GLFW discovery on macOS / Homebrew before mujoco is imported.
# The pyglfw package only searches /usr/lib and /usr/local/lib by default;
# Homebrew installs to /opt/homebrew/lib on Apple Silicon.
# PYGLFW_LIBRARY is checked first by glfw/library.py so this is safe to set
# only when the variable is not already provided by the user.
# ---------------------------------------------------------------------------
if not os.environ.get("PYGLFW_LIBRARY"):
    _homebrew_candidates = glob.glob("/opt/homebrew/lib/libglfw.dylib")
    if _homebrew_candidates:
        os.environ["PYGLFW_LIBRARY"] = _homebrew_candidates[0]

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Network definitions (must match the training code exactly)
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)),      nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_action(self, obs):
        return self.actor_mean(obs)  # deterministic mean


class SACActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale, action_bias):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias",  action_bias)

    def get_action(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _detect_alg(run_dir: str) -> str:
    for part in Path(run_dir).parts:
        if part in ("ppo", "sac"):
            return part
    raise ValueError(f"Cannot detect algorithm from path: {run_dir}")


def load_agent(run_dir: str, device: str = "cpu"):
    """Load a PPO or SAC agent from a run directory (model.pt + config.yaml).

    Returns
    -------
    agent   : nn.Module with a ``get_action(obs_tensor)`` method
    env_id  : str
    alg     : "ppo" | "sac"
    obs_rms : dict{"mean", "var"} for PPO observation normalisation, or None
    """
    cfg_path   = Path(run_dir) / "config.yaml"
    model_path = Path(run_dir) / "model.pt"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env_id = cfg["env_id"]
    alg    = _detect_alg(run_dir)
    data   = torch.load(model_path, map_location=device, weights_only=False)

    tmp_env    = gym.make(env_id)
    obs_space  = tmp_env.observation_space
    act_space  = tmp_env.action_space
    obs_dim    = int(np.prod(obs_space.shape)) if obs_space.shape else 0
    action_dim = int(np.prod(act_space.shape)) if act_space.shape else 0  # type: ignore[union-attr]

    if alg == "ppo":
        agent = PPOAgent(obs_dim, action_dim).to(device)
        # strict=False: saved dict also contains critic / actor_logstd weights
        agent.load_state_dict(data["state_dict"], strict=False)
        obs_rms = data.get("obs_rms")  # {"mean": ..., "var": ...} or None
        tmp_env.close()
        return agent, env_id, alg, obs_rms

    else:  # sac – action_space is a Box for all MuJoCo envs
        from gymnasium.spaces import Box as BoxSpace
        assert isinstance(act_space, BoxSpace), "SAC requires a Box action space"
        a_scale = torch.tensor((act_space.high - act_space.low) / 2.0, dtype=torch.float32)
        a_bias  = torch.tensor((act_space.high + act_space.low) / 2.0, dtype=torch.float32)
        actor   = SACActorNet(obs_dim, action_dim, a_scale, a_bias).to(device)
        actor.load_state_dict(data)
        tmp_env.close()
        return actor, env_id, alg, None


# ---------------------------------------------------------------------------
# Episode rollout helpers
# ---------------------------------------------------------------------------

def _normalise_obs(obs, obs_rms):
    obs = (obs - obs_rms["mean"]) / np.sqrt(obs_rms["var"] + 1e-8)
    return np.clip(obs, -10.0, 10.0)


def _step_agent(agent, obs, obs_rms, device):
    if obs_rms is not None:
        obs = _normalise_obs(obs, obs_rms)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = agent.get_action(obs_t).cpu().numpy()[0]
    return action


# ---------------------------------------------------------------------------
# pygame live-window playback
# ---------------------------------------------------------------------------

def play_episode_pygame(agent, env_id: str, obs_rms, device: str, fps: int, max_steps: int,
                        title: str = "Policy Viewer"):
    """Play one episode in a live pygame window.

    Returns (frames, total_reward). The window stays open until the episode
    ends or the user closes it (pressing Q or the window X button).

    Frames are collected as RGB arrays so a GIF can be saved afterwards.
    """
    import pygame  # imported here so the module works without pygame on eval-only runs

    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset()

    # Grab the first frame to find the resolution
    first_frame: np.ndarray = env.render()  # type: ignore[assignment]
    assert first_frame is not None, "env.render() returned None – check render_mode"
    h, w = first_frame.shape[:2]

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(title)
    clock  = pygame.time.Clock()

    frames: list       = [first_frame]
    total_reward: float = 0.0
    running = True

    # Act on the first observation before we already have its frame
    action = _step_agent(agent, obs, obs_rms, device)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += float(reward)
    done = terminated or truncated

    step = 1
    while running and step < max_steps and not done:
        # Handle window close / Q key
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        if not running:
            break

        # Render + display
        frame: np.ndarray = env.render()  # type: ignore[assignment]
        frames.append(frame)

        # pygame expects (W, H, 3) surface from a (H, W, 3) array
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(fps)

        # Step
        action = _step_agent(agent, obs, obs_rms, device)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        step += 1

    env.close()
    pygame.quit()
    return frames, total_reward


# ---------------------------------------------------------------------------
# GIF saving
# ---------------------------------------------------------------------------

def save_gif(frames: list, path: str, fps: int = 30) -> None:
    """Save a list of RGB frames as an animated GIF using imageio."""
    import imageio  # optional; only needed when --save-gif is used

    duration_ms = 1000 / fps
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), [f.astype(np.uint8) for f in frames], duration=duration_ms, loop=0)
    print(f"  Saved GIF -> {out.resolve()}  ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Headless rollout (used by eval mode – no window)
# ---------------------------------------------------------------------------

def rollout_headless(agent, env_id: str, obs_rms, device: str, max_steps: int):
    """Run one episode without rendering; return total reward."""
    env = gym.make(env_id)
    obs, _ = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        action = _step_agent(agent, obs, obs_rms, device)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    env.close()
    return total_reward


# ---------------------------------------------------------------------------
# CLI command handlers
# ---------------------------------------------------------------------------

def cmd_single(args) -> None:
    """Play one run live in pygame; optionally save a GIF."""
    print(f"Loading agent from: {args.run_dir}")
    agent, env_id, alg, obs_rms = load_agent(args.run_dir, device=args.device)
    print(f"  env={env_id}  alg={alg}")
    print(f"  Press Q or close the window to stop early.")

    title = f"{env_id} – {alg.upper()} – {Path(args.run_dir).name}"
    frames, reward = play_episode_pygame(
        agent, env_id, obs_rms,
        device=args.device,
        fps=args.fps,
        max_steps=args.max_steps,
        title=title,
    )
    print(f"  Episode return: {reward:.2f}  ({len(frames)} frames)")

    if args.save_gif:
        save_gif(frames, args.save_gif, fps=args.fps)


def cmd_eval(args) -> None:
    """Headless evaluation of all runs found under --output-dir."""
    base     = Path(args.output_dir)
    run_dirs = sorted([p.parent for p in base.rglob("model.pt")])

    if not run_dirs:
        print(f"No model.pt files found under {base}")
        return

    results: dict = {}

    for run_dir in run_dirs:
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            print(f"  Skipping {run_dir} (no config.yaml)")
            continue

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        print(f"\nLoading: {run_dir.name}")
        agent, env_id, alg, obs_rms = load_agent(str(run_dir), device=args.device)

        reward_sum = 0.0
        for trial in range(1, args.trials + 1):
            reward = rollout_headless(agent, env_id, obs_rms, device=args.device, max_steps=args.max_steps)
            reward_sum += reward
            print(f"  trial {trial}/{args.trials}  return={reward:.2f}")

        avg = reward_sum / args.trials

        # Key by whichever hyperparams exist in this config
        key_items = {k: cfg[k] for k in ("clip_coef", "gae_lambda", "gamma", "tau") if k in cfg}
        key = tuple(sorted(key_items.items()))
        results[key] = avg
        print(f"  -> avg over {args.trials} trials: {avg:.2f}  {dict(key_items)}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for key, avg in sorted(results.items()):
        print(f"  {dict(key):50s}  avg_return={avg:.2f}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualise a saved PPO or SAC policy using a live pygame window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- single ------------------------------------------------------------
    p_single = sub.add_parser(
        "single",
        help="Play one episode live in a pygame window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_single.add_argument(
        "--run-dir", required=True,
        help="Path to a run directory containing model.pt and config.yaml.",
    )
    p_single.add_argument(
        "--save-gif", default=None, metavar="PATH",
        help="If given, also save the episode as an animated GIF at this path.",
    )
    p_single.add_argument("--fps",       type=int, default=30,   help="Playback speed (frames per second).")
    p_single.add_argument("--max-steps", type=int, default=1000, help="Maximum steps before the episode is cut off.")
    p_single.add_argument("--device",    default="cpu",          help="PyTorch device (cpu / cuda / mps).")
    p_single.set_defaults(func=cmd_single)

    # -- eval --------------------------------------------------------------
    p_eval = sub.add_parser(
        "eval",
        help="Headless evaluation of all runs under an output directory tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_eval.add_argument(
        "--output-dir", required=True,
        help="Root directory to search recursively for run directories.",
    )
    p_eval.add_argument("--trials",     type=int, default=5,    help="Evaluation episodes per run.")
    p_eval.add_argument("--max-steps",  type=int, default=1000, help="Maximum steps per episode.")
    p_eval.add_argument("--device",     default="cpu",          help="PyTorch device (cpu / cuda / mps).")
    p_eval.set_defaults(func=cmd_eval)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
