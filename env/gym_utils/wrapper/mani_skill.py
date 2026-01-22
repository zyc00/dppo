"""
Environment wrapper for ManiSkill environments with partial reset support.

Inherits from ManiSkill's FrameStack and FlattenRGBDObservationWrapper and adds:
- Partial reset support for GPU vectorized environments
- Customizable observation flattening
- Observation/action normalization
"""

import json
import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from mani_skill.utils.wrappers.frame_stack import FrameStack
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils import common

log = logging.getLogger(__name__)


class FlattenObservationWrapper(FlattenRGBDObservationWrapper):
    """Customizable observation flattening wrapper.

    Inherits from ManiSkill's FlattenRGBDObservationWrapper and allows:
    - Customizing which state keys to include
    - Customizing RGB/depth inclusion
    - Processing observations to match policy input format

    Args:
        env: The environment to wrap
        rgb: Whether to include rgb images
        depth: Whether to include depth images
        state: Whether to include state data
        sep_depth: Whether to separate depth and rgb images
        state_keys: List of state keys to include. If None, includes all.
    """

    def __init__(
        self,
        env,
        rgb: bool = True,
        depth: bool = False,
        state: bool = True,
        sep_depth: bool = True,
        state_keys: list[str] | None = None,
    ) -> None:
        self.state_keys = state_keys
        super().__init__(env, rgb=rgb, depth=depth, state=state, sep_depth=sep_depth)

    def observation(self, observation: dict):
        """Process observation with customizable state keys."""
        sensor_data = observation.pop("sensor_data")
        if "sensor_param" in observation:
            del observation["sensor_param"]

        rgb_list: list[torch.Tensor] = []
        depth_list: list[torch.Tensor] = []
        for cam_data in sensor_data.values():
            if self.include_rgb and "rgb" in cam_data:
                rgb_list.append(cam_data["rgb"])
            if self.include_depth and "depth" in cam_data:
                depth_list.append(cam_data["depth"])

        rgb_tensor: torch.Tensor | None = None
        depth_tensor: torch.Tensor | None = None
        if len(rgb_list) > 0:
            rgb_tensor = torch.concat(rgb_list, dim=-1)
        if len(depth_list) > 0:
            depth_tensor = torch.concat(depth_list, dim=-1)

        # Filter state keys if specified
        if self.state_keys is not None:
            filtered_obs = {}
            for key in self.state_keys:
                if key in observation:
                    filtered_obs[key] = observation[key]
                # Handle nested keys like "agent/qpos"
                elif "/" in key:
                    parts = key.split("/")
                    val = observation
                    for part in parts:
                        if isinstance(val, dict) and part in val:
                            val = val[part]
                        else:
                            val = None
                            break
                    if val is not None:
                        filtered_obs[key] = val
            observation = filtered_obs

        # Flatten the state data
        state_obs = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )

        ret: dict = {}
        if self.include_state:
            # NOTE: only qpos now
            ret["state"] = state_obs[..., :9].contiguous() # type: ignore
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = rgb_tensor.permute(0, 3, 1, 2).contiguous() # type: ignore
        elif self.include_rgb and self.include_depth:
            if self.sep_depth:
                ret["rgb"] = rgb_tensor.permute(0, 3, 1, 2).contiguous() # type: ignore
                ret["depth"] = depth_tensor.permute(0, 3, 1, 2).contiguous() # type: ignore
            else:
                assert rgb_tensor is not None and depth_tensor is not None
                ret["rgbd"] = torch.concat([rgb_tensor, depth_tensor], dim=-1)
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = depth_tensor
        return ret


class FrameStackWithPartialReset(FrameStack):
    """FrameStack wrapper that supports partial resets and multi-step actions.

    ManiSkill's FrameStack raises an error on partial reset. This subclass
    handles partial resets by detecting done envs and resetting their frame
    history while preserving the history of other envs.

    Also supports multi-step action execution for diffusion policies.

    Designed for use with ManiSkillVectorEnv(ignore_terminations=False):
    - Preserves proper terminated/truncated signals for GAE computation
    - Explicitly calls partial reset for done envs
    - Returns the new episode's first observation after reset

    Args:
        env: The environment to wrap
        num_stack: Number of frames to stack
        n_action_steps: Number of action steps to execute per call (default 1)
        reward_agg_method: Method to aggregate rewards ("sum", "mean", "max", "min")
    """

    def __init__(
        self,
        env,
        num_stack: int,
        n_action_steps: int = 1,
        reward_agg_method: str = "sum",
        sparse_reward: bool = False,
    ):
        super().__init__(env, num_stack)
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.sparse_reward = sparse_reward

    @property
    def n_obs_steps(self):
        """Alias for num_stack for compatibility."""
        return self.num_stack

    def step(self, action):
        """Step with partial reset support and multi-step action execution.

        Args:
            action: Actions of shape (num_envs, n_action_steps, action_dim)
                    or (num_envs, action_dim) for single step

        Returns:
            obs: Stacked observation after all steps (reset obs for done envs)
            reward: Aggregated reward across all steps
            terminated: Whether any env terminated during the steps (for GAE: don't bootstrap)
            truncated: Whether any env was truncated during the steps (for GAE: bootstrap)
            info: Info dict from last step
        """
        # Handle single-step action
        if action.ndim == 2:
            return self._single_step(action)

        # Multi-step execution
        n_steps = action.shape[1]
        rewards = []
        terminated_list = []
        truncated_list = []
        terminated = torch.full((self.num_envs,), False, dtype=torch.bool, device=self.env.device)

        for step_idx in range(n_steps):
            # Get action for this step: (num_envs, action_dim)
            act = action[:, step_idx, :]

            obs, reward, terminated_step, truncated, info = self._single_step(act)
            if self.sparse_reward:
                reward[terminated] = 0.0
            terminated = terminated | terminated_step
            rewards.append(reward)
            terminated_list.append(terminated_step)
            truncated_list.append(truncated)

        # Aggregate rewards
        total_reward = self._aggregate_rewards(rewards)

        # Aggregate terminated/truncated: True if any step was done
        terminated_any = self._aggregate_done(terminated_list)
        truncated_any = self._aggregate_done(truncated_list)

        return obs, total_reward, terminated_any, truncated_any, info

    def _single_step(self, action):
        """Execute a single environment step with partial reset support.

        Supports two modes based on ManiSkillVectorEnv configuration:
        - ignore_terminations=True: Auto-reset happens internally, obs after done
          is already the new episode's first obs. terminated is always False,
          truncated contains both terminated and truncated signals.
        - ignore_terminations=False: No auto-reset, we need explicit partial reset.
          (Note: requires custom env to support partial reset properly)

        For GAE, use info["success"] to determine if episode ended due to task
        completion (terminated) vs time limit (truncated only).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self.observation(None), reward, terminated, truncated, info

    def _aggregate_rewards(self, rewards):
        """Aggregate rewards from multiple steps."""
        if isinstance(rewards[0], torch.Tensor):
            rewards_stacked = torch.stack(rewards, dim=0)
            if self.reward_agg_method == "sum":
                return rewards_stacked.sum(dim=0)
            elif self.reward_agg_method == "mean":
                return rewards_stacked.mean(dim=0)
            elif self.reward_agg_method == "max":
                return rewards_stacked.max(dim=0)[0]
            elif self.reward_agg_method == "min":
                return rewards_stacked.min(dim=0)[0]
        else:
            rewards_stacked = np.stack(rewards, axis=0)
            if self.reward_agg_method == "sum":
                return rewards_stacked.sum(axis=0)
            elif self.reward_agg_method == "mean":
                return rewards_stacked.mean(axis=0)
            elif self.reward_agg_method == "max":
                return rewards_stacked.max(axis=0)
            elif self.reward_agg_method == "min":
                return rewards_stacked.min(axis=0)
        raise ValueError(f"Unknown reward aggregation method: {self.reward_agg_method}")

    def _aggregate_done(self, done_list):
        """Aggregate done signals: True if any step was done."""
        if len(done_list) == 0:
            return False
        if isinstance(done_list[0], torch.Tensor):
            stacked = torch.stack(done_list, dim=0)
            return stacked.any(dim=0)
        elif isinstance(done_list[0], np.ndarray):
            stacked = np.stack(done_list, axis=0)
            return stacked.any(axis=0)
        else:
            # Scalar bool
            return any(done_list)

    def reset(self, seed=None, options=None):
        """Reset the environment with kwargs.

        Args:
            seed: The seed for the environment reset
            options: The options for the environment reset

        Returns:
            The stacked observations
        """
        if (
            isinstance(options, dict)
            and "env_idx" in options
            and len(options["env_idx"]) < self.base_env.num_envs
        ):
            # Partial reset: only reset frames for specified env indices
            env_idx = options["env_idx"]
            obs, info = self.env.reset(seed=seed, options=options)

            # Update frames only for the reset environments
            for frame in self.frames:
                for key in frame:
                    if isinstance(frame[key], torch.Tensor):
                        frame[key][env_idx] = obs[key][env_idx]
                    elif isinstance(frame[key], np.ndarray):
                        frame[key][env_idx] = obs[key][env_idx]

            return self.observation(None), info

        # Full reset
        obs, info = self.env.reset(seed=seed, options=options)

        [self.frames.append(obs) for _ in range(self.num_stack)]

        return self.observation(None), info


class ManiSkillVectorEnvWrapper:
    """Wrapper around ManiSkillVectorEnv that adds reset_arg and reset_one_arg methods.

    This provides compatibility with the dppo training code that expects these methods.
    Also converts CUDA tensors to numpy arrays for compatibility.
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        """Forward all attribute access to wrapped env."""
        return getattr(self.env, name)

    def _obs_to_numpy(self, obs: dict) -> dict:
        """Convert observation dict values from CUDA tensors to numpy arrays."""
        return {
            key: val.cpu().numpy() if isinstance(val, torch.Tensor) else val
            for key, val in obs.items()
        }

    def reset_arg(self, options_list=None):
        """Reset all environments, ignoring options_list for now.

        ManiSkill GPU envs reset all envs together.
        """
        obs, info = self.env.reset()
        return self._obs_to_numpy(obs)

    def reset_one_arg(self, env_ind, options=None):
        """Reset a single environment by index.

        Uses partial reset with env_idx option.
        """
        obs, info = self.env.reset(options={"env_idx": [env_ind]})
        obs = self._obs_to_numpy(obs)
        # Return observation for just the reset env
        return {key: obs[key][env_ind] for key in obs}

    def reset(self, **kwargs):
        """Forward reset to wrapped env, converting obs to numpy."""
        obs, info = self.env.reset(**kwargs)
        return self._obs_to_numpy(obs), info

    def step(self, action):
        """Forward step to wrapped env, converting obs to numpy."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.env.device)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._obs_to_numpy(obs)
        # Convert reward/terminated/truncated to numpy if they're tensors
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()
        return obs, reward, terminated, truncated, info

    def seed(self, seeds):
        """ManiSkill envs handle seeding differently - this is a no-op."""
        pass


class NormalizeWrapper:
    """Wrapper that normalizes observations and unnormalizes actions using min-max normalization.

    This wrapper is used to match the normalization used during training:
    - Observations (states) are normalized to [-1, 1] before being returned
    - Actions from the policy (in [-1, 1]) are unnormalized before being sent to env

    Args:
        env: The environment to wrap
        stats_path: Path to JSON file with normalization stats (must contain min/max)
        stats: Dict with normalization stats (alternative to stats_path)
        normalize_state: Whether to normalize state observations
        unnormalize_action: Whether to unnormalize actions
        device: Device for tensor operations
    """

    def __init__(
        self,
        env,
        stats_path: Optional[str] = None,
        stats: Optional[Dict] = None,
        normalize_state: bool = True,
        unnormalize_action: bool = True,
        device: str = "cuda:0",
    ):
        self.env = env
        self.normalize_state = normalize_state
        self.unnormalize_action_flag = unnormalize_action
        self.device = device

        # Load stats
        if stats_path is not None:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            log.info(f"NormalizeWrapper: Loaded stats from {stats_path}")

        if stats is None:
            raise ValueError("Must provide either stats_path or stats dict")

        # Extract min-max normalization parameters
        self.state_min = torch.tensor(
            stats["state"]["min"], dtype=torch.float32, device=device
        )
        self.state_max = torch.tensor(
            stats["state"]["max"], dtype=torch.float32, device=device
        )
        self.action_min = torch.tensor(
            stats["action"]["min"], dtype=torch.float32, device=device
        )
        self.action_max = torch.tensor(
            stats["action"]["max"], dtype=torch.float32, device=device
        )

        log.info(f"NormalizeWrapper: state_min={self.state_min.tolist()}")
        log.info(f"NormalizeWrapper: state_max={self.state_max.tolist()}")
        log.info(f"NormalizeWrapper: action_min={self.action_min.tolist()}")
        log.info(f"NormalizeWrapper: action_max={self.action_max.tolist()}")

    def __getattr__(self, name):
        """Forward all attribute access to wrapped env."""
        return getattr(self.env, name)

    def _normalize_obs(self, obs: Dict) -> Dict:
        """Normalize the state in observation dict to [-1, 1] using min-max."""
        if not self.normalize_state or "state" not in obs:
            return obs

        obs = obs.copy()
        state = obs["state"]

        if isinstance(state, np.ndarray):
            state_min = self.state_min.cpu().numpy()
            state_max = self.state_max.cpu().numpy()
            obs["state"] = (state - state_min) / (state_max - state_min + 1e-8) * 2 - 1
        elif isinstance(state, torch.Tensor):
            obs["state"] = (state - self.state_min) / (self.state_max - self.state_min + 1e-8) * 2 - 1

        return obs

    def _unnormalize_action(
        self, action: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert normalized action from [-1, 1] back to original scale."""
        if not self.unnormalize_action_flag:
            return action

        if isinstance(action, np.ndarray):
            action_min = self.action_min.cpu().numpy()
            action_max = self.action_max.cpu().numpy()
            return (action + 1) / 2 * (action_max - action_min + 1e-8) + action_min
        elif isinstance(action, torch.Tensor):
            return (action + 1) / 2 * (self.action_max - self.action_min + 1e-8) + self.action_min
        return action

    def reset(self, **kwargs):
        """Reset and normalize observation."""
        obs, info = self.env.reset(**kwargs)
        return self._normalize_obs(obs), info

    def reset_arg(self, options_list=None):
        """Reset all environments and normalize observation."""
        obs = self.env.reset_arg(options_list)
        return self._normalize_obs(obs)

    def reset_one_arg(self, env_ind, options=None):
        """Reset single environment and normalize observation."""
        obs = self.env.reset_one_arg(env_ind, options)
        return self._normalize_obs(obs)

    def step(self, action):
        """Unnormalize action, step env, normalize observation."""
        # Unnormalize action before sending to env
        action = self._unnormalize_action(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Normalize observation
        obs = self._normalize_obs(obs)

        return obs, reward, terminated, truncated, info