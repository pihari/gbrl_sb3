
from typing import Any, Dict, Optional
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv

from algos.ppo import PPO_GBRL

try:
    from stable_baselines3.common.distributions import DiagGaussianDistribution
except Exception:
    DiagGaussianDistribution = object  # fallback if not available

def _flatten_grads(params):
    flats = []
    for p in params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            flats.append(th.zeros_like(p).view(-1))
        else:
            flats.append(p.grad.view(-1))
    if not flats:
        dev = params[0].device if len(params) > 0 else th.device("cpu")
        return th.zeros(0, device=dev)
    return th.cat(flats)

def _write_back_grads(params, flat: th.Tensor):
    offset = 0
    for p in params:
        if not p.requires_grad:
            continue
        n = p.numel()
        if p.grad is None:
            p.grad = th.zeros_like(p)
        p.grad.copy_(flat[offset:offset + n].view_as(p))
        offset += n

def _a_projection(policy_grad: th.Tensor,
                  cost_grad: th.Tensor,
                  cost_value: float,
                  cost_threshold: float,
                  eps: float = 1e-10) -> th.Tensor:
    """
    g* = g - λ* b,  λ* = max(0, (<g,b> - (c - d)) / ||b||^2)
    """
    if policy_grad.numel() == 0 or cost_grad.numel() == 0:
        return policy_grad
    inner = th.dot(policy_grad, cost_grad)
    violation = cost_value - cost_threshold
    bnorm2 = th.dot(cost_grad, cost_grad) + eps
    lam = (inner - violation) / bnorm2
    lam = th.clamp(lam, min=0.0)
    return policy_grad - lam * cost_grad


class SafePPO_GBRL(PPO_GBRL):
    """
    TODO: description
    """

    def __init__(self, *args, **kwargs):
        self.cost_threshold: float = float(kwargs.pop("cost_threshold", 0.0))
        self.use_safety_projection: bool = bool(kwargs.pop("use_safety_projection", True))
        self.cost_value_source = kwargs.pop("cost_value_source", None)

        super().__init__(*args, **kwargs)

        if not hasattr(self, "current_cost_estimate"):
            self.current_cost_estimate = 0.0

    def train(self) -> None:
        self.policy.set_training_mode(True)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # Optional: schedules and logging (guard attributes)
        if hasattr(self.policy, "action_dist") and isinstance(self.policy.action_dist, DiagGaussianDistribution):
            if hasattr(self.policy, "log_std_optimizer") and hasattr(self.policy, "log_std_schedule"):
                from stable_baselines3.common.utils import update_learning_rate
                update_learning_rate(self.policy.log_std_optimizer,
                                     self.policy.log_std_schedule(self._current_progress_remaining))

        if hasattr(self.policy, "nn_critic") and self.policy.nn_critic and hasattr(self.policy, "value_optimizer"):
            self._update_learning_rate(self.policy.value_optimizer)
            self.logger.record("train/nn_critic", "True")
        else:
            self.logger.record("train/nn_critic", "False")

        if hasattr(self.policy, "get_schedule_learning_rates"):
            try:
                policy_lr, value_lr = self.policy.get_schedule_learning_rates()
                self.logger.record("train/policy_learning_rate", policy_lr)
                self.logger.record("train/value_learning_rate", value_lr)
            except Exception:
                pass

        entropy_losses = []
        policy_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []
        theta_maxs, theta_mins = [], []
        theta_grad_maxs, theta_grad_mins = [], []
        values_maxs, values_mins = [], []
        values_grad_maxs, values_grad_mins = [], []
        log_std_s = []

        continue_training = True

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                action_masks = None if not getattr(self, "use_masking", False) else getattr(rollout_data, "action_masks", None)
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, action_masks=action_masks
                )

                advantages = rollout_data.advantages
                if getattr(self, "normalize_advantage", True) and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = 0.5 * F.mse_loss(rollout_data.returns, values_pred)

                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if hasattr(self.policy, "nn_critic") and self.policy.nn_critic and hasattr(self.policy, "value_optimizer"):
                    self.policy.value_optimizer.zero_grad()

                loss.backward()

                use_safety = (
                    self.use_safety_projection
                    and hasattr(rollout_data, "cost_advantages")
                    and (rollout_data.cost_advantages is not None)
                )
                if use_safety:
                    # recompute log_prob for a clean graph (keeps autograd tidy)
                    _, log_prob_cost, _ = self.policy.evaluate_actions(
                        rollout_data.observations, actions, action_masks=action_masks
                    )
                    A_cost = rollout_data.cost_advantages.to(log_prob_cost.device)
                    if A_cost.shape != log_prob_cost.shape:
                        A_cost = A_cost.view_as(log_prob_cost)

                    cost_loss = -(log_prob_cost * A_cost).mean()

                    policy_params = [p for p in self.policy.parameters() if p.requires_grad]
                    if len(policy_params) > 0:
                        cost_grads = th.autograd.grad(cost_loss, policy_params, retain_graph=True, allow_unused=True)

                        g = _flatten_grads(policy_params)
                        b_parts = []
                        for p, cg in zip(policy_params, cost_grads):
                            b_parts.append(th.zeros_like(p).view(-1) if cg is None else cg.view(-1))
                        b = th.cat(b_parts) if len(b_parts) > 0 else th.zeros_like(g)

                        if callable(self.cost_value_source):
                            c_val = float(self.cost_value_source())
                        else:
                            c_val = float(getattr(self, "current_cost_estimate", float(A_cost.mean().item())))

                        g_safe = _a_projection(g, b, c_val, self.cost_threshold)
                        _write_back_grads(policy_params, g_safe)

                if hasattr(self.policy, "nn_critic") and self.policy.nn_critic:
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                entropy_losses.append(entropy_loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

                # Optional Gaussian log_std optimization (guarded)
                if hasattr(self.policy, "action_dist") and isinstance(self.policy.action_dist, DiagGaussianDistribution) \
                   and not getattr(self, "fixed_std", False) \
                   and hasattr(self.policy, "log_std") and hasattr(self.policy, "log_std_optimizer"):
                    if getattr(self, "max_policy_grad_norm", None):
                        th.nn.utils.clip_grad_norm_(self.policy.log_std, max_norm=self.max_policy_grad_norm, error_if_nonfinite=True)
                    if self.policy.log_std.grad is not None:
                        self.policy.log_std_optimizer.step()
                        self.policy.log_std_optimizer.zero_grad()
                        log_std_s.append(self.policy.log_std.detach().cpu().numpy())
                    else:
                        self.policy.log_std_optimizer.zero_grad()

                # Approx reverse KL for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).float().item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and self.target_kl > 0 and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # Fit GBRL models on the grads currently in .grad
                self.policy.step(policy_grad_clip=getattr(self, "max_policy_grad_norm", None),
                                 value_grad_clip=getattr(self, "max_value_grad_norm", None))

                # Optional detailed logging for trees (guard methods)
                if hasattr(self.policy, "get_params"):
                    try:
                        params, grads = self.policy.get_params()
                        if isinstance(grads, tuple):
                            theta_grad, values_grad = grads
                            theta, values = params
                            values_grad_maxs.append(values_grad.max().item())
                            values_grad_mins.append(values_grad.min().item())
                        else:
                            theta_grad = grads
                            theta = params
                        values_maxs.append(values.max().item())
                        values_mins.append(values.min().item())
                        theta_maxs.append(theta.max().item())
                        theta_mins.append(theta.min().item())
                        theta_grad_maxs.append(theta_grad.max().item())
                        theta_grad_mins.append(theta_grad.min().item())
                    except Exception:
                        pass

                self._n_updates += 1
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

            if not continue_training:
                break

        # Aggregate logs
        if len(theta_maxs) > 0:
            from stable_baselines3.common.utils import explained_variance
            self.rollout_cntr = getattr(self, "rollout_cntr", 0) + 1
            try:
                ev = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
                self.logger.record("train/explained_variance", ev)
            except Exception:
                pass

            self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
            self.logger.record("train/policy_gradient_loss", float(np.mean(policy_losses)))
            self.logger.record("train/value_loss", float(np.mean(value_losses)))
            self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
            self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))

            # Tree stats if available
            if hasattr(self.policy, "get_total_iterations"):
                self.logger.record("train/total_boosting_iterations", self.policy.get_total_iterations())
            if hasattr(self.policy, "get_iteration"):
                try:
                    it = self.policy.get_iteration()
                    if isinstance(it, tuple):
                        pit, vit = it
                        self.logger.record("train/policy_boosting_iterations", pit)
                        self.logger.record("train/value_boosting_iteration", vit)
                    else:
                        self.logger.record("train/policy_boosting_iterations", it)
                except Exception:
                    pass
            if hasattr(self.policy, "get_num_trees"):
                try:
                    nt = self.policy.get_num_trees()
                    if isinstance(nt, tuple):
                        pnt, vnt = nt
                        self.logger.record("train/policy_num_trees", pnt)
                        self.logger.record("train/value_num_trees", vnt)
                    else:
                        self.logger.record("train/policy_num_trees", nt)
                except Exception:
                    pass

            if log_std_s and hasattr(self.policy, "log_std"):
                self.logger.record("param/std", float(np.mean(np.mean(np.exp(np.concatenate(log_std_s, axis=0)), axis=0))))
                self.logger.record("param/log_std", float(np.mean(np.mean(np.concatenate(log_std_s, axis=0), axis=0))))
            if hasattr(self.policy, "log_std"):
                try:
                    self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
                except Exception:
                    pass

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


    def collect_rollouts(self,
                         env: VecEnv,
                         callback,
                         rollout_buffer: RolloutBuffer,
                         n_rollout_steps: int) -> bool:
        """
        Collect experiences from the environment and store them in the buffer.
        Overridden to extract 'cost' from info dict.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        rollout_buffer.reset()
        episode_costs = np.zeros(env.num_envs)
        episode_rewards = np.zeros(env.num_envs)

        n_steps = 0
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                actions, values, log_probs = self.policy.forward(self._last_obs, deterministic=False)

            new_obs, rewards, dones, infos = env.step(actions.cpu().numpy())

            self.num_timesteps += env.num_envs

            # Collect per-step cost from info
            costs = np.array([info.get("cost", 0.0) for info in infos])
            episode_costs += costs
            episode_rewards += rewards

            # Logging cost and reward per step
            for idx, info in enumerate(infos):
                info["episode_cost"] = episode_costs[idx]
                info["episode_reward"] = episode_rewards[idx]

            # Store data in the buffer
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts,
                               values, log_probs)

            self._last_obs = new_obs
            self._last_episode_starts = dones

            n_steps += 1

            # Handle episode ends, safety specific data
            for idx, done in enumerate(dones):
                if done:
                    self.logger.record("rollout/ep_cost", episode_costs[idx])
                    self.logger.record("rollout/ep_rew", episode_rewards[idx])
                    self.logger.record("rollout/constraint_satisfied", int(episode_costs[idx] <= self.cost_threshold))
                    self.logger.record("rollout/constraint_violation", int(episode_costs[idx] > self.cost_threshold))
                    episode_costs[idx] = 0.0
                    episode_rewards[idx] = 0.0

            if not callback.on_step():
                return False

        with th.no_grad():
            values = self.policy.predict_values(self._last_obs)

        rollout_buffer.compute_returns_and_advantage(last_values=values,
                                                     dones=self._last_episode_starts)

        callback.on_rollout_end()
        return True


"""
from algos.ppo import PPO_GBRL
import torch as th

class SafePPO_GBRL(PPO_GBRL):
    def __init__(self, *args, cost_fn=None, cost_threshold=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_fn = cost_fn  # Should return cost given obs, action
        self.cost_threshold = cost_threshold

    def project_gradient(self, grad, cost_grad, c_val):
        """"""
        Implements the a-projection from Chow et al. (2021).
        Solves: min ||g' - g||^2 s.t. <g', cost_grad> <= cost_threshold - c_val
        """"""
        lagrange_mult = th.clamp(
            (th.dot(grad, cost_grad) - self.cost_threshold + c_val) /
            (th.norm(cost_grad) ** 2 + 1e-10), min=0.0
        )
        return grad - lagrange_mult * cost_grad

    def train(self) -> None:
        super().train()  # Call base to handle standard PPO flow up to gradient calc

        # After grads are calculated but BEFORE `self.policy.step(...)`, intercept:
        # This part will replace the gradient if safety constraint violated.

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            obs, act = rollout_data.observations, rollout_data.actions
            if self.cost_fn is None:
                continue

            # Estimate cost and cost gradient (finite difference, learned model, or custom logic)
            c_val = self.cost_fn(obs, act).mean().item()
            grads = self.policy.get_params()[1]  # θ_grad (policy gradients)

            if c_val > self.cost_threshold:
                # Get cost gradient (dummy example: assume self.policy has method)
                cost_grad = self.policy.compute_cost_gradient(obs, act)
                new_grads = self.project_gradient(grads, cost_grad, c_val)
                self.policy.set_policy_gradient(new_grads)  # Inject corrected gradient

        # Now call policy.step() as usual
        self.policy.step(policy_grad_clip=self.max_policy_grad_norm, value_grad_clip=self.max_value_grad_norm)

"""