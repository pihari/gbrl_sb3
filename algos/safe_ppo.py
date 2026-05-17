
from typing import Any, Dict, Optional, Union
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, update_learning_rate
from stable_baselines3.common.vec_env import VecEnv
from typing import NamedTuple
import math

from algos.ppo import PPO_GBRL
from buffers.rollout_buffer import CategoricalRolloutBuffer, MaskableRolloutBuffer

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

def _theta_projection(policy_grad: th.Tensor,
                      cost_grad: th.Tensor,
                      cost_value: float,
                      cost_threshold: float,
                      eps: float = 1e-10) -> th.Tensor:
    """
    closer to theta-projection
    g* = g - λ* b,  λ* = max(0, (<g,b> - (d - c)) / ||b||^2)
    """
    if policy_grad.numel() == 0 or cost_grad.numel() == 0:
        return policy_grad
    inner = th.dot(policy_grad, cost_grad)
    violation = cost_value - cost_threshold
    bnorm2 = th.dot(cost_grad, cost_grad) + eps

    # skip projection if within constraints:
    if violation <= 0.0:
        return policy_grad

    # actual theta-like projection
    lam = (inner + violation) / bnorm2
    lam = th.clamp(lam, min=0.0)
    return policy_grad - lam * cost_grad

def _lyapunov_safety_projection(a_raw: th.Tensor,
                               a_baseline: th.Tensor,
                               grad_Q_cost: th.Tensor,
                               e: th.Tensor,
                               eps: float = 1e-10) -> th.Tensor:
    """
    Project raw action onto Lyapunov-feasible set:
    a_safe = argmin_a 0.5 * ||a - a_raw||^2
             s.t. (a - a_baseline)^T grad_Q_cost <= e.
    """
    diff = a_raw - a_baseline
    g = grad_Q_cost
    numerator = (diff * g).sum(dim=-1) - e
    denominator = (g * g).sum(dim=-1) + eps
    lam = th.clamp(numerator / denominator, min=0.0).unsqueeze(-1)
    a_safe = a_raw - lam * g
    return a_safe

class SafeRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor
    cost_values: th.Tensor

class SafeRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cost_gae = True
        self.cost_gamma = self.gamma
        self.cost_gae_lambda = self.gae_lambda
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = None
        self.cost_returns = None

    def reset(self):
        super().reset()
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = None
        self.cost_returns = None

    def add(self, obs, action, reward, episode_start, value, log_prob, cost=0.0, cost_value=0.0):
        super().add(obs, action, reward, episode_start, value, log_prob)
        self.costs[self.pos-1] = cost  # offsets super().add(...) increment
        # counteract type mismatch
        if isinstance(cost_value, th.Tensor):
            cost_value = cost_value.clone().cpu().numpy().flatten()
        self.cost_values[self.pos - 1] = cost_value


    def get(self, batch_size=None):
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            self.observations = self.swap_and_flatten(self.observations)
            self.actions = self.swap_and_flatten(self.actions)
            self.values = self.swap_and_flatten(self.values)
            self.log_probs = self.swap_and_flatten(self.log_probs)
            self.advantages = self.swap_and_flatten(self.advantages)
            self.returns = self.swap_and_flatten(self.returns)
            self.costs = self.costs.reshape(-1)  # flatten (n_steps, n_envs)
            self.cost_values = self.cost_values.reshape(-1)
            self.cost_returns = self.cost_returns.reshape(-1)
            self.cost_advantages = self.cost_advantages.reshape(-1)
            self.generator_ready = True

        for start_idx in range(0, self.buffer_size * self.n_envs, batch_size or self.buffer_size):
            batch_indices = indices[start_idx: start_idx + (batch_size or self.buffer_size)]

            observations = th.tensor(self.observations[batch_indices], dtype=th.float32, device=self.device)
            actions = th.tensor(self.actions[batch_indices], dtype=th.float32, device=self.device)
            old_values = th.tensor(self.values[batch_indices], dtype=th.float32, device=self.device)
            old_log_prob = th.tensor(self.log_probs[batch_indices], dtype=th.float32, device=self.device)
            advantages = th.tensor(self.advantages[batch_indices], dtype=th.float32, device=self.device)
            returns = th.tensor(self.returns[batch_indices], dtype=th.float32, device=self.device)
            cost_advantages = th.tensor(self.cost_advantages[batch_indices], dtype=th.float32, device=self.device)
            cost_returns = th.tensor(self.cost_returns[batch_indices], dtype=th.float32, device=self.device)
            old_cost_values = th.tensor(self.cost_values[batch_indices], dtype=th.float32, device=self.device)

            yield SafeRolloutBufferSamples(
                observations=observations,
                actions=actions,
                old_values=old_values,
                old_log_prob=old_log_prob,
                advantages=advantages,
                returns=returns,
                cost_advantages=cost_advantages,
                cost_returns=cost_returns,
                cost_values=old_cost_values
            )

    def compute_returns_and_advantage(self, last_values, dones, last_cost_values, use_undisc_ep_cost=False):
        super().compute_returns_and_advantage(last_values, dones)

        costs = th.tensor(self.costs, dtype=th.float32, device=self.device)
        cost_values = th.tensor(self.cost_values, dtype=th.float32, device=self.device)
        last_cost_values = last_cost_values.detach().view(1, -1)

        adv_c = th.zeros_like(costs)
        last_gae = th.zeros((self.n_envs,), dtype=th.float32, device=self.device)

        dones_t = th.tensor(dones, dtype=th.float32, device=self.device)

        # episodic discount
        gamma_c = 1.0 if use_undisc_ep_cost else self.cost_gamma
        lam_c = self.cost_gae_lambda

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones_t
                next_values = last_cost_values.squeeze(0)
            else:
                next_non_terminal = 1.0 - th.tensor(self.episode_starts[step + 1], dtype=th.float32, device=self.device)
                next_values = cost_values[step + 1]

            delta = costs[step] + gamma_c * next_values * next_non_terminal - cost_values[step]
            last_gae = delta + gamma_c * lam_c * next_non_terminal * last_gae
            adv_c[step] = last_gae

        self.cost_advantages = adv_c
        self.cost_returns = adv_c + cost_values

class SafePPO_GBRL(PPO_GBRL):
    """
    TODO: description
    """

    def __init__(self, *args, **kwargs):
        self.cost_threshold: float = float(kwargs.pop("cost_threshold", 25.0)) # https://arxiv.org/html/2409.01245v1
        self.use_safety_projection: bool = bool(kwargs.pop("use_safety_projection", True))
        self.use_lagrangian_rlx: bool = not self.use_safety_projection
        self.cost_value_source = kwargs.pop("cost_value_source", None)
        self.cost_vf_coef = 0.5  # could be its own input arg with float(kwargs.pop("cost_vf_coef", 0.5))
        self.lag_lr = 0.001
        self.lag_lambda = 0.0

        # delay setup
        kwargs["_init_setup_model"] = False

        #use value optimizer for cost critic
        tree_opt = kwargs.get("tree_optimizer", None)
        if tree_opt is not None:
            if "cost_value_optimizer" not in tree_opt:
                if "value_optimizer" in tree_opt:
                    tree_opt["cost_value_optimizer"] = dict(tree_opt["value_optimizer"])
                else:
                    tree_opt["cost_value_optimizer"] = dict(tree_opt["policy_optimizer"])
            T = tree_opt["policy_optimizer"].get("T", None)
            if T is not None and "T" not in tree_opt["cost_value_optimizer"]:
                tree_opt["cost_value_optimizer"]["T"] = T

        super().__init__(*args, **kwargs)
        self.rollout_buffer_class = SafeRolloutBuffer
        self.ppo_setup_model()

        if not hasattr(self, "current_cost_estimate"):
            self.current_cost_estimate = 0.0
        # --- Diagnostic storage for empirical bound validation ---
        self.projection_log = []
        self.prev_returns_per_state = None
        self.projection_was_active_prev = False

    def train(self) -> None:
        """
        Update policy using current safe rollout buffer.
        """
        self.policy.set_training_mode(True)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # Optional: schedules and logging (guard attributes)
        if isinstance(self.policy.action_dist, DiagGaussianDistribution):
            update_learning_rate(self.policy.log_std_optimizer, self.policy.log_std_schedule(
                self._current_progress_remaining))
        if self.policy.nn_critic:
            self._update_learning_rate(self.policy.value_optimizer)
            self.logger.record("train/nn_critic", "True")
        else:
            self.logger.record("train/nn_critic", "False")
        policy_lr, value_lr = self.policy.get_schedule_learning_rates()
        self.logger.record("train/policy_learning_rate", policy_lr)
        self.logger.record("train/value_learning_rate", value_lr)

        # cost critic learning rate logs
        if hasattr(self.policy, "get_cost_schedule_learning_rate"):
            self.logger.record("train/cost_value_learning_rate", self.policy.get_cost_schedule_learning_rate())

        entropy_losses = []
        policy_losses, value_losses = [], []
        cost_value_losses = []
        clip_fractions = []
        approx_kl_divs = []
        theta_maxs, theta_mins = [], []
        theta_grad_maxs, theta_grad_mins = [], []
        values_maxs, values_mins = [], []
        values_grad_maxs, values_grad_mins = [], []
        log_std_s = []
        all_cost_returns = []

        continue_training = True

        for e in range(self.n_epochs):
            all_cost_returns = [] # reset per episode needed
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # TEMP DEBUG BLOCK
                """
                r = rollout_data.returns
                v = rollout_data.old_values
                print(
                    f"[DIAG] returns:    mean={r.mean():.3f}  std={r.std():.4f}  min={r.min():.3f}  max={r.max():.3f}")
                print(
                    f"[DIAG] old_values: mean={v.mean():.3f}  std={v.std():.4f}  min={v.min():.3f}  max={v.max():.3f}")
                ev = 1.0 - (r - v).var() / (r.var() + 1e-8)
                print(f"[DIAG] EV={ev:.6f}  n={r.shape[0]}")
                break
                """

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

                advantages = advantages.view(-1)
                log_prob = log_prob.view(-1)
                ratio = th.exp(log_prob - rollout_data.old_log_prob.view(-1))
                # Per-sample functional gradients
                g_func = advantages * ratio # this flattens the g
                use_safety = (
                        (self.use_safety_projection or self.use_lagrangian_rlx)
                        and rollout_data.cost_advantages is not None
                )
                if use_safety:
                    if self.use_safety_projection:
                        b_func = rollout_data.cost_advantages.view(-1) # this flattens the b
                        exp_ep_cost = getattr(self, 'undiscounted_ep_cost', rollout_data.cost_returns.mean().item()) #rollout_data.cost_returns.mean().item()
                        violation = exp_ep_cost - self.cost_threshold
                        #print(f"[cost_check] exp_ep_cost={exp_ep_cost:.3f}, violation={violation:.3f}, threshold={self.cost_threshold}")
                        if violation > 0:
                            g_func_original = g_func.detach().clone()

                            inner_F = (g_func * b_func).mean()
                            bnorm2_F = (b_func * b_func).mean() + 1e-10
                            b_rms = bnorm2_F.sqrt()
                            # violation scaling with ||b||
                            lam = th.clamp((inner_F + violation * b_rms) / bnorm2_F, min=0.0, max=1.0)
                            g_func = g_func - lam * b_func

                            # Diagnostics
                            with th.no_grad():
                                g_norm = g_func_original.norm().item()
                                b_norm = b_func.norm().item()
                                g_safe_norm = g_func.norm().item()

                                # phi: angle between reward and cost gradients
                                if g_norm > 1e-10 and b_norm > 1e-10:
                                    cos_phi = (g_func_original * b_func).sum().item() / (g_norm * b_norm)
                                    cos_phi = max(-1.0, min(1.0, cos_phi))
                                    phi = math.acos(cos_phi)
                                else:
                                    phi = 0.0

                                # rho: dimensionless projection strength
                                rho = lam.item() * b_norm / (g_norm + 1e-10)

                                # alpha_predicted from Lemma (Gradient Rotation)
                                cos_phi_val = math.cos(phi)
                                num = 1.0 - rho * cos_phi_val
                                den = math.sqrt(max(1e-20, 1.0 - 2.0 * rho * cos_phi_val + rho ** 2))
                                cos_alpha_pred = max(-1.0, min(1.0, num / den))
                                alpha_predicted = math.acos(cos_alpha_pred)

                                # alpha_observed: actual rotation
                                if g_norm > 1e-10 and g_safe_norm > 1e-10:
                                    cos_alpha_obs = (g_func_original * g_func).sum().item() / (g_norm * g_safe_norm)
                                    cos_alpha_obs = max(-1.0, min(1.0, cos_alpha_obs))
                                    alpha_observed = math.acos(cos_alpha_obs)
                                else:
                                    alpha_observed = 0.0

                                self.projection_log.append({
                                    "phi": phi,
                                    "alpha_predicted": alpha_predicted,
                                    "alpha_observed": alpha_observed,
                                    "rho": rho,
                                    "lambda_star": lam.item(),
                                    "violation": violation,
                                    "b_rms": b_rms.item(),
                                    "g_norm": g_norm,
                                    "b_norm": b_norm,
                                    "inner_F": inner_F.item(),
                                })
                        else:
                            self.projection_log.append({
                                "phi": 0.0, "alpha_predicted": 0.0,
                                "alpha_observed": 0.0, "rho": 0.0,
                                "lambda_star": 0.0, "violation": violation,
                                "b_rms": 0.0, "g_norm": 0.0, "b_norm": 0.0,
                                "inner_F": 0.0,
                            })

                    else:
                        cost_advantages = rollout_data.cost_advantages.view(-1)
                        cost_penalty = self.lag_lambda * cost_advantages * ratio.detach()
                        g_func = g_func - cost_penalty


                # Build policy loss from (projected) per-sample targets
                policy_loss_1 = g_func
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

                # Combined loss — identical structure to base PPO, no cost term
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if hasattr(self.policy, "nn_critic") and self.policy.nn_critic and hasattr(self.policy,
                                                                                           "value_optimizer"):
                    self.policy.value_optimizer.zero_grad()

                # Single backward — populates both params[0].grad and params[1].grad correctly
                loss.backward()

                if hasattr(self.policy, "nn_critic") and self.policy.nn_critic:
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                entropy_losses.append(entropy_loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

                max_trees = 25000
                max_trees_reached = self.policy.get_num_trees()[0] >= max_trees

                # Optional Gaussian log_std optimization (guarded)
                if isinstance(self.policy.action_dist, DiagGaussianDistribution) and not self.fixed_std and not max_trees_reached:
                    if self.max_policy_grad_norm is not None and self.max_policy_grad_norm > 0.0:
                        th.nn.utils.clip_grad_norm_(self.policy.log_std, max_norm=self.max_policy_grad_norm,
                                                    error_if_nonfinite=True)
                    self.policy.log_std_optimizer.step()
                    log_std_grad = self.policy.log_std.grad.clone().detach().cpu().numpy()
                    self.policy.log_std_optimizer.zero_grad()
                    assert ~np.isnan(log_std_grad).any(), "nan in assigned grads"
                    assert ~np.isinf(log_std_grad).any(), "infinity in assigned grads"
                    log_std_s.append(self.policy.log_std.detach().cpu().numpy())

                # Approx reverse KL for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).float().item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and self.target_kl > 0 and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # Fit GBRL models on the grads currently in .grad
                obs_np = rollout_data.observations.cpu().numpy() if isinstance(rollout_data.observations,
                                                                               th.Tensor) else rollout_data.observations

                if not max_trees_reached:
                    self.policy.model.actor_step(
                        observations=obs_np,
                        policy_grad_clip=self.max_policy_grad_norm
                    )

                    saved_params = self.policy.model.params
                    reward_values = self.policy.predict_values(
                        rollout_data.observations, requires_grad=True
                    )
                    reward_values.retain_grad()
                    reward_value_pred = reward_values.view_as(rollout_data.returns)
                    reward_value_loss = 0.5 * F.mse_loss(rollout_data.returns, reward_value_pred)
                    reward_value_loss.backward()

                    n_samples = reward_values.shape[0]
                    value_grad = reward_values.grad.detach() * n_samples

                    self.policy.model.critic_step(
                        observations=obs_np,
                        value_grad=value_grad.cpu().numpy(),
                        value_grad_clip=self.max_value_grad_norm
                    )

                    self.policy.model.params = saved_params
                else:
                    self.policy.zero_grad()

                # Cost critic: isolated forward → backward → step cycle
                cost_values_pred = self.policy.predict_cost_values(
                    rollout_data.observations, requires_grad=True
                )
                cost_values_pred = cost_values_pred.view_as(rollout_data.cost_returns)
                cost_value_loss = 0.5 * F.mse_loss(rollout_data.cost_returns, cost_values_pred)
                cost_value_loss.backward()
                cost_value_losses.append(cost_value_loss.item())

                if not max_trees_reached:
                    self.policy.cost_critic_step(
                        observations=rollout_data.observations,
                        cost_value_grad_clip=self.max_value_grad_norm
                    )

                params, grads = self.policy.get_params()
                #print(f"DEBUG get_params: params type={type(params)}, grads type={type(grads)}, params={params}, grads={grads}")
                if isinstance(grads, tuple):
                    theta_grad, values_grad = grads
                    theta, values_params = params
                    if values_grad is not None:
                        values_grad_maxs.append(values_grad.max().item())
                        values_grad_mins.append(values_grad.min().item())
                    if values_params is not None:
                        values_maxs.append(values_params.max().item())
                        values_mins.append(values_params.min().item())
                else:
                    theta_grad = grads
                    theta = params

                if theta is not None:
                    theta_maxs.append(theta.max().item())
                    theta_mins.append(theta.min().item())
                if theta_grad is not None:
                    theta_grad_maxs.append(theta_grad.max().item())
                    theta_grad_mins.append(theta_grad.min().item())

                self._n_updates += 1
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                all_cost_returns.append(rollout_data.cost_returns)

            with th.no_grad():
                if all_cost_returns: # prevent crash on early stopping
                    self.current_cost_estimate = th.cat(all_cost_returns).mean().item()

            if not continue_training:
                break

        if self.use_lagrangian_rlx:
            # Compute mean episode cost over the rollout buffer
            exp_ep_cost = self.rollout_buffer.cost_returns.mean().item()
            violation = exp_ep_cost - self.cost_threshold

            # Dual ascent update
            laglam_cap = 2.0
            self.lag_lambda = max(0.0, min(self.lag_lambda + self.lag_lr * violation, laglam_cap))

            # Log it
            self.logger.record("train/lag_lambda", self.lag_lambda)

        # Aggregate logs
        if len(theta_maxs) > 0:
            from stable_baselines3.common.utils import explained_variance
            self.rollout_cntr = getattr(self, "rollout_cntr", 0) + 1
            try:
                ev = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
                self.logger.record("train/explained_variance", ev)
            except Exception:
                ev = 0.0

            active_projs = [d for d in self.projection_log[-100:]
                            if d.get("lambda_star", 0) > 0]
            if active_projs:
                self.logger.record("safety/phi_deg_mean",
                                   np.mean([math.degrees(d["phi"]) for d in active_projs]))
                self.logger.record("safety/alpha_predicted_mean",
                                   np.mean([math.degrees(d["alpha_predicted"]) for d in active_projs]))
                self.logger.record("safety/alpha_observed_mean",
                                   np.mean([math.degrees(d["alpha_observed"]) for d in active_projs]))
                self.logger.record("safety/rho_mean",
                                   np.mean([d["rho"] for d in active_projs]))
                self.logger.record("safety/lambda_star_mean",
                                   np.mean([d["lambda_star"] for d in active_projs]))
                self.logger.record("safety/b_rms_mean",
                                   np.mean([d["b_rms"] for d in active_projs]))
                self.logger.record("safety/violation_mean",
                                   np.mean([d["violation"] for d in active_projs]))

                # Lemma validation: mean absolute error between predicted and observed alpha
                alpha_errors = [abs(d["alpha_predicted"] - d["alpha_observed"])
                                for d in active_projs]
                self.logger.record("safety/alpha_mae_deg",
                                   np.mean([math.degrees(e) for e in alpha_errors]))

            self.logger.record("safety/projection_active_frac",
                               len(active_projs) / max(1, len(self.projection_log[-100:])))

            # --- EV bound validation ---
            try:
                current_returns = self.rollout_buffer.returns.flatten()
                current_values = self.rollout_buffer.values.flatten()
                var_R2 = np.var(current_returns)

                if self.prev_returns_per_state is not None and var_R2 > 1e-10:
                    min_len = min(len(current_returns), len(self.prev_returns_per_state))
                    delta = current_returns[:min_len] - self.prev_returns_per_state[:min_len]
                    var_delta = np.var(delta)
                    ev_upper_bound = 1.0 - var_delta / var_R2

                    self.logger.record("ev_bound/var_delta", float(var_delta))
                    self.logger.record("ev_bound/var_R2", float(var_R2))
                    self.logger.record("ev_bound/ev_upper_bound", float(ev_upper_bound))
                    self.logger.record("ev_bound/ev_actual", float(ev))
                    self.logger.record("ev_bound/bound_slack", float(ev_upper_bound - ev))

                    # Track regime switches
                    proj_active_now = len(active_projs) > 0
                    if proj_active_now != self.projection_was_active_prev:
                        self.logger.record("ev_bound/regime_switch", 1.0)
                    else:
                        self.logger.record("ev_bound/regime_switch", 0.0)
                    self.projection_was_active_prev = proj_active_now

                self.prev_returns_per_state = current_returns.copy()
            except Exception:
                pass

            iteration = self.policy.get_iteration()
            num_trees = self.policy.get_num_trees()
            value_iteration = 0

            if isinstance(iteration, tuple):
                iteration, value_iteration = iteration
            value_num_trees = 0
            if isinstance(num_trees, tuple):
                num_trees, value_num_trees = num_trees

            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(policy_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("param/theta_max", np.mean(theta_maxs))
            self.logger.record("param/theta_min", np.mean(theta_mins))
            self.logger.record("param/value_max", np.mean(values_maxs))
            self.logger.record("param/value_min", np.mean(values_mins))
            self.logger.record("param/theta_grad_max", np.mean(theta_grad_maxs))
            self.logger.record("param/theta_grad_min", np.mean(theta_grad_mins))
            if values_grad_maxs:
                self.logger.record("param/value_grad_max", np.mean(values_grad_maxs))
                self.logger.record("param/value_grad_min", np.mean(values_grad_mins))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/explained_variance", ev)
            self.logger.record("train/total_boosting_iterations", self.policy.get_total_iterations())
            self.logger.record("train/policy_boosting_iterations", iteration)
            self.logger.record("train/value_boosting_iteration", value_iteration)
            self.logger.record("train/policy_num_trees", num_trees)
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.record("train/value_num_trees", value_num_trees)

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

        # dumping logger for test run with few steps
        self.logger.dump(self.num_timesteps)


    def collect_rollouts(self,
                         env: VecEnv,
                         callback,
                         rollout_buffer: Union[RolloutBuffer, SafeRolloutBuffer, CategoricalRolloutBuffer, MaskableRolloutBuffer],
                         n_rollout_steps: int) -> bool:
        """
        Collect experiences from the environment and store them in the buffer.
        Overridden to extract 'cost' from info dict.
        """

        self.policy.set_training_mode(False)

        assert self._last_obs is not None, "No previous observation was provided"
        rollout_buffer.reset()
        episode_costs = np.zeros(env.num_envs)
        episode_rewards = np.zeros(env.num_envs)

        n_steps = 0
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                #actions, values, log_probs = self.policy.forward(self._last_obs, deterministic=False)
                obs_tensor = self._last_obs if self.is_categorical else obs_as_tensor(self._last_obs, self.device)
                action_masks = get_action_masks(env) if self.use_masking else None
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks, requires_grad=False)
                cost_values = self.policy.predict_cost_values(obs_tensor, requires_grad=False)

            actions = actions.cpu().numpy()
            # Lyapunov safety layer
            # safe_action = _lyapunov_safety_projection(actions, a_baseline=...)

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Collect per-step cost from info
            costs = np.array([info.get("cost", 0.0) for info in infos])
            episode_costs += costs
            episode_rewards += rewards

            # Logging cost and reward per step
            for idx, info in enumerate(infos):
                info["episode_cost"] = episode_costs[idx]
                info["episode_reward"] = episode_rewards[idx]

            # Handle episode ends, safety specific data
            completed_costs = []
            completed_rewards = []
            violations = 0

            for idx, done in enumerate(dones):
                if done:
                    cost = episode_costs[idx]
                    reward = episode_rewards[idx]
                    completed_costs.append(cost)
                    completed_rewards.append(reward)
                    if cost > self.cost_threshold:
                        violations += 1
                    episode_costs[idx] = 0.0
                    episode_rewards[idx] = 0.0
                    if (infos[idx].get("terminal_observation") is not None
                            and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = infos[idx]["terminal_observation"] if self.is_categorical else \
                            self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]

                        rewards[idx] += self.gamma * terminal_value

            # Store data in the buffer
            kwargs = {}
            if self.use_masking:
                kwargs['action_masks'] = action_masks
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts,
                               values, log_probs, costs, cost_value=cost_values, **kwargs)

            self._last_obs = new_obs
            self._last_episode_starts = dones

            if completed_costs:
                self.logger.record("rollout/ep_cost_mean", np.mean(completed_costs))
                self.logger.record("rollout/ep_rew_mean", np.mean(completed_rewards))
                self.logger.record("rollout/constraint_violation_count", violations)
                self.logger.record("rollout/constraint_violation_rate", violations / len(completed_costs))

            #if not callback.on_step():
            #    return False

        with th.no_grad():
            obs_tensor = self._last_obs if self.is_categorical else obs_as_tensor(self._last_obs, self.device)
            last_values = self.policy.predict_values(obs_tensor, requires_grad=False)
            last_cost_values = self.policy.predict_cost_values(obs_tensor, requires_grad=False)

        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            last_cost_values=last_cost_values,
            dones=self._last_episode_starts,
        )

        callback.on_rollout_end()
        return True