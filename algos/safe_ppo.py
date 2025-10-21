from typing import Any, Dict, Optional, Tuple, Type, Union
import time, sys
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    update_learning_rate,
    safe_mean,
)
from stable_baselines3.common.distributions import DiagGaussianDistribution

def _flatten_grads(params):
    flats = []
    for p in params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            flats.append(th.zeros_like(p).view(-1))
        else:
            flats.append(p.grad.view(-1))
    if len(flats) == 0:
        # return a 0 tensor on the right device if policy has no params
        dev = next(iter(params)).device if len(list(params)) > 0 else th.device("cpu")
        return th.zeros(0, device=dev)
    return th.cat(flats)

def _write_back_grads(params, flat: th.Tensor):
    offset = 0
    for p in params:
        if not p.requires_grad:
            continue
        n = p.numel() # number of tensor elements
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
    Implements the a-projection closed form
    g* = g - λ* b, where λ* = max(0, (<g,b> - (c - d)) / ||b||^2).
    - policy_grad: flattened gradient from reward objective (and regularizers)
    - cost_grad: flattened gradient of expected cost surrogate
    - cost_value: current estimate of expected cost J_C (scalar)
    - cost_threshold: constraint budget d (scalar)
    """
    if cost_grad.numel() == 0 or policy_grad.numel() == 0:
        return policy_grad
    inner = th.dot(policy_grad, cost_grad)
    violation = cost_value - cost_threshold
    bnorm2 = th.dot(cost_grad, cost_grad) + eps
    lam = (inner - violation) / bnorm2
    lam = th.clamp(lam, min=0.0)
    return policy_grad - lam * cost_grad

def _project_gradient(self, grad, cost_grad, c_val):
    """
    Implements the a-projection
    Solves: min ||g' - g||^2 s.t. <g', cost_grad> <= cost_threshold - c_val
    """
    lagrange_mult = th.clamp(
        (th.dot(grad, cost_grad) - self.cost_threshold + c_val) /
        (th.norm(cost_grad) ** 2 + 1e-10), min=0.0
    )
    return grad - lagrange_mult * cost_grad

class SafePPO_GBRL(OnPolicyAlgorithm):
    """
    TODO: add description
    """

    def __init__(
        self,
        policy: Union[str, Type[th.nn.Module]],
        env,
        learning_rate: Union[float, str] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, callable] = 0.2,
        clip_range_vf: Optional[Union[float, callable]] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # PPO-GBRL specific:
        use_masking: bool = False,
        fixed_std: bool = False,
        max_policy_grad_norm: Optional[float] = None,
        max_value_grad_norm: Optional[float] = None,
        # Safety:
        cost_threshold: float = 0.0,
        # expected interfaces: rollout_data.cost_advantages (Tensor), or set current_cost_estimate externally
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        )
        # PPO specifics
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        # GBRL extras
        self.use_masking = use_masking
        self.fixed_std = fixed_std
        self.max_policy_grad_norm = max_policy_grad_norm
        self.max_value_grad_norm = max_value_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        # safety
        self.cost_threshold = float(cost_threshold)
        self.current_cost_estimate = 0.0

        # logging helpers
        self.rollout_cntr = 0

        if _init_setup_model:
            self.ppo_setup_model()

    def ppo_setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **(self.policy_kwargs or {})
        ).to(self.device)

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, or None to disable."
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def get_action_bound_min(self):
        if isinstance(self.action_space, spaces.Box):
            bound_min = self.action_space.low
        else:
            bound_min = -np.inf * np.ones(1)
        return th.tensor(bound_min, device=self.device)

    def get_action_bound_max(self):
        if isinstance(self.action_space, spaces.Box):
            bound_max = self.action_space.high
        else:
            bound_max = np.inf * np.ones(1)
        return th.tensor(bound_max, device=self.device)

    def train(self) -> None:
        """
        TODO: add description.
        """
        self.policy.set_training_mode(True)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        if isinstance(self.policy.action_dist, DiagGaussianDistribution):
            update_learning_rate(self.policy.log_std_optimizer, self.policy.log_std_schedule(self._current_progress_remaining))
        if getattr(self.policy, "nn_critic", False):
            self._update_learning_rate(self.policy.value_optimizer)
            self.logger.record("train/nn_critic", "True")
        else:
            self.logger.record("train/nn_critic", "False")
        policy_lr, value_lr = self.policy.get_schedule_learning_rates()
        self.logger.record("train/policy_learning_rate", policy_lr)
        self.logger.record("train/value_learning_rate", value_lr)

        # accumulators
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

        # training loop
        for _ in range(self.n_epochs):
            # mini-batches
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                action_masks = None if not self.use_masking else rollout_data.action_masks
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                # evaluate current policy
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions, action_masks=action_masks)

                # normalize advantages if > 1
                advantages = rollout_data.advantages
                if getattr(self, "normalize_advantage", True) and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # clipped surrogate
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = 0.5 * F.mse_loss(rollout_data.returns, values_pred)

                # entropy bonus
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                # total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # backward (fills .grad buffers)
                if getattr(self.policy, "nn_critic", False):
                    self.policy.value_optimizer.zero_grad()
                loss.backward()

                # If cost advantages are available for this minibatch, project policy grads.
                if hasattr(rollout_data, "cost_advantages") and rollout_data.cost_advantages is not None:
                    A_cost = rollout_data.cost_advantages
                    # Build a cost surrogate loss using the SAME minibatch; recompute log_prob to keep a clean graph:
                    _, log_prob_cost, _ = self.policy.evaluate_actions(
                        rollout_data.observations, actions, action_masks=action_masks
                    )
                    cost_loss = -(log_prob_cost * A_cost).mean()

                    policy_params = [p for p in self.policy.parameters() if p.requires_grad]
                    # gradient of cost loss (do not zero; retain graph for others if needed)
                    cost_grads = th.autograd.grad(cost_loss, policy_params, retain_graph=True, allow_unused=True)

                    # flatten current policy grad g
                    g = _flatten_grads(policy_params)

                    # flatten cost gradient b
                    b_parts = []
                    for p, cg in zip(policy_params, cost_grads):
                        b_parts.append(th.zeros_like(p).view(-1) if cg is None else cg.view(-1))
                    b = th.cat(b_parts) if len(b_parts) > 0 else th.zeros_like(g)

                    # current cost estimate (can be updated externally each rollout); fallback: batch mean of cost adv
                    c_val = float(getattr(self, "current_cost_estimate", float(A_cost.mean().item())))
                    c_thresh = float(self.cost_threshold)

                    # project and write back
                    g_safe = _a_projection(g, b, c_val, c_thresh)
                    _write_back_grads(policy_params, g_safe)

                # (optional) clip total grad norm of policy params if you use a NN critic tied to same params
                if getattr(self.policy, "nn_critic", False):
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                # logs
                entropy_losses.append(entropy_loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

                # std optimizer (if diagonal Gaussian and std is trainable)
                if isinstance(self.policy.action_dist, DiagGaussianDistribution) and not self.fixed_std:
                    if self.max_policy_grad_norm is not None and self.max_policy_grad_norm > 0.0:
                        th.nn.utils.clip_grad_norm_(
                            self.policy.log_std, max_norm=self.max_policy_grad_norm, error_if_nonfinite=True
                        )
                    self.policy.log_std_optimizer.step()
                    log_std_grad = self.policy.log_std.grad.clone().detach().cpu().numpy()
                    self.policy.log_std_optimizer.zero_grad()
                    assert ~np.isnan(log_std_grad).any(), "nan in assigned grads"
                    assert ~np.isinf(log_std_grad).any(), "infinity in assigned grads"
                    log_std_s.append(self.policy.log_std.detach().cpu().numpy())

                # approx reverse KL for early stop
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and self.target_kl > 0 and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # Fit GBRL trees using the (possibly projected) grads now in .grad
                self.policy.step(policy_grad_clip=self.max_policy_grad_norm, value_grad_clip=self.max_value_grad_norm)

                # (optional) retrieve params/grads for logging
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

                self._n_updates += 1

                # clip fraction log
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

            if not continue_training:
                break

        # aggregate logs
        if len(theta_maxs) > 0:
            self.rollout_cntr += 1
            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            iteration = self.policy.get_iteration()
            num_trees = self.policy.get_num_trees()
            value_iteration = 0
            if isinstance(iteration, tuple):
                iteration, value_iteration = iteration
            value_num_trees = 0
            if isinstance(num_trees, tuple):
                num_trees, value_num_trees = num_trees

            self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
            self.logger.record("train/policy_gradient_loss", float(np.mean(policy_losses)))
            self.logger.record("train/value_loss", float(np.mean(value_losses)))
            self.logger.record("param/theta_max", float(np.mean(theta_maxs)))
            self.logger.record("param/theta_min", float(np.mean(theta_mins)))
            self.logger.record("param/value_max", float(np.mean(values_maxs)))
            self.logger.record("param/value_min", float(np.mean(values_mins)))
            self.logger.record("param/theta_grad_max", float(np.mean(theta_grad_maxs)))
            self.logger.record("param/theta_grad_min", float(np.mean(theta_grad_mins)))
            if values_grad_maxs:
                self.logger.record("param/value_grad_max", float(np.mean(values_grad_maxs)))
                self.logger.record("param/value_grad_min", float(np.mean(values_grad_mins)))
            self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
            self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/total_boosting_iterations", self.policy.get_total_iterations())
            self.logger.record("train/policy_boosting_iterations", iteration)
            self.logger.record("train/value_boosting_iteration", value_iteration)
            self.logger.record("train/policy_num_trees", num_trees)
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.record("train/value_num_trees", value_num_trees)

            if log_std_s:
                self.logger.record("param/std", float(np.mean(np.mean(np.exp(np.concatenate(log_std_s, axis=0)), axis=0))))
                self.logger.record("param/log_std", float(np.mean(np.mean(np.concatenate(log_std_s, axis=0), axis=0))))
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "GBRL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            # single training phase
            self.train()

        callback.on_training_end()
        return self

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