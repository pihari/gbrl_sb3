from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union
import copy

import numpy as np
import torch as th
from gbrl.models.actor import ParametricActor

from policies.actor_critic_policy import ActorCriticPolicy


class ActorCriticSafePolicy(ActorCriticPolicy):
    """
    """

    def __init__(
        self,
        *args,
        cost_tree_struct: Optional[Dict] = None,
        cost_value_optimizer: Optional[Dict] = None,
        **kwargs,
    ):
        # Keep copies of configs so we can build the extra model after super().__init__()
        self._cost_tree_struct_in = cost_tree_struct
        self._cost_value_optimizer_in = cost_value_optimizer

        self._base_tree_struct = None
        self._base_tree_optimizer = None

        super().__init__(*args, **kwargs)

        tree_struct = kwargs.get("tree_struct", None)
        tree_optimizer = kwargs.get("tree_optimizer", None)

        if tree_struct is None or tree_optimizer is None:
            raise ValueError(
                "ActorCriticSafePolicy requires policy_kwargs to include tree_struct and tree_optimizer "
                "so it can construct a separate cost critic."
            )

        self._base_tree_struct = tree_struct
        self._base_tree_optimizer = tree_optimizer

        # Build separate cost critic
        self.cost_value_model = self._build_cost_critic_model(
            tree_struct=tree_struct,
            tree_optimizer=tree_optimizer,
            cost_tree_struct=self._cost_tree_struct_in,
            cost_value_optimizer=self._cost_value_optimizer_in,
        )

    def _build_cost_critic_model(
        self,
        tree_struct: Dict,
        tree_optimizer: Dict,
        cost_tree_struct: Optional[Dict],
        cost_value_optimizer: Optional[Dict],
    ) -> ParametricActor:
        # Decide structure
        c_struct = cost_tree_struct if cost_tree_struct is not None else tree_struct

        # Decide optimizer config for the cost critic model
        if cost_value_optimizer is not None:
            c_opt = copy.deepcopy(cost_value_optimizer)
        else:
            # Prefer cloning value_optimizer if available, else policy_optimizer
            if "value_optimizer" in tree_optimizer and tree_optimizer["value_optimizer"] is not None:
                c_opt = copy.deepcopy(tree_optimizer["value_optimizer"])
            else:
                c_opt = copy.deepcopy(tree_optimizer["policy_optimizer"])

        # ParametricActor expects policy_optimizer dict with start/stop indices
        c_opt["start_idx"] = 0
        c_opt["stop_idx"] = 1

        # Keep same device/params as base models
        device = tree_optimizer.get("device", "cpu")
        params = tree_optimizer["params"]

        return ParametricActor(
            tree_struct=c_struct,
            input_dim=self.features_dim,
            output_dim=1,
            policy_optimizer=c_opt,
            params=params,
            device=device,
        )

    def predict_cost_values(self, obs: Union[th.Tensor, np.ndarray], requires_grad: bool = True) -> th.Tensor:
        if self.cost_value_model is None:
            if not isinstance(obs, th.Tensor):
                obs = th.tensor(obs, device=self.device)
            return th.zeros((obs.shape[0], 1), device=obs.device, dtype=th.float32)
        return self.cost_value_model(obs, requires_grad, tensor=True)

    def cost_critic_step(
        self,
        observations: Optional[Union[np.ndarray, th.Tensor]] = None,
        cost_value_grad_clip: Optional[float] = None,
    ) -> None:
        if self.cost_value_model is None:
            return
        self.cost_value_model.step(observations=observations, policy_grad_clip=cost_value_grad_clip)

    def update_cost_learning_rate(self, cost_learning_rate: float) -> None:
        if self.cost_value_model is None:
            return
        self.cost_value_model.adjust_learning_rates(cost_learning_rate)

    def get_cost_schedule_learning_rate(self) -> float:
        if self.cost_value_model is None:
            return 0.0
        lrs = self.cost_value_model.get_schedule_learning_rates()
        return float(lrs[0]) if len(lrs) > 0 else 0.0