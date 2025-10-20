from ppo import PPO_GBRL
import torch as th

class SafePPO_GBRL(PPO_GBRL):
    def __init__(self, *args, cost_fn=None, cost_threshold=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_fn = cost_fn  # Should return cost given obs, action
        self.cost_threshold = cost_threshold

    def project_gradient(self, grad, cost_grad, c_val):
        """
        Implements the a-projection from Chow et al. (2021).
        Solves: min ||g' - g||^2 s.t. <g', cost_grad> <= cost_threshold - c_val
        """
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
