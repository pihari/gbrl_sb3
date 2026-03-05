from tensorboard.backend.event_processing import event_accumulator
import sys

logdir = sys.argv[1] if len(sys.argv) > 1 else 'runs/'

ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()

tags = [
    'rollout/ep_rew_mean',
    'rollout/ep_cost_mean',
    'rollout/constraint_violation_rate',
    'rollout/constraint_violation_count',
    'train/explained_variance',
    'train/approx_kl',
    'train/policy_gradient_loss',
    'train/value_loss',
    'param/theta_grad_max',
    'param/theta_grad_min',
    'param/theta_max',
    'param/theta_min',
    'train/policy_num_trees',
    'train/value_num_trees',
]

for tag in tags:
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        for e in events:
            print(f"{tag:45s} step={e.step:>8d}  value={e.value:.4f}")
    else:
        print(f"{tag:45s} NOT FOUND")