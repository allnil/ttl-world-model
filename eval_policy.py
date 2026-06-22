import numpy as np
import torch

from train_policy import PolicyNet, build_policy_targets, masked_policy_probs


def load_policy_model(path="policy.pt"):
    model = PolicyNet()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    data = np.load("values_oracle.npz")
    states = data["states"]
    values = data["values"]
    policy_states, targets, optimal_action_sets = build_policy_targets(states, values)

    model = load_policy_model()
    top1_correct = 0
    legal_mass_on_optimal = 0.0

    for state, target, optimal_actions in zip(policy_states, targets, optimal_action_sets):
        legal_actions = np.where(state[:9] == 0)[0]
        probs = masked_policy_probs(model, state, legal_actions)
        best_action = max(probs, key=probs.get)
        if best_action in optimal_actions:
            top1_correct += 1
        legal_mass_on_optimal += sum(probs[action] for action in optimal_actions)

    n = len(policy_states)
    print(f"examples: {n}")
    print(f"top-1 optimal action accuracy: {top1_correct / n:.1%}")
    print(f"mean probability mass on optimal actions: {legal_mass_on_optimal / n:.1%}")


if __name__ == "__main__":
    main()
