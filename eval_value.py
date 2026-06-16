import numpy as np
import torch

from train_value import ValueNet, encode_value_states, predict_value


def load_value_model(path="value.pt"):
    model = ValueNet()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    data = np.load("values_oracle.npz")
    states = data["states"]
    values = data["values"]

    model = load_value_model()

    x = encode_value_states(states)
    targets = torch.from_numpy(values.astype(np.int64)) + 1

    with torch.no_grad():
        logits = model(x)

    predicted_classes = logits.argmax(dim=1)
    acc = (predicted_classes == targets).float().mean().item()

    empty_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)

    print(f"examples: {len(states)}")
    print(f"value accuracy: {acc:.1%}")
    print("empty-board value:", predict_value(model, empty_state))


if __name__ == "__main__":
    main()
