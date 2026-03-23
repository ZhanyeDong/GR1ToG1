import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset


class MLP_self(nn.Module):
    def __init__(self, num_i, num_h1, num_h2, num_h3, num_h4, num_o):
        super(MLP_self, self).__init__()

        self.linear1 = nn.Linear(num_i, num_h1)
        self.Tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(num_h1, num_h2)
        self.Tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(num_h2, num_h3)
        self.Tanh3 = nn.Tanh()
        self.linear4 = nn.Linear(num_h3, num_h4)
        self.Tanh4 = nn.Tanh()
        self.linear5 = nn.Linear(num_h4, num_o)

    def forward(self, input_tensor):
        x = self.linear1(input_tensor)
        x = self.Tanh1(x)
        x = self.linear2(x)
        x = self.Tanh2(x)
        x = self.linear3(x)
        x = self.Tanh3(x)
        x = self.linear4(x)
        x = self.Tanh4(x)
        x = self.linear5(x)
        return x


def _to_nx7_tensor(data: torch.Tensor) -> torch.Tensor:
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(data)}")
    t = data.to(torch.float32)
    if t.ndim == 1:
        if t.numel() != 7:
            raise ValueError(f"Expected 7 values, got shape {tuple(t.shape)}")
        return t.unsqueeze(0)
    if t.ndim == 2 and t.shape[1] == 7:
        return t
    raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    inputs_all = _to_nx7_tensor(torch.load(args.input_dataset))
    targets_all = _to_nx7_tensor(torch.load(args.target_dataset))
    if inputs_all.shape[0] != targets_all.shape[0]:
        raise ValueError(
            f"Input/target sample count mismatch: {inputs_all.shape[0]} vs {targets_all.shape[0]}"
        )
    if not (0.0 <= args.test_split <= 1.0):
        raise ValueError("--test_split must be in [0, 1]")

    total_n = inputs_all.shape[0]
    if args.test_split > 0.0:
        perm = torch.randperm(total_n)
        test_n = int(total_n * args.test_split)
        test_idx = perm[:test_n]
        train_idx = perm[test_n:]
        train_inputs = inputs_all[train_idx]
        train_targets = targets_all[train_idx]
        test_inputs = inputs_all[test_idx]
        test_targets = targets_all[test_idx]
    else:
        train_inputs = inputs_all
        train_targets = targets_all
        test_inputs = inputs_all
        test_targets = targets_all

    pin_memory = device.type == "cuda"
    data_loader_train = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    epochs = args.epochs
    num_sum_list = []
    model = MLP_self(7, 32, 64, 64, 32, 7).to(device)
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=1e-4
)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

   

    for _epoch in range(epochs):
        model.train()
        sum_loss = 0.0

        for inputs, targets in data_loader_train:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            intermediate_outputs = model(inputs)

            loss = F.mse_loss(intermediate_outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            optimizer.step()
            sum_loss += loss.item()

        num_sum_list.append(sum_loss)
        print("sum_loss:", sum_loss)
        scheduler.step()

    model.eval()
    data_loader_test = DataLoader(
        TensorDataset(test_inputs, test_targets),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    sum_test_loss = 0.0
    with torch.no_grad():
        for inputs_test, targets_test in data_loader_test:
            inputs_test = inputs_test.to(device, non_blocking=True)
            targets_test = targets_test.to(device, non_blocking=True)
            intermediate_outputs_test = model(inputs_test)

            loss_test = F.mse_loss(intermediate_outputs_test, targets_test)
            sum_test_loss += loss_test.item()

    print("sum_loss_test:", sum_test_loss)

    draw_epochs = list(range(epochs))
    plt.plot(draw_epochs, num_sum_list)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Process")
    plt.show()

    torch.save(model, "model_complete_G1toGR10318.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training MLP for G1 to GR1 mapping")

    parser.add_argument(
        "--input_dataset",
        type=str,
        default="goal_dataset_0319.pt",
        help="Input dataset path (.pt), shape (N,7)",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="train_dataset_0319.pt",
        help="Target dataset path (.pt), shape (N,7)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction for test set split (0-1). If 0, test == train.",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="input batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50000, help="number of training epochs")
    parser.add_argument("--clip", type=float, default=5, help="gradient clip value")
    args = parser.parse_args()

    main(args)
