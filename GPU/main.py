import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import StepLR
import FK_G1_7DOF
import FK_GR1_7DOF


def cos(a):
    return torch.cos(a)


def sin(a):
    return torch.sin(a)


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

    def forward(self, input):
        x = self.linear1(input)
        x = self.Tanh1(x)
        x = self.linear2(x)
        x = self.Tanh2(x)
        x = self.linear3(x)
        x = self.Tanh3(x)
        x = self.linear4(x)
        x = self.Tanh4(x)
        x = self.linear5(x)
        return x


def normalize(v, eps=1e-6):
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def direction_loss(v1, v2):
    v1_n = normalize(v1)
    v2_n = normalize(v2)
    return 1.0 - torch.sum(v1_n * v2_n, dim=-1)


def triangle_area(a, b, c):
    ab = b - a
    ac = c - a
    cross = torch.cross(ab, ac, dim=-1)
    area = 0.5 * torch.norm(cross, dim=-1)
    return area


def triangle_area_loss(a1, b1, c1, a2, b2, c2):
    area1 = triangle_area(a1, b1, c1)
    area2 = triangle_area(a2, b2, c2)
    return F.mse_loss(area1, area2)


def calculate_IK_loss(
    GR1_joint1, G1_joint1,
    GR1_joint3, G1_joint3,
    GR1_joint4, G1_joint4,
    GR1_position, GR1_orientation,
    G1_position, G1_orientation,
    w_pos=3.0,
    w_ori=1.0,
    w_dir=0.001,
):
    v_gr1_13 = GR1_joint3 - GR1_joint1
    v_gr1_34 = GR1_joint4 - GR1_joint3

    v_g1_13 = G1_joint3 - G1_joint1
    v_g1_34 = G1_joint4 - G1_joint3

    dir_loss_1 = direction_loss(v_gr1_13, v_g1_13)
    dir_loss_2 = direction_loss(v_gr1_34, v_g1_34)
    dir_loss = (dir_loss_1 + dir_loss_2).mean()

    area_loss = triangle_area_loss(
        GR1_joint1, GR1_joint3, GR1_joint4,
        G1_joint1, G1_joint3, G1_joint4,
    )

    pos_loss = F.mse_loss(G1_position, GR1_position)
    ori_loss = F.mse_loss(G1_orientation, GR1_orientation)

    total_loss = (
        w_pos * pos_loss
        + w_ori * ori_loss
        + w_dir * dir_loss
    )

    return total_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_num = []
    test_num = []
    loaded_tensor1 = torch.load('train_list1.pt')
    loaded_tensor2 = torch.load('train_list2.pt')
    loaded_tensor3 = torch.load('train_list3.pt')
    loaded_tensor4 = torch.load('train_list4.pt')
    loaded_tensor5 = torch.load('train_list5.pt')

    for _ in range(args.num_train):
        joint_angles = torch.stack([
            torch.tensor(np.random.uniform(-0.3, 1), dtype=torch.float32),
            torch.tensor(np.random.uniform(-0.1, 0.7), dtype=torch.float32),
            torch.tensor(np.random.uniform(-1, 0.6), dtype=torch.float32),
            torch.tensor(np.random.uniform(0.3, np.pi / 2), dtype=torch.float32),
            torch.tensor(np.random.uniform(-0.5, np.pi / 2), dtype=torch.float32),
            torch.tensor(np.random.uniform(-0.5, np.pi / 2), dtype=torch.float32),
            torch.tensor(np.random.uniform(-np.pi / 2, np.pi / 2), dtype=torch.float32),
        ], dim=-1)
        train_num.append(joint_angles)

    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor1[i])
        train_num.append(GR1_R_change(joint_angles))
    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor2[i])
        train_num.append(GR1_R_change(joint_angles))
    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor3[i])
        train_num.append(GR1_R_change(joint_angles))
    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor4[i])
        train_num.append(GR1_R_change(joint_angles))
    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor5[i])
        train_num.append(GR1_R_change(joint_angles))

    for _ in range(args.num_test):
        joint_angles = torch.stack([
            torch.tensor(np.random.uniform(-0.3, 1), dtype=torch.float32),
            torch.tensor(np.random.uniform(-0.1, 0.7), dtype=torch.float32),
            torch.tensor(np.random.uniform(-1, 0.6), dtype=torch.float32),
            torch.tensor(np.random.uniform(0.3, np.pi / 2), dtype=torch.float32),
            torch.tensor(np.random.uniform(-0.5, np.pi / 2), dtype=torch.float32),
            torch.tensor(np.random.uniform(-0.5, np.pi / 2), dtype=torch.float32),
            torch.tensor(np.random.uniform(-np.pi / 2, np.pi / 2), dtype=torch.float32),
        ], dim=-1)
        test_num.append(joint_angles)

    train_tensor = torch.stack(train_num)
    data = TensorDataset(train_tensor)
    pin_memory = device.type == "cuda"
    data_loader_train = DataLoader(data, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    model = MLP_self(7, 32, 64, 64, 32, 7).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.95)

    epochs = args.epochs
    num_sum_list = []
    train_size = len(data)

    GR1_DOF = FK_GR1_7DOF.FK_7DOF(device=device)
    G1_DOF = FK_G1_7DOF.FK_7DOF(device=device)
    GR1_joint1 = torch.tensor([0.19346, 0.0, 0.0], dtype=torch.float32, device=device)
    G1_joint1 = torch.tensor([0.15596, -0.0025, -0.05], dtype=torch.float32, device=device)

    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0

        for (inputs,) in data_loader_train:
            inputs = inputs.to(device, non_blocking=True)
            intermediate_outputs = model(inputs)

            GR1_position, GR1_orientation, GR1_joint3, GR1_joint4 = GR1_DOF.compute_fk(inputs)
            G1_position, G1_orientation, G1_joint3, G1_joint4 = G1_DOF.compute_fk(intermediate_outputs)

            loss = calculate_IK_loss(
                GR1_joint1, G1_joint1,
                GR1_joint3, G1_joint3,
                GR1_joint4, G1_joint4,
                GR1_position, GR1_orientation,
                G1_position, G1_orientation,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            optimizer.step()
            batch_size_now = inputs.shape[0]
            sum_loss += loss.item() * batch_size_now

        epoch_mean_loss = sum_loss / train_size
        num_sum_list.append(epoch_mean_loss)
        print('epoch:', epoch, 'loss:', epoch_mean_loss)
        scheduler.step()

    model.eval()
    test_tensor = torch.stack(test_num)
    data_test = TensorDataset(test_tensor)
    data_loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    test_size = len(data_test)

    sum_test_loss = 0.0
    with torch.no_grad():
        for (inputs_test,) in data_loader_test:
            inputs_test = inputs_test.to(device, non_blocking=True)
            intermediate_outputs_test = model(inputs_test)

            GR1_position, GR1_orientation, GR1_joint3, GR1_joint4 = GR1_DOF.compute_fk(inputs_test)
            G1_position, G1_orientation, G1_joint3, G1_joint4 = G1_DOF.compute_fk(intermediate_outputs_test)

            loss_test = calculate_IK_loss(
                GR1_joint1, G1_joint1,
                GR1_joint3, G1_joint3,
                GR1_joint4, G1_joint4,
                GR1_position, GR1_orientation,
                G1_position, G1_orientation,
            )
            batch_size_now = inputs_test.shape[0]
            sum_test_loss += loss_test.item() * batch_size_now

    print('mean_loss_test:', sum_test_loss / test_size)

    draw_epochs = list(range(epochs))
    plt.plot(draw_epochs, num_sum_list)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Process')
    plt.show()

    torch.save(model, 'model_complete0306.pth')


def GR1_R_change(theta_list):
    theta_list_change = torch.stack([
        -theta_list[0], -theta_list[1], -theta_list[2], -theta_list[3],
        theta_list[4], theta_list[6], theta_list[5]
    ])
    return theta_list_change


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training MLP')

    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate (default: 0.003)')
    parser.add_argument('--epochs', type=int, default=5000, help='gradient clip value (default: 300)')
    parser.add_argument('--clip', type=float, default=5, help='gradient clip value (default: 1)')
    parser.add_argument('--num_train', type=int, default=5000)
    parser.add_argument('--num_test', type=int, default=30)
    args = parser.parse_args()

    main(args)
