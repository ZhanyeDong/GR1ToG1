import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import numpy as np
import os
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import StepLR
import FK_G1_7DOF
import FK_GR1_7DOF
def cos(a):
    return torch.cos(a)
# 瀹氫箟鍙井鍒嗗舰寮?
def sin(a):
    return torch.sin(a)
class MLP_self(nn.Module):

    def __init__(self, num_i, num_h1,num_h2,num_h3,num_h4,num_o):
        super(MLP_self, self).__init__()

        self.linear1 = nn.Linear(num_i, num_h1) # 娣诲姞绗竴涓猟ropout灞?
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
    """
    瀹夊叏褰掍竴鍖栵紝闃叉闄?0
    """
    return v / (torch.norm(v) + eps)
def direction_loss(v1, v2):
    """
    鏂瑰悜涓€鑷存€ф崯澶憋紙1 - cos胃锛?
    v1, v2: shape (3,)
    """
    v1_n = normalize(v1)
    v2_n = normalize(v2)
    return 1.0 - torch.sum(v1_n * v2_n)

def triangle_area(a, b, c, eps=1e-8):
    """
    璁＄畻涓夌偣鏋勬垚涓夎褰㈢殑闈㈢Н锛堝彲寰級

    a, b, c: shape (3,)
    """
    ab = b - a
    ac = c - a
    cross = torch.cross(ab, ac)
    area = 0.5 * torch.norm(cross)
    return area


def triangle_area_loss(a1, b1, c1, a2, b2, c2):
    """
    涓や釜涓夎褰㈤潰绉竴鑷存€ф崯澶?
    """
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
    w_area=0.,
    w_area2=0.0,
):
    """
    鍩轰簬涓夌偣鏋勫瀷锛堥潰绉級鐨勭ǔ瀹?IK Loss
    """
    v_gr1_13 = GR1_joint3 - GR1_joint1
    v_gr1_34 = GR1_joint4 - GR1_joint3

    # G1
    v_g1_13 = G1_joint3 - G1_joint1
    v_g1_34 = G1_joint4 - G1_joint3

    # ========= 鏂瑰悜鎹熷け =========
    dir_loss_1 = direction_loss(v_gr1_13, v_g1_13)
    dir_loss_2 = direction_loss(v_gr1_34, v_g1_34)

    dir_loss = dir_loss_1 + dir_loss_2
    # ========= 涓夌偣鏋勫瀷闈㈢Н鎹熷け =========
    area_loss = triangle_area_loss(
        GR1_joint1, GR1_joint3, GR1_joint4,
        G1_joint1,  G1_joint3,  G1_joint4
    )

    # ========= 浣嶅Э鎹熷け =========
    pos_loss = F.mse_loss(G1_position, GR1_position)
    ori_loss = F.mse_loss(G1_orientation, GR1_orientation)

    # ========= 鎬绘崯澶?=========
    total_loss = (
        w_pos * pos_loss +
        w_ori * ori_loss +
        w_area * area_loss+
        w_area2 * dir_loss
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
    for i in range(args.num_train):
        joint_angles = torch.stack([torch.tensor(np.random.uniform(-0.3, 1), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-0.1, 0.7), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-1, 0.6), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(0.3, np.pi/2), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-0.5, np.pi/2), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-0.5, np.pi/2), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-np.pi/2, np.pi/2), dtype=torch.float32)],
                                   dim=-1)
        train_num.append(joint_angles)
    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor1[i])
        joint_angles = GR1_R_change(joint_angles)
        train_num.append(joint_angles)
    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor2[i])
        joint_angles = GR1_R_change(joint_angles)
        train_num.append(joint_angles)
    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor3[i])
        joint_angles = GR1_R_change(joint_angles)
        train_num.append(joint_angles)
    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor4[i])
        joint_angles = GR1_R_change(joint_angles)
    for i in range(200):
        joint_angles = torch.tensor(loaded_tensor5[i])
        joint_angles = GR1_R_change(joint_angles)
        train_num.append(joint_angles)
    for i in range(args.num_test):
        joint_angles = torch.stack([torch.tensor(np.random.uniform(-0.3, 1), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-0.1, 0.7), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-1, 0.6), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(0.3, np.pi / 2), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-0.5, np.pi / 2), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-0.5, np.pi / 2), dtype=torch.float32),
                                    torch.tensor(np.random.uniform(-np.pi / 2, np.pi / 2), dtype=torch.float32)],
                                   dim=-1)
        test_num.append(joint_angles)

    train_tensor = torch.stack(train_num)
    data = TensorDataset(train_tensor)
    data_loader_train = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    num_i = 7
    num_h1 = 32
    num_h2 = 64
    num_h3 = 64
    num_h4 = 32
    # num_o = 3
    num_o = 7

    model = MLP_self(num_i, num_h1,num_h2,num_h3,num_h4,num_o).to(device)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=0.000)  # 瀹氫箟浼樺寲鍣?
    scheduler = StepLR(optimizer, step_size=50, gamma=0.95)
    epochs = args.epochs
    num_sum_list = []
    sevenD_angle_list = []
    data_list = []
    output_list = []
    sevenD_joint3_list = []
    GR1_DOF = FK_GR1_7DOF.FK_7DOF(device=torch.device("cpu"))
    G1_DOF = FK_G1_7DOF.FK_7DOF(device=torch.device("cpu"))
    for epoch in range(epochs):
        model.train()

        sum_loss = 0.0
        for (inputs,) in data_loader_train:  # 璇诲叆鏁版嵁寮€濮嬭缁?
            # a = 0
            inputs = inputs.to(device)
            intermediate_outputs = model(inputs)
            intermediate_outputs.retain_grad()
            IK_loss_batch = torch.tensor(0.0)
            for i in range(len(inputs)):
            # 璁＄畻 IK_loss_batch
                input_cpu = inputs[i].detach().cpu()
                output_cpu = intermediate_outputs[i].cpu()
                GR1_position, GR1_orientation, GR1_joint3, GR1_joint4 = GR1_DOF.compute_fk(input_cpu)

                end_eff_calcu_by_FK = G1_DOF.compute_fk(output_cpu)
            # print(end_eff_calcu_by_FK)
                G1_position, G1_orientation, G1_joint3, G1_joint4 = end_eff_calcu_by_FK
                if epoch == epochs - 1:
                    sevenD_angle = output_cpu.detach()
                    data_list.append(input_cpu)
                    sevenD_angle_list.append(sevenD_angle)
                    output_list.append(output_cpu.detach())
                    sevenD_joint3_list.append(G1_joint3)
                GR1_joint1 = torch.tensor([0.19346,0. , 0.0], dtype=torch.float32)
                G1_joint1 = torch.tensor([0.15596,-0.0025,-0.05], dtype=torch.float32)
                tamp = calculate_IK_loss(GR1_joint1,G1_joint1,GR1_joint3,G1_joint3,GR1_joint4,G1_joint4
                                     , GR1_position,GR1_orientation,G1_position,G1_orientation)
                IK_loss_batch = IK_loss_batch + tamp
            loss = (IK_loss_batch) / len(inputs)
            loss.retain_grad()
            optimizer.zero_grad()
                # 璁板綍x杞互鍚庣綉缁滄ā鍨媍heckpoint锛岀敤鏉ユ煡鐪嬫暟鎹祦锛岃矾寰勯€夎嚜宸辩數鑴戠殑鐩爣鏂囦欢澶?
            loss.backward()  # 鍙嶅悜浼犳挱姹傛搴?
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)  # 杩涜姊害瑁佸壀
            optimizer.step()  # 鏇存柊鎵€鏈夋搴?
            sum_loss = sum_loss + loss.item()
        num_sum_list.append(sum_loss)
        print('epochs:',epoch,'sum_loss:', sum_loss)
        scheduler.step()
    # torch.save(sevenD_joint3_list, "./sevenD_joint3_list.pt")

#娴嬭瘯闆?
    model.eval()
    test_tensor = torch.stack(test_num)
    data_test = TensorDataset(test_tensor)
    data_loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    num_sum_test_list = []
    sevenD_angle_test_list = []
    data_test_list = []
    output_test_list = []
    sum_test_loss = 0.0
    for data_test in data_loader_test:  # 璇诲叆鏁版嵁寮€濮嬭缁?
        # a = 0
        inputs_test = data_test[0].to(device)
        intermediate_outputs_test = model(inputs_test)
        IK_loss_batch_test = torch.tensor(0.0)
        for i in range(len(inputs_test)):
            input_test_cpu = inputs_test[i].detach().cpu()
            output_test_cpu = intermediate_outputs_test[i].cpu()
            result = GR1_DOF.compute_fk(input_test_cpu)
            GR1_position, GR1_orientation, GR1_joint3, GR1_joint4 = result
            end_eff_calcu_by_FK = G1_DOF.compute_fk(output_test_cpu)
            # print(end_eff_calcu_by_FK)
            G1_position, G1_orientation, G1_joint3, G1_joint4 = end_eff_calcu_by_FK
            sevenD_angle = intermediate_outputs_test
            sevenD_angle_test_list.append(sevenD_angle[i])
            data_test_list.append(input_test_cpu)
            output_test_list.append(output_test_cpu.detach())
            GR1_joint1 = torch.tensor([0.19346, 0., 0.], dtype=torch.float32)
            G1_joint1 = torch.tensor([0.15596, -0.0025, -0.05], dtype=torch.float32)
            tamp = calculate_IK_loss(GR1_joint1, G1_joint1, GR1_joint3, G1_joint3, GR1_joint4, G1_joint4
                                     , GR1_position, GR1_orientation, G1_position, G1_orientation)
            IK_loss_batch_test = IK_loss_batch_test + tamp
        loss_test = (IK_loss_batch_test) / len(inputs_test)
        sum_test_loss = sum_test_loss + loss_test.item()
    print('sum_loss_test:', sum_test_loss)

    draw_epochs = list(range(epochs))
    plt.plot(draw_epochs, num_sum_list)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Process')
    # 鏄剧ず鍥句緥
    plt.show()
    torch.save(model, 'model_complete0305.pth')

def GR1_R_change(theta_list):
    theta_list_change = torch.stack([-theta_list[0], -theta_list[1], -theta_list[2], -theta_list[3],
                                     theta_list[4], theta_list[6], theta_list[5]])
    return theta_list_change



if __name__ == '__main__':
    # 鍒涘缓瑙ｆ瀽鍣?
    parser = argparse.ArgumentParser(description='Training MLP')

    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate (default: 0.003)')
    parser.add_argument('--epochs', type=int, default=5000 ,help='gradient clip value (default: 300)')
    parser.add_argument('--clip', type=float, default=5, help='gradient clip value (default: 1)')
    parser.add_argument('--num_train', type=int, default=10000)
    parser.add_argument('--num_test', type=int, default=30)
    args = parser.parse_args()

    main(args)


