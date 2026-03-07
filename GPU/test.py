import torch
import torch.nn as nn
import FK_G1_7DOF
import FK_GR1_7DOF
import GR1_to_SIM_R


class MLP_self(nn.Module):

    def __init__(self, num_i, num_h1,num_h2,num_h3,num_h4,num_o):
        super(MLP_self, self).__init__()

        self.linear1 = nn.Linear(num_i, num_h1) # 添加第一个dropout层
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


def use_model(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
    return outputs


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    theta_list = torch.load('train_list1.pt')
    model = torch.load('model_complete0305.pth', map_location=device)
    model = model.to(device)
    model.eval()

    # epochs = len(theta_list)
    epochs = 100
    sum = 0
    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 0
    fk_gr1 = FK_GR1_7DOF.FK_7DOF(device=device)
    fk_g1 = FK_G1_7DOF.FK_7DOF(device=device)
    for i in range(epochs):
        inputs = torch.as_tensor(theta_list[i], dtype=torch.float32, device=device)
        inputs = GR1_to_SIM_R.R_change(inputs)
        # print(inputs)
        result = fk_gr1.compute_fk(inputs)
        GR1_position, GR1_orientation, GR1_joint3, GR1_joint4 = result
        inputs2 = use_model(model, inputs)
        result2 = fk_g1.compute_fk(inputs2)
        G1_position, G1_orientation, G1_joint3, G1_joint4 = result2
        dif2 = GR1_position - G1_position
        dif2_norm = torch.norm(dif2)
        sum = sum + dif2_norm
        if dif2_norm < 0.005:
            a0 = a0 + 1
        if dif2_norm < 0.01:
            a1 = a1 + 1
        if dif2_norm < 0.03:
            a2 = a2 + 1
        if dif2_norm < 0.05:
            a3 = a3 + 1
    print(a0, a1, a2,a3,sum/epochs)
