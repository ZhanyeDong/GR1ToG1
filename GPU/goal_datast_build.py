import torch
import torch.nn as nn
import FK_G1_7DOF
import FK_GR1_7DOF
import G1_to_SIM_R
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
def use_model(inputs):
    model = torch.load('model_complete0319_2.pth', map_location="cpu")
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    return outputs
if __name__ == '__main__':
    theta_list = torch.load('train_dataset_0319.pt')
    goal_list = []
    epochs = len(theta_list)
    sum = 0
    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 0
    FK_GR1_7DOF = FK_GR1_7DOF.FK_7DOF()
    FK_G1_7DOF = FK_G1_7DOF.FK_7DOF()
    for i in range(epochs):
        inputs = torch.tensor(theta_list[i]).to(torch.float32)
        inputs = GR1_to_SIM_R.R_change(inputs)
        # print(inputs)
        result = FK_GR1_7DOF.compute_fk(inputs)
        GR1_position, GR1_orientation, GR1_joint3, GR1_joint4 = result
        inputs2 = use_model(inputs)
        result2 = FK_G1_7DOF.compute_fk(inputs2)
        G1_position, G1_orientation, G1_joint3, G1_joint4 = result2
        dif2 = GR1_position - G1_position
        dif2_norm = torch.norm(dif2)
        sum = sum + dif2_norm
        outputs = G1_to_SIM_R.R_change(inputs2)
        goal_list.append(outputs)
        if dif2_norm < 0.005:
            a0 = a0 + 1
        if dif2_norm < 0.01:
            a1 = a1 + 1
        if dif2_norm < 0.03:
            a2 = a2 + 1
        if dif2_norm < 0.05:
            a3 = a3 + 1
    goal_dataset = torch.stack(goal_list, dim=0)
    torch.save(goal_dataset, "./goal_dataset_0319.pt")
    print(a0, a1, a2,a3,sum/epochs)
