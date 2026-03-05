import torch

import numpy as np

class FK_7DOF:
    def __init__(self):
        # DH parameters from Table 1
        self.d = torch.tensor([0., 0., 0.1877, 0.0, 0.138, 0.0, 0., 0.])
        self.a = torch.tensor([0., 0.,  0.0081, 0.0, 0.0, 0.0, 0.046, 0.036])
        self.alpha = torch.tensor([0., torch.pi/2,-torch.pi/2, -torch.pi/2, torch.pi/2, -torch.pi/2, -torch.pi/2, 0.])


    def compute_fk(self, intermediate_outputs):

        theta = [intermediate_outputs[0]-torch.pi/2, intermediate_outputs[1]-0.24-torch.pi/2, intermediate_outputs[2]-torch.pi/2,
                 intermediate_outputs[3]-torch.pi/2, intermediate_outputs[4]-torch.pi/2,intermediate_outputs[5]-torch.pi/2,intermediate_outputs[6],torch.tensor(0)]
        T00 =torch.tensor([[0.,-0.275,0.961,0.15596],[1.,0.,0.,-0.0025],[0.,0.961,0.275,-0.07],[0.,0.,0.,1.]])
        T01 = THT(theta[0], self.a[0], self.d[0], self.alpha[0])
        T12 = THT(theta[1], self.a[1], self.d[1], self.alpha[1])
        T23 = THT(theta[2], self.a[2], self.d[2], self.alpha[2])
        T34 = THT(theta[3], self.a[3], self.d[3], self.alpha[3])
        T45 = THT(theta[4], self.a[4], self.d[4], self.alpha[4])
        T56 = THT(theta[5], self.a[5], self.d[5], self.alpha[5])
        T67 = THT(theta[6], self.a[6], self.d[6], self.alpha[6])
        T78 = THT(theta[7], self.a[7], self.d[7], self.alpha[7])
        T0 = torch.mm(T00, T01)
        T1 = torch.mm(T0, T12)
        T2 = torch.mm(T1, T23)
        T3 = torch.mm(T2, T34)
        T4 = torch.mm(T3, T45)
        T5 = torch.mm(T4, T56)
        T6 = torch.mm(T5, T67)
        T7 = torch.mm(T6, T78)
        position_7dof = T7[0:3,3]
        orientation_7dof = T7[0:3,0:3]
        joint3_7dof = T3[0:3,3]
        joint4_7dof = T5[0:3,3]

        return position_7dof, orientation_7dof, joint3_7dof, joint4_7dof

def THT(Theta, A, D, Alpha):

    T = torch.stack([
        torch.stack([torch.cos(Theta), -torch.sin(Theta), torch.tensor(0.0), A], dim=-1),
        torch.stack([torch.sin(Theta) * torch.cos(Alpha), torch.cos(Theta) * torch.cos(Alpha), - torch.sin(Alpha),
                     -D * torch.sin(Alpha)], dim=-1),
        torch.stack([torch.sin(Alpha) * torch.sin(Theta),torch.cos(Theta) * torch.sin(Alpha), torch.cos(Alpha), D * torch.cos(Alpha)], dim=-1),
        torch.tensor([0.0, 0.0, 0.0, 1.0])
    ])
    return T
if __name__ == '__main__':
    # theta_list = torch.tensor([1.0405,   1.7923,  1.1068,  -0.0665,  -0.5062,  -3.0588,  -0.0038])
    theta_list = torch.tensor([0.,0.,  0., 0., 0., 0., 0.])
    DOF_7 = FK_7DOF()
    result = DOF_7.compute_fk(theta_list)
    print(result)