import torch

class FK_7DOF:
    def __init__(self, device=None, dtype=torch.float32):
        # DH parameters from Table 1
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype
        self.d = torch.tensor([0., 0., 0.1877, 0.0, 0.138, 0.0, 0., 0.], device=self.device, dtype=self.dtype)
        self.a = torch.tensor([0., 0., 0.0081, 0.0, 0.0, 0.0, 0.046, 0.036], device=self.device, dtype=self.dtype)
        self.alpha = torch.tensor(
            [0., torch.pi / 2, -torch.pi / 2, -torch.pi / 2, torch.pi / 2, -torch.pi / 2, -torch.pi / 2, 0.],
            device=self.device,
            dtype=self.dtype,
        )
        self.T00 = torch.tensor(
            [[0., -0.275, 0.961, 0.15596], [1., 0., 0., -0.0025], [0., 0.961, 0.275, -0.1], [0., 0., 0., 1.]],
            device=self.device,
            dtype=self.dtype,
        )


    def compute_fk(self, intermediate_outputs):
        squeeze_out = False
        if intermediate_outputs.dim() == 1:
            intermediate_outputs = intermediate_outputs.unsqueeze(0)
            squeeze_out = True
        if intermediate_outputs.size(-1) != 7:
            raise ValueError("intermediate_outputs must have shape (7,) or (B, 7)")

        zero = torch.zeros(intermediate_outputs.size(0), device=intermediate_outputs.device, dtype=intermediate_outputs.dtype)
        theta = [
            intermediate_outputs[:, 0] - torch.pi / 2,
            intermediate_outputs[:, 1] - 0.24 - torch.pi / 2,
            intermediate_outputs[:, 2] - torch.pi / 2,
            intermediate_outputs[:, 3] - torch.pi / 2,
            intermediate_outputs[:, 4] - torch.pi / 2,
            intermediate_outputs[:, 5] - torch.pi / 2,
            intermediate_outputs[:, 6],
            zero,
        ]
        T01 = THT(theta[0], self.a[0], self.d[0], self.alpha[0])
        T12 = THT(theta[1], self.a[1], self.d[1], self.alpha[1])
        T23 = THT(theta[2], self.a[2], self.d[2], self.alpha[2])
        T34 = THT(theta[3], self.a[3], self.d[3], self.alpha[3])
        T45 = THT(theta[4], self.a[4], self.d[4], self.alpha[4])
        T56 = THT(theta[5], self.a[5], self.d[5], self.alpha[5])
        T67 = THT(theta[6], self.a[6], self.d[6], self.alpha[6])
        T78 = THT(theta[7], self.a[7], self.d[7], self.alpha[7])
        batch_size = intermediate_outputs.size(0)
        T0 = torch.bmm(self.T00.unsqueeze(0).expand(batch_size, -1, -1), T01)
        T1 = torch.bmm(T0, T12)
        T2 = torch.bmm(T1, T23)
        T3 = torch.bmm(T2, T34)
        T4 = torch.bmm(T3, T45)
        T5 = torch.bmm(T4, T56)
        T6 = torch.bmm(T5, T67)
        T7 = torch.bmm(T6, T78)
        position_7dof = T7[:, 0:3, 3]
        orientation_7dof = T7[:, 0:3, 0:3]
        joint3_7dof = T3[:, 0:3, 3]
        joint4_7dof = T7[:, 0:3, 3]

        if squeeze_out:
            return (
                position_7dof.squeeze(0),
                orientation_7dof.squeeze(0),
                joint3_7dof.squeeze(0),
                joint4_7dof.squeeze(0),
            )

        return position_7dof, orientation_7dof, joint3_7dof, joint4_7dof

def THT(Theta, A, D, Alpha):
    if Theta.dim() != 1:
        raise ValueError("Theta must have shape (B,)")

    A = torch.as_tensor(A, device=Theta.device, dtype=Theta.dtype)
    D = torch.as_tensor(D, device=Theta.device, dtype=Theta.dtype)
    Alpha = torch.as_tensor(Alpha, device=Theta.device, dtype=Theta.dtype)

    zeros = torch.zeros_like(Theta)
    ones = torch.ones_like(Theta)
    cos_t = torch.cos(Theta)
    sin_t = torch.sin(Theta)
    sin_a = torch.sin(Alpha)
    cos_a = torch.cos(Alpha)

    T = torch.stack([
        torch.stack([cos_t, -sin_t, zeros, A.expand_as(Theta)], dim=-1),
        torch.stack([sin_t * cos_a, cos_t * cos_a, (-sin_a).expand_as(Theta), (-D * sin_a).expand_as(Theta)], dim=-1),
        torch.stack([sin_a * sin_t, cos_t * sin_a, cos_a.expand_as(Theta), (D * cos_a).expand_as(Theta)], dim=-1),
        torch.stack([zeros, zeros, zeros, ones], dim=-1),
    ], dim=1)
    return T
if __name__ == '__main__':
    # theta_list = torch.tensor([1.0405,   1.7923,  1.1068,  -0.0665,  -0.5062,  -3.0588,  -0.0038])
    theta_list = torch.tensor([0.,0.,  0., 0., 0., 0., 0.])
    DOF_7 = FK_7DOF()
    result = DOF_7.compute_fk(theta_list)

    print(result)
