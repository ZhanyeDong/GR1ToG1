import mujoco as mj
import mujoco.viewer as viewer
import torch
import FK_G1_7DOF
import numpy as np
def R_change(theta_list):
    theta_list_change = torch.stack([-theta_list[0], -theta_list[1], -theta_list[2], theta_list[3],
                                     -theta_list[4], theta_list[6], theta_list[5]])
    return theta_list_change


if __name__ == '__main__':
    theta_list = torch.tensor([0.5, 0.5,0.5, 0.5, 0.,0.,0.,0,0,0,0,0,0,0])
    DOF_7 = FK_G1_7DOF.FK_7DOF()
    result = DOF_7.compute_fk(theta_list[0:7])
    print('DH',result)
    MJ_XML_PATH2 = r"D:\learning source\tamp\g1\g1_change.xml"
    mj_model2 = mj.MjModel.from_xml_path(MJ_XML_PATH2)
    data2 = mj.MjData(mj_model2)
    data2.ctrl = theta_list
    data2.ctrl[0:7] = R_change(theta_list)
    for _ in range(1000):
        mj.mj_step(mj_model2, data2)
        mj.mj_forward(mj_model2, data2)
    # viewer.launch(mj_model2, data2)
    id = data2.body("r_shoulder_pitch_link").id
    print('mujoco')
    for i in range(8):
        print(data2.body(id+i).xpos,end="\n")

    ee_quat = data2.body(26).xquat  # 四元数[w, x, y, z]
    w = ee_quat[0]
    x = ee_quat[1]
    y = ee_quat[2]
    z = ee_quat[3]
    norm = np.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    R = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])
    print('R7:',R)
    # viewer.launch(mj_model2, data2)
# if __name__ == '__main__':
#     theta_list = torch.tensor([ 1.8114,  -4.3805,  -2.7383, -13.2812,   1.9395,   0.5099,  -6.6303])
#     DOF_7 = FK_7DOF.FK_7DOF()
#     result = DOF_7.compute_fk(theta_list)
#     print(L_change(theta_list))

