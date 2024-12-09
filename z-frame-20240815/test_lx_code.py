import numpy as np

# 定义变量
p_RAS = np.array([-35.6, 8.9, -40.6]).reshape(3, 1)  # slicer选点

p_gold = np.array([-285.42, 87.38, 151.99]).reshape(3, 1)  # NDI采集

deg = np.pi / 180

t_RAS_Z = np.array([-4.4891, -17.784, 86.4622]).reshape(3, 1)  # Z在RAS上偏移

R_RAS_Z = np.array([[0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0]])  # Z在RAS上旋转

R_Z_RAS = np.linalg.inv(R_RAS_Z)

T_RASZ = np.vstack((np.hstack((R_RAS_Z, t_RAS_Z)), np.array([0, 0, 0, 1])))

T_ZRAS = np.linalg.inv(T_RASZ)

R_ROB_Z = np.array([[0, 1, 0],
                    [0, 0, -1],
                    [-1, 0, 0]])

t_ROB_Z = np.array([-396, 60, 149]).reshape(3, 1)  # Z在板子上的偏移

T_ROBZ = np.vstack((np.hstack((R_ROB_Z, t_ROB_Z)), np.array([0, 0, 0, 1])))

t_z_p = R_Z_RAS @ p_RAS + T_ZRAS[:3, 3].reshape(3, 1)
t_robp = R_ROB_Z @ t_z_p + t_ROB_Z
print(t_robp)
plterror = np.linalg.norm(p_gold - t_robp)

# 输出误差
print("Plt Error:", plterror)
