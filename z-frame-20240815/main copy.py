import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from Registration import Registration
from scipy.spatial.transform import Rotation as R
import cv2
import os
import yaml
import utils
import pydicom

def main(path, Ld):
    vol_path = path

    '''
        demension:     图像维度
        origin:        图像在世界坐标系下的原点
        spacing:       空间分辨率
        direction:     图像坐标系到解剖学坐标系之间的转换关系
    '''
    # Get the list of DICOM files in the folder
    dicom_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.dcm')]
    # dicom_files.sort()

    # 读dicom文件并获取原始信息
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    # print(image)
    # dimension, origin, spacing, direction = utils.get_dicom_info(image)
    dimension =np.array([524, 640, 550])
    spacing = np.array([0.343630731, 0.343630731, 0.5])
    direction = np.array([[-1.000,0.0012,-0.0056],
                 [-0.0011,-1.0000,-0.0069],
                 [-0.0056,-0.0069,1.0000]])
    origin = np.array([86.5182495, 81.2109229, -139.5489350])

    # dimension, origin, spacing, direction = utils.get_dicom_info(image)
    # dimension = dimension2
    # spacing = spacing2
    # direction3 = -direction2
    # direction3[2,:]= direction2[2,:]
    # origin = 


    # 计算像素坐标系ijk到ras的旋转矩阵与位移
    # ijk2ras_origin = origin
    ijk2ras_origin, ijk2ras_matrix = utils.get_ijk2ras_matrix(spacing, direction, origin)

    print("{0:<30s}:".format("Origin from IJK to RAS"), ijk2ras_origin)
    print("{0:<30s}:\n".format("Rotation matrix from IJK to RAS"), ijk2ras_matrix)
    print("----------------------------------------------------------")


    # TODO:确认slice_idx的转换关系
    slice_idx = 262 #dimension[0]//2
    vol = sitk.GetArrayFromImage(image)
    img_slice = vol[:, :, slice_idx]
    # plt.imshow(img_slice, cmap='gray')
    # plt.show()

    # 从配置文件读取Z框架配置
    zframe_configration = utils.get_zframe_configration()

    # 创建Registration对象并初始化
    r = Registration(slice_idx, img_slice, dimension, ijk2ras_matrix, ijk2ras_origin, zframe_configration, origin, direction,spacing)
    T_Zframe2RAS, _, _ = r.register()
    print(T_Zframe2RAS)

    return T_Zframe2RAS



if __name__ == '__main__':

    # Data file path
    vol_path = 'z-frame-20240815\Data\\tof3d_tra_2801'
    # The size of Z-frame(mm)
    Ld = 44.0
    # The index of the slice
    slice_idx = 262
    # Transformation matrix from Z frame coordinate system to RAS coordinate system
    T_Zframe2RAS = main(vol_path, Ld)
    # print(T_Zframe2RAS.T)

########NDI
# # 定义变量
p_RAS = np.array([-61.9, -47.2, 57.9]).reshape(3, 1)  # slicer选点
# p_RAS = np.array([207.78805628,43.5631813,  -51.92346244]).reshape(3,1)
# p_RAS_2 = np.array([-61.2, -49.3, -55.2]).reshape(3, 1)  # slicer选点
# p_RAS_2 = np.array([58.4, -47, 57.6]).reshape(3, 1)  # slicer选点
p_RAS_2 = np.array([56.4,-48.4,59.6]).reshape(3, 1)  # slicer选点
p_RAS_3 = np.array([-59.8,-48,-60.2]).reshape(3, 1)  # slicer选点
p_RAS_4 = np.array([59.1,-47.2,-58.4]).reshape(3, 1)  # slicer选点

p_RAS_5 = np.array([-62.2,-29.7,57.9]).reshape(3, 1)  # slicer选点
p_RAS_6 = np.array([-61.6,-31.1,-59.5]).reshape(3, 1)  # slicer选点
p_RAS_7 = np.array([58.4,-29,58]).reshape(3, 1)  # slicer选点
p_RAS_8 = np.array([59,-29.6,-59.6]).reshape(3, 1)  # slicer选点

p_gold = np.array([-365.23, 113.06, 118.14]).reshape(3, 1)  # NDI采集
p_gold_2 = np.array([-365.42, -5.21, 118.36]).reshape(3, 1)  # NDI采集
p_gold_3 = np.array([-246.2,113.6,120.79]).reshape(3, 1)  # NDI采集
p_gold_4 = np.array([-246.20,-5.22,119.52]).reshape(3, 1)  # NDI采集

p_gold_5 = np.array([-366.41, 113.17, 136.19]).reshape(3, 1)  # NDI采集
p_gold_7 = np.array([-366.24, -6.24, 135.94]).reshape(3, 1)  # NDI采集
p_gold_6 = np.array([-246.25,113.98,137.49]).reshape(3, 1)  # NDI采集
p_gold_8 = np.array([-245.98,-6.31,136.26]).reshape(3, 1)  # NDI采集

deg = np.pi / 180

# t_RAS_Z = np.array([-4.4891, -17.784, 86.4622]).reshape(3, 1)  # Z在RAS上偏移
t_RAS_Z =  T_Zframe2RAS[0:3,3].reshape(3, 1) 

# R_RAS_Z = np.array([[0, 0, 1],
#                     [-1, 0, 0],
#                     [0, -1, 0]])  # Z在RAS上旋转

R_RAS_Z = T_Zframe2RAS[0:3, 0:3]
R_Z_RAS = np.linalg.inv(R_RAS_Z)

T_RASZ = np.vstack((np.hstack((R_RAS_Z, t_RAS_Z)), np.array([0, 0, 0, 1])))

T_ZRAS = np.linalg.inv(T_RASZ)

R_ROB_Z = np.array([[0, 1, 0],
                    [0, 0, -1],
                    [-1, 0, 0]])

t_ROB_Z = np.array([-396, 56.9, 124.75]).reshape(3, 1)  # Z在板子上的偏移

T_ROBZ = np.vstack((np.hstack((R_ROB_Z, t_ROB_Z)), np.array([0, 0, 0, 1])))

t_z_p = R_Z_RAS @ p_RAS + T_ZRAS[:3, 3].reshape(3, 1)
t_robp = R_ROB_Z @ t_z_p + t_ROB_Z
print(t_robp)

t_z_p_2 = R_Z_RAS @ p_RAS_2 + T_ZRAS[:3, 3].reshape(3, 1)
t_robp_2 = R_ROB_Z @ t_z_p_2 + t_ROB_Z
print(t_robp_2)

plterror = np.linalg.norm(p_gold - t_robp)
plterror_2 = np.linalg.norm(p_gold_2 - t_robp_2)
# 输出误差
print("Plt Error:", plterror)
# print("Plt Error2:", plterror_2)

# t_z_p_3 = R_Z_RAS @ p_RAS_3 + T_ZRAS[:3, 3].reshape(3, 1)
# t_robp_3 = R_ROB_Z @ t_z_p_3 + t_ROB_Z
# print(t_robp_3)

# t_z_p_4 = R_Z_RAS @ p_RAS_4 + T_ZRAS[:3, 3].reshape(3, 1)
# t_robp_4 = R_ROB_Z @ t_z_p_4 + t_ROB_Z
# print(t_robp_4)

# plterror_3 = np.linalg.norm(p_gold_3 - t_robp_3)
# plterror_4 = np.linalg.norm(p_gold_4 - t_robp_4)
# # 输出误差
# print("Plt Error3:", plterror_3)
# print("Plt Error4:", plterror_4)

# t_z_p_5 = R_Z_RAS @ p_RAS_5 + T_ZRAS[:3, 3].reshape(3, 1)
# t_robp_5 = R_ROB_Z @ t_z_p_5 + t_ROB_Z
# print(t_robp_5)

# t_z_p_6 = R_Z_RAS @ p_RAS_6 + T_ZRAS[:3, 3].reshape(3, 1)
# t_robp_6 = R_ROB_Z @ t_z_p_6 + t_ROB_Z
# print(t_robp_6)

# plterror_5 = np.linalg.norm(p_gold_5 - t_robp_5)
# plterror_6 = np.linalg.norm(p_gold_6 - t_robp_6)
# # 输出误差
# print("Plt Error5:", plterror_5)
# print("Plt Error6:", plterror_6)

# t_z_p_7 = R_Z_RAS @ p_RAS_7 + T_ZRAS[:3, 3].reshape(3, 1)
# t_robp_7 = R_ROB_Z @ t_z_p_7 + t_ROB_Z
# print(t_robp_7)

# t_z_p_8 = R_Z_RAS @ p_RAS_8 + T_ZRAS[:3, 3].reshape(3, 1)
# t_robp_8 = R_ROB_Z @ t_z_p_8 + t_ROB_Z
# print(t_robp_8)

# plterror_7 = np.linalg.norm(p_gold_7 - t_robp_7)
# plterror_8 = np.linalg.norm(p_gold_8 - t_robp_8)
# # 输出误差
# print("Plt Error7:", plterror_7)
# print("Plt Error8:", plterror_8)

# avg=np.mean(np.array([ plterror_2, plterror_4, plterror_6, plterror_8]))
# print("avg",avg)
# avg.append(plterror)
# avg.append(plterror_2)