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
    dimension = np.array([480, 480, 390])
    spacing = np.array([0.467, 0.467, 0.467])
    direction = np.array([[-0.0143, -0.0118, 0.9998],
                 [-0.9999, 0.0004, -0.0143],
                 [-0.0003,-0.9999,-0.0118]])
    origin = np.array([-88.3870621, 113.5876310, 80.2534637])

    # 计算像素坐标系ijk到ras的旋转矩阵与位移
    # ijk2ras_origin = origin
    ijk2ras_origin, ijk2ras_matrix = utils.get_ijk2ras_matrix(spacing, direction, origin)

    print("{0:<30s}:".format("Origin from IJK to RAS"), ijk2ras_origin)
    print("{0:<30s}:\n".format("Rotation matrix from IJK to RAS"), ijk2ras_matrix)
    print("----------------------------------------------------------")


    # TODO:确认slice_idx的转换关系
    slice_idx = 195


    vol = sitk.GetArrayFromImage(image)
    img_slice = vol[slice_idx, :, :]
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
    vol_path = 'z-frame-20240815\Data\\t1_gre_fsp3d_sag_iso0.7mm_ACS_501'
    # The size of Z-frame(mm)
    Ld = 44.0
    # The index of the slice
    slice_idx = 195
    # Transformation matrix from Z frame coordinate system to RAS coordinate system
    T_Zframe2RAS = main(vol_path, Ld)
    print(T_Zframe2RAS)

    p_RAS = np.array([-34.5, 17.3, -121.5]).reshape(3, 1) ##选点
    # p_gold = np.array([-258.85, 89.27, 169.79]).reshape(3, 1)  # NDI采集
    p_gold = np.array([-258.7442791013014,90.5427947150370,136.9940598863524])

    deg = np.pi / 180
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

    plterror = np.linalg.norm(p_gold - t_robp)
    # plterror_2 = np.linalg.norm(p_gold_2 - t_robp_2)
    # 输出误差
    print("Plt Error:", plterror)





