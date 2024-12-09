import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from Registration import Registration
from scipy.spatial.transform import Rotation as R
import cv2
import os
import yaml
import utils

def main(path, slice_idx, Ld):
    vol_path = path

    '''
        demension:     图像维度
        origin:        图像在世界坐标系下的原点
        spacing:       空间分辨率
        direction:     图像坐标系到解剖学坐标系之间的转换关系
    '''
    # Get the list of DICOM files in the folder
    dicom_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.dcm')]

    # 读dicom文件并获取原始信息
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    dimension, origin, spacing, direction = utils.get_dicom_info(image)
    print("{0:<30s}:".format("Image Origin in World coordinate"), origin)
    print("{0:<30s}:".format("Image Dimension"), dimension)
    print("{0:<30s}:".format("Spacing of pixels"), spacing)
    # 计算像素坐标系ijk到ras的旋转矩阵与位移
    ijk2ras_origin, ijk2ras_matrix = utils.get_ijk2ras_matrix(spacing, direction, origin)
    print("{0:<30s}:".format("Origin from IJK to RAS"), ijk2ras_origin)
    print("{0:<30s}:\n".format("Rotation matrix from IJK to RAS"), ijk2ras_matrix)
    print("----------------------------------------------------------")


    # TODO:确认slice_idx的转换关系
    slice_idx = slice_idx
    vol = sitk.GetArrayFromImage(image)
    img_slice = vol[:,:,slice_idx]
    plt.imshow(img_slice, cmap='gray')
    plt.show()

    cv2.imwrite("ori_slice.png", img_slice)

    # 从配置文件读取Z框架配置
    zframe_configration = utils.get_zframe_configration()

    # 创建Registration对象并初始化
    r = Registration(slice_idx, img_slice, dimension, ijk2ras_matrix, ijk2ras_origin, zframe_configration)
    T_Zframe2RAS, _, _ = r.register()

    return T_Zframe2RAS



if __name__ == '__main__':

    # Data file path
    vol_path = 'z-frame-20240815\Data\\tof3d_tra_2801'
    # The size of Z-frame(mm)
    Ld = 44.0
    # The index of the slice
    slice_idx = 262
    # Transformation matrix from Z frame coordinate system to RAS coordinate system
    T_Zframe2RAS = main(vol_path, slice_idx, Ld)
    # print(T_Zframe2RAS.T)

