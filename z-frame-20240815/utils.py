import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml

def get_dicom_info(image):
    dimension = np.array(list(image.GetSize()))
    origin = np.array(list(image.GetOrigin()))
    spacing = np.array(list(image.GetSpacing()))
    direction = np.array(list(image.GetDirection())).reshape(3, 3)
    return dimension, origin, spacing, direction


def get_ijk2ras_matrix(spacing, ijk2lps_direction, origin):
    ijk2lps_matrix = ijk2lps_direction.copy()
    for i in range(spacing.size):
        ijk2lps_matrix[:, i] = ijk2lps_matrix[:, i] * spacing[i]

    # ijk2ras_origin = origin.copy()
    # ijk2ras_origin[0] = -ijk2ras_origin[0]
    # ijk2ras_origin[1] = -ijk2ras_origin[1]

    lps2ras_matrix = np.identity(3)
    lps2ras_matrix[0][0] = -1.0
    lps2ras_matrix[1][1] = -1.0
    lps2ras_matrix[2][2] = 1.0
    ijk2ras_matrix = np.matmul(lps2ras_matrix, ijk2lps_matrix)

    # TODO: 自动转换origin 目前还需要根据3D slicer里面的Origin值手动输入
    ijk2ras_origin = np.array([-87.4785080, 122.1938480, 91.0866165])#np.array([-90.7666702, 129.663666, 92.7629547])

    return ijk2ras_origin, ijk2ras_matrix

def get_zframe_configration():
    with open('z-frame-20240815\zframe_configration\config.yaml', 'r', encoding='utf-8') as f:
        zframe_configration = yaml.load(f.read(), Loader=yaml.FullLoader)
    return zframe_configration

def slice_preprocess(slice):
    # plt.imshow(slice, cmap='gray')  # 不进行形态学闭运算的结果
    # plt.show()
    # cv2.imwrite("origin_slice.png", slice)
    slice = (slice / np.max(slice)) * 255
    slice = np.uint8(slice)

    # 对比度增强
    slice = cv2.convertScaleAbs(slice, 2, 5)


    # 去噪
    slice = cv2.medianBlur(slice, 9)

    return slice

def get_binary(slice):
    # plt.imshow(slice, cmap='gray')  # 不进行形态学闭运算的结果
    # plt.show()
    # cv2.imwrite("origin_slice.png", slice)
    ret2, bi_slice = cv2.threshold(slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 5), np.uint8)
    # 形态学闭运算 填充内部空洞
    bi_slice_morph = cv2.morphologyEx(bi_slice, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)  # 形态学闭运算
    cv2.imwrite("bi_slice_morph.png", bi_slice_morph)


    # TODO: 自动去除非Z框架部分像素


    # 把纵坐标大于137的坐标对应的像素值都设置为0
    rows, cols = bi_slice_morph.shape
    for i in range(0, rows):
        for j in range(0, cols):
            if i < 28:
                bi_slice_morph[i, j] = bi_slice_morph[1, 1]
                bi_slice[i, j] = bi_slice[1, 1]
            if i > 141:
                bi_slice_morph[i, j] = bi_slice_morph[1, 1]
                bi_slice[i, j] = bi_slice[1, 1]
            if j > 360:
                bi_slice_morph[i, j] = bi_slice_morph[1, 1]
                bi_slice[i, j] = bi_slice[1, 1]
            # if j < 269:
            #     bi_slice_morph[i, j] = bi_slice_morph[1, 1]
            #     bi_slice[i, j] = bi_slice[1, 1]
                

    plt.imshow(bi_slice, cmap='gray')
    plt.show()

    return bi_slice, bi_slice_morph

def check_fiducial_geometry(centroid_Points):
    # 检查几何关系
    # 1、检查质心坐标是否在规定范围内

    # 2、检查顶点是否平行
    P1, P3, P5, P7 = np.array(centroid_Points[0]), np.array(centroid_Points[2]), \
        np.array(centroid_Points[4]), np.array(centroid_Points[6])

    # 计算两对对边向量71-53、13-75
    D71 = P7 - P1
    D53 = P5 - P3
    D13 = P1 - P3
    D75 = P7 - P5

    # 计算对边向量夹角cos值，cos_value = 内积 / 向量模的积
    cos_value = np.dot(D71, D53) / (np.linalg.norm(D71) * np.linalg.norm(D53))
    if (cos_value < 0):
        cos_value = -cos_value  # 只看锐角值
    if (cos_value < np.cos(10 * 3.14159 / 360)):
        raise Exception('对边角度大于{}'.format(10))

    cos_value = np.dot(D13, D75) / (np.linalg.norm(D13) * np.linalg.norm(D75))
    if (cos_value < 0):
        cos_value = -cos_value
    if (cos_value < np.cos(10 * 3.14159 / 360)):
        raise Exception('对边角度: {}, 大于{}'.format(np.arccos(cos_value), 10))

    return


def ijk_to_ras(ijk, origin, matrix, space):
    """像素坐标IJK到RAS的转化
    """
    ras = ijk * space
    ras = np.matmul(matrix, ras) + origin
    # ras = temp + origin
    return ras


# 0 -- array, 1 -- transpose
def get_rotation_matrix_by_3points(points, mode =1):
    #  以三个点在Z框架坐标系下的坐标来建立坐标系，以第三个点为建立坐标系的原点
    vx = points[0] - points[2]
    vy = points[1] - points[2]
    vz = np.cross(vx, vy)  # 叉乘
    vy = np.cross(vz, vx)

    vx = vx / np.linalg.norm(vx)
    vy = vy / np.linalg.norm(vy)
    vz = vz / np.linalg.norm(vz)

    #  计算建立坐标系到Z框架坐标系的旋转矩阵
    # TODO : 确认该处是否也使用transpose
    rotation_matrix = np.array([3, 3])
    if mode == 0:
        rotation_matrix = np.array([vx.tolist(), vy.tolist(), vz.tolist()])
    elif mode == 1:
        rotation_matrix = np.transpose([vx.tolist(), vy.tolist(), vz.tolist()])
    return rotation_matrix

