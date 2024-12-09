import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2  # OpenCV for image processing
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import DBSCAN
from scipy.optimize import leastsq
# 读取DICOM文件夹
def load_dicom_images(folder_path):
    dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
    dicom_files.sort()  # 根据文件名排序以确保切片顺序
    images = []
    for file in dicom_files:
        file_path = os.path.join(folder_path, file)
        ds = pydicom.dcmread(file_path)
        images.append(ds.pixel_array)
    return np.array(images)



# 去噪声处理
def denoise_images(images):
    denoised_images = []
    for img in images:
        # 使用高斯滤波器去噪
        denoised_img = cv2.GaussianBlur(img, (1, 1), 1)
        denoised_images.append(denoised_img)
    return denoised_images
def binarize_images(images, threshold=5):
    binarized_images = []
    for img in images:
        # Convert to binary image using threshold
        _, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        binarized_images.append(bin_img)
    return binarized_images

def slice_process(images):
    slice = np.array(images)
    pro_images=[]
    for slice in images:
        slice = (slice / np.max(slice)) * 255
        slice = np.uint8(slice)
        # 对比度增强
        # slice = cv2.convertScaleAbs(slice, 2, 5)
        # 去噪
        slice = cv2.medianBlur(slice, 5)
        ret2, bi_slice = cv2.threshold(slice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((1, 5), np.uint8)
        # # 形态学闭运算 填充内部空洞
        bi_slice_morph = cv2.morphologyEx(bi_slice, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)  # 形态学闭运算
        # cv2.imwrite("bi_slice_morph.png", bi_slice_morph)
        pro_images.append(bi_slice_morph)
    return pro_images

def binarize_3d_matrix(matrix, threshold):
    """
    对三维矩阵进行二值化处理。
    
    参数：
    matrix (numpy.ndarray): 输入的三维矩阵
    threshold (float): 二值化的阈值
    
    返回：
    numpy.ndarray: 二值化后的三维矩阵
    """
    # 创建一个与输入矩阵形状相同的矩阵，用于存储二值化结果
    binarized_matrix = np.zeros_like(matrix)
    
    # 对大于指定阈值的元素进行二值化处理
    binarized_matrix[matrix > threshold] = 255
    
    return binarized_matrix

# 创建查看器
def create_viewer(images):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    img_display = ax.imshow(images[0], cmap='gray')

    # 添加滑块
    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Slice', 0, len(images) - 1, valinit=0, valfmt='%0.0f')

    # 更新图像
    def update(val):
        slice_idx = int(slider.val)
        img_display.set_data(images[slice_idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

# 主程序
folder_path = 'z-frame-20240815\Data\\tof3d_tra_2801'  # 替换为你的DICOM文件夹路径
images = load_dicom_images(folder_path)
# 设置阈值
threshold = 50
# images[:26,:, :] = 0
# images[:,:, 0:160] = 0
# images[:,:, 300:] = 0
# images[28,:, :] = 0
# images[29,:, :] = 0
images[127:,:, :] = 0

images = slice_process(images)
images_np = np.array(images)

# 进行二值化处理
binarized_matrix = binarize_3d_matrix(images_np, threshold)
binarized_matrix = np.array(binarized_matrix)
print(binarized_matrix.shape)
binarized_matrix[0:27,:, :] = 0
# binarized_matrix[27,:, :] = 0
# 先进行维度交换，使得新的顺序为 (640, 550, 524)
transposed_images = np.transpose(binarized_matrix, (2, 0, 1))

# 将结果 reshape 为 (640, 550, 524)，其中每个二维数组的形状为 (550, 524)
reshaped_images = transposed_images.reshape(524,550, 640)

create_viewer(reshaped_images)




# # '''
# # 随机选择一个值为255的点
# points_255 = np.argwhere(binarized_matrix == 255)
# # # 随机选择10个不重复的点
# # selected_points = points_255[np.random.choice(points_255.shape[0], size=10, replace=False)]
# selected_points = np.array([[34,278,284],
#                            [34,247,345],
#                            [36,209,411],
#                            [79,255,280],
#                            [79,262,411],
#                            [123,291,288],
#                            [123,200,411]])
# # 指定一个距离范围
# distance_threshold = 50.0

# # 计算该点周围距离小于指定值的所有值为255的点
# def calculate_distance(point1, point2):
#     return np.linalg.norm(point1 - point2)


# # 输出随机选择的点
# print("随机选择的10个点的坐标为：")
# # 可视化散点和拟合的直线
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # random_point = points_255[np.random.choice(points_255.shape[0])]
# for i, points_10 in enumerate(selected_points, start=1):
#     print(f"Point {i}: {points_10}")
#     nearby_points = []
#     for point in points_255:
#         if calculate_distance(points_10, point) < distance_threshold:
#             nearby_points.append(point)
#     nearby_points = np.array(nearby_points)

#     if nearby_points.shape[0] < 10:
#         print("范围内点不足以拟合直线")
#     else:
#         # 拟合直线
#         x = nearby_points[:, 0]
#         y = nearby_points[:, 1]
#         z = nearby_points[:, 2]

#         def line_model(params, x):
#             return params[0] * x + params[1]

#         def residuals_y(params, y, x):
#             return y - line_model(params, x)

#         def residuals_z(params, z, x):
#             return z - line_model(params, x)

#         params_y, _ = leastsq(residuals_y, [0, 0], args=(y, x))
#         params_z, _ = leastsq(residuals_z, [0, 0], args=(z, x))

#         # 存储直线参数
#         fitted_line_params = [params_y, params_z]

#         # 生成拟合直线的点
#         # 固定直线长度
#         fixed_length = 150.0  # 固定长度的值
#         midpoint_x = (x.max() + x.min()) / 2  # 使用拟合点的中点作为中心
#         half_length = fixed_length / 2
#         x_fit = np.linspace(midpoint_x - half_length, midpoint_x + half_length, 100)
#         y_fit = params_y[0] * x_fit + params_y[1]
#         z_fit = params_z[0] * x_fit + params_z[1]
#         #打印或存储直线参数
#         print("Fitted Line Parameters (y = ax + b, z = cx + d):")
#         print(f"a: {params_y[0]}, b: {params_y[1]}")
#         print(f"c: {params_z[0]}, d: {params_z[1]}")
#         ax.scatter(x, y, z, color='blue', label='Data Points')

#         ax.plot(x_fit, y_fit, z_fit, color='red', label='Fitted Line')
#         # 设置绘图的空间大小与binarized_matrix的大小一致
# ax.set_xlim(0, binarized_matrix.shape[0])
# ax.set_ylim(0, binarized_matrix.shape[1])
# ax.set_zlim(0, binarized_matrix.shape[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # plt.legend()
# plt.show()

    #


# images[26] = np.zeros(images[26].shape)
# 去噪声处理
# images = denoise_images(images)


# # 二值化
# images = binarize_images(images)
# 先进行维度交换，使得新的顺序为 (640, 550, 524)
# transposed_images = np.transpose(images, (1, 0, 2))

# # 将结果 reshape 为 (640, 550, 524)，其中每个二维数组的形状为 (550, 524)
# reshaped_images = transposed_images.reshape(640, 550, 524)
# images = np.array(images)
# 获取非零元素的索引
# x, y, z = np.nonzero(binarized_matrix)
# # 创建三维图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制三维散点图
# ax.scatter(x, y, z, c='r', marker='o')

# # 设置轴标签
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# # 显示图形
# plt.show()

# create_viewer(images)
# '''
