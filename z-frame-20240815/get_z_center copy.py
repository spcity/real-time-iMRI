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
from scipy.optimize import minimize
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

def least_squares_fit(points):
    def objective_func(params):
        a, b, c, d = params
        residuals = []
        for x, y, z in points:
            residuals.append(a*x + b*y + c*z + d)
        return residuals
    
    p0 = [1, 1, 1, 0]  # 初始参数估计值
    params, _ = leastsq(objective_func, p0)
    a, b, c, d = params
    return a, b, c, d
# 使用最小二乘法拟合直线
def fit_line(points):
    x, y, z = points.T
    A = np.column_stack((x, y, np.ones(len(x))))
    params, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = params
    return a, b, c



def get_line(centroid_points_2):
    # 计算点的均值
    mean = np.mean(centroid_points_2, axis=0)

    # 中心化点集
    centered_points = centroid_points_2 - mean

    # 计算协方差矩阵
    cov_matrix = np.cov(centered_points, rowvar=False)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 找到最大的特征值对应的特征向量（主方向）
    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]

    # 创建一个点到直线的线段
    point1 = mean - principal_vector * 50
    point2 = mean + principal_vector * 50
    return point1,point2,mean
def line_params(p1, p2):
    """Return the direction vector and a point on the line."""
    direction = p2 - p1
    return direction, p1
def distance_point_to_point(p1, p2):
    return np.linalg.norm(p1 - p2)
def closest_points_on_lines(p2_1, p2_2, p4_1, p4_2):
    # Define direction vectors
    d1 = p2_2 - p2_1
    d2 = p4_2 - p4_1
    
    def objective(params):
        t, s = params
        point1 = p2_1 + t * d1
        point2 = p4_1 + s * d2
        return distance_point_to_point(point1, point2)
    
    # Minimize the distance function
    result = minimize(objective, [0, 0], bounds=[(-np.inf, np.inf), (-np.inf, np.inf)])
    t_opt, s_opt = result.x
    
    closest_point_1 = p2_1 + t_opt * d1
    closest_point_2 = p4_1 + s_opt * d2
    
    return closest_point_1, closest_point_2

def midpoint(p1, p2):
    """Compute the midpoint of two points."""
    return (p1 + p2) / 2

# 主程序
folder_path = 'z-frame-20240815\Data\\t1_gre_fsp3d_sag_iso0.7mm_ACS_201'  # 替换为你的DICOM文件夹路径
images = load_dicom_images(folder_path)
# 设置阈值
threshold = 5
# images[0:146,:, :] = 0
# images[:,:, 0:160] = 0
# images[:,:, 300:] = 0
# images[28,:, :] = 0
# images[29,:, :] = 0
images[299:,:, :] = 0

images = slice_process(images)
images_np = np.array(images)

# 进行二值化处理
binarized_matrix = binarize_3d_matrix(images_np, threshold)
binarized_matrix = np.array(binarized_matrix)
print(binarized_matrix.shape)
binarized_matrix[:,168:, :] = 0
# binarized_matrix[/,:, :] = 0
# 先进行维度交换，使得新的顺序为 (640, 550, 524)
# transposed_images = np.transpose(binarized_matrix, (2, 0, 1))

# # 将结果 reshape 为 (640, 550, 524)(480,480,390)，其中每个二维数组的形状为 (550, 524)
# reshaped_images = transposed_images.reshape(390,480, 480)

# create_viewer(binarized_matrix)
# '''

centroid_points_2 = []
centroid_points_4 = []
centroid_points_6 = []
for slice_index in range(192, 202):
    bi_slice = binarized_matrix[slice_index,:,:]#utils.get_binary(self.img_slice)
    # bi_slice_name = 'bi_slice_'+str(slice_index)+'.png'
    # cv2.imwrite(bi_slice_name, bi_slice)
    # 连通域
    contours, opt = cv2.findContours(bi_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 画出边缘线
    centroid_slice = np.zeros((binarized_matrix.shape[1], binarized_matrix.shape[2]))  # 新建图像用于绘制外轮廓以及质心
    cv2.drawContours(centroid_slice, contours, -1, (255, 255, 255), 1)
    # cv2.imshow('centroid_slice', centroid_slice)
    # central_name = 'centroid_slice_'+str(slice_index)+'.png'
    # cv2.imwrite(central_name, centroid_slice)
    # 计算质心
    centroid_points = []
    i = 1
    for c in contours:
        # if c.shape[0] > self.dimension[0] / 5:
        #     continue
        # 计算图像距
        M = cv2.moments(c)
        # 计算质心x，y坐标 -- 中心距
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid_points.append([cX, cY])
        i = i + 1

    centroid_points = sorted(centroid_points, key=lambda x: x[0])
    # 缺口朝下
    centroid_points = sorted(centroid_points[0: 3], key=lambda x: x[1], reverse=True) + [centroid_points[3]] + sorted(centroid_points[4:], key=lambda x: x[1])
    # 缺口朝上
    # centroid_points = sorted(centroid_points[0: 3], key=lambda x: x[1]) + [centroid_points[3]] + sorted(centroid_points[4:], key=lambda x: x[1], reverse=True)



    col_gap = centroid_points[0][1] - centroid_points[1][1]
    for n, point in enumerate(centroid_points):
        cv2.circle(centroid_slice, point, 1, (255, 255, 255), -1)

        # 根据质心纵向间隙确认标号偏移
        cv2.putText(centroid_slice, 'P' + str(n + 1), (point[0], point[1] - int(col_gap * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 标号
        new_point = [slice_index,point[0],point[1]]
        if n+1==2:
            centroid_points_2.append(new_point)
        if n+1==4:
            centroid_points_4.append(new_point)
        if n+1==6:
            centroid_points_6.append(new_point)

    check_fiducial_geometry(centroid_points)
    centroid_points = np.array(centroid_points)
    # plt.figure(figsize=(12, 4))
    # plt.imshow(centroid_slice, cmap='gray')
    # # plt.tight_layout()
    # plt.show()
centroid_points_2 = np.array(centroid_points_2)
centroid_points_4 = np.array(centroid_points_4)
centroid_points_6 = np.array(centroid_points_6)
# 获取直线参数
# line2_params = least_squares_fit(centroid_points_2)
# line4_params = least_squares_fit(centroid_points_4)
# line6_params = least_squares_fit(centroid_points_6)

# 计算直线参数
# a2, b2, c2, d2 = least_squares_fit(centroid_points_2)
# a4, b4, c4, d4 = least_squares_fit(centroid_points_4)
# a6, b6, c6, d6 = least_squares_fit(centroid_points_6)

# # 假设我们有以下三条直线的参数
# line2_params = (a2, b2, c2, d2)
# line4_params = (a4, b4, c4, d4)
# line6_params = (a6, b6, c6, d6)
# 拟合三条直线
# a2, b2, c2 = fit_line(centroid_points_2)
# a4, b4, c4 = fit_line(centroid_points_4)
# a6, b6, c6 = fit_line(centroid_points_6)
point_2_1,point_2_2,point_2_mean = get_line(centroid_points_2)
point_4_1,point_4_2,point_4_mean = get_line(centroid_points_4)
point_6_1,point_6_2,point_6_mean = get_line(centroid_points_6)

# Compute the closest points on the lines
closest_point_2_6_1, closest_point_2_6_2 = closest_points_on_lines(point_2_1, point_2_2, point_6_1, point_6_2)

# Compute the midpoint of the closest points
mid_point_2_6 = midpoint(closest_point_2_6_1, closest_point_2_6_2)

# 计算中横线与底边的最近点
closest_point_26_4_1, closest_point_26_4_2 = closest_points_on_lines(closest_point_2_6_1, closest_point_2_6_2, point_4_1, point_4_2)
# 创建 3D 图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(centroid_points_2[:, 0], centroid_points_2[:, 1], centroid_points_2[:, 2], label='centroid_points_2')
ax.scatter(centroid_points_4[:, 0], centroid_points_4[:, 1], centroid_points_4[:, 2], label='centroid_points_4')
ax.scatter(centroid_points_6[:, 0], centroid_points_6[:, 1], centroid_points_6[:, 2], label='centroid_points_6')
ax.scatter(mid_point_2_6[0], mid_point_2_6[1], mid_point_2_6[2], c='b', label='mid_point_2_6')


# 绘制直线
line_points_2 = np.array([point_2_1, point_2_2])
ax.plot(line_points_2[:, 0], line_points_2[:, 1], line_points_2[:, 2], c='r', marker='o')
line_points_4 = np.array([point_4_1, point_4_2])
ax.plot(line_points_4[:, 0], line_points_4[:, 1], line_points_4[:, 2], c='r', marker='o')
line_points_6 = np.array([point_6_1, point_6_2])
ax.plot(line_points_6[:, 0], line_points_6[:, 1], line_points_6[:, 2], c='r', marker='o')

line_2_6 = np.array([closest_point_2_6_1, closest_point_2_6_2])
ax.plot(line_2_6[:, 0], line_2_6[:, 1], line_2_6[:, 2], c='g', marker='o', label='line_2_6')
line_26_4 = np.array([closest_point_26_4_1, closest_point_26_4_2])
ax.plot(line_26_4[:, 0], line_26_4[:, 1], line_26_4[:, 2], c='b', marker='o', label='line_26_4')
allpoint = np.array([closest_point_2_6_1,closest_point_2_6_2,closest_point_26_4_2])

print('line center 1, g',closest_point_2_6_1)
print('line center 2, g',closest_point_2_6_2)
print('mid center',mid_point_2_6)
print('line center 3',closest_point_26_4_1)
print('line center 4',closest_point_26_4_2)
print('allpoint',allpoint)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Line Fitting using Least Squares')
plt.legend()
plt.show()


'''
# 计算line2和line4的交点
intersection_24 = line_intersection(line2_params, line4_params)

# 计算line4和line6的交点
intersection_46 = line_intersection(line4_params, line6_params)

# 打印结果
print("line2和line4的交点:", intersection_24)
print("line4和line6的交点:", intersection_46)






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
