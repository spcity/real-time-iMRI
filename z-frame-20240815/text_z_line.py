import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成示例数据（centroid_points_2）
np.random.seed(0)
centroid_points_2 = np.random.rand(100, 3) * 10

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
point1 = mean
point2 = mean + principal_vector * 10

# 绘制三维散点图和拟合直线
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点
ax.scatter(centroid_points_2[:, 0], centroid_points_2[:, 1], centroid_points_2[:, 2], c='b', marker='o')

# 绘制直线
line_points = np.array([point1, point2])
ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], c='r', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Line Fitting using Least Squares')
plt.show()
