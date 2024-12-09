import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
import utils


class Registration:
    def __init__(self, slice_idx, input_slice, input_dim, input_ijk2ras_matrix, input_ijk2ras_origin, zframe_configration, origin, matrix, space):
        self.slice_idx = slice_idx
        self.img_slice = input_slice
        self.dimension = input_dim
        self.ijk2ras_matrix = input_ijk2ras_matrix
        self.ijk2ras_origin = input_ijk2ras_origin
        self.zframe_configration = zframe_configration
        self.origin = origin
        self.direction = matrix
        self.space = space

    def register(self):
        # 预处理
        self.img_slice = utils.slice_preprocess(self.img_slice)#对比度增强，去噪

        centroid_points, centroid_slice = self.get_centroid_points()
        print("{0:<30s}:\n".format("7 centroid Points"), centroid_points)
        # centroid_points=[]
        # T_Zframe2RAS is transformation matrix from Z frame coordinate system to RAS coordinate system;
        # p_zframe is Coordinates of the three reference points in the Z frame coordinate system
        # p_ras is Coordinates of the three reference points in the RAS coordinate system
        T_Zframe2RAS, p_zframe, p_ras = self.localize_frame(self.slice_idx, centroid_points)
        print("{0:<30s}:\n".format("Zframe2RAS transformation matrix"), T_Zframe2RAS)

        return T_Zframe2RAS, p_zframe, p_ras

    def get_centroid_points(self):
        bi_slice, bi_slice_morph = utils.get_binary(self.img_slice)

        # 连通域
        contours, opt = cv2.findContours(bi_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 画出边缘线
        centroid_slice = np.zeros((self.dimension[0], self.dimension[1]))  # 新建图像用于绘制外轮廓以及质心
        cv2.drawContours(centroid_slice, contours, -1, (255, 255, 255), 1)
        # cv2.imshow('centroid_slice', centroid_slice)
        cv2.imwrite('centroid_slice.png', centroid_slice)

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


        utils.check_fiducial_geometry(centroid_points)
        centroid_points = np.array(centroid_points)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Original image')
        plt.imshow(self.img_slice, cmap='gray')  # 不进行形态学闭运算的结果
        plt.subplot(1, 3, 2)
        plt.title('Binary Slice')
        plt.imshow(bi_slice, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('Centroid Slice')
        plt.imshow(centroid_slice, cmap='gray')
        plt.tight_layout()
        plt.show()

        return centroid_points, centroid_slice

    def localize_frame(self, slice_idx, centroid_points):
        ''' -----------------计算对角线与截面交点在物理z框架坐标系中的位置----------------------
            对角线fiducial的原点和方向向量（使用numpy储存向量）
            origin_vector : 每三组点中，Pz3所在边的末端点 在z框架中的坐标
            direction_vector： 对角线在z框架坐标下的方向
        '''
        direction_vector = np.array(self.zframe_configration['direction_vector'])
        origin_vector = np.array(self.zframe_configration['origin_vector'])

        # ----------计算RAS坐标系与ZFrame坐标系之间的Transformation-------------
        # 计算三个参考点建立的坐标系与Z框架坐标系之间的转换关系
        # 空列表p_zframe储存p2、p4、p6在Zframe coordinate中的位置p2f,p4f,p6f
        # point1 = np.array([256.,87.87729395,195.2273])
        # point2 = np.array([303.21551288,37., 195.2273])
        # point3 = np.array([350.,87.8772939, 195.2273])
        # point1 = np.array([257.,89,195])
        # point2 = np.array([304,38., 195])
        # point3 = np.array([351.,88, 195])
        # point1 = np.array([257,123.60420716,194])
        # point2 = np.array([305.49108947 , 74.1098443, 194])
        # point3 = np.array([351.03978143, 124.49349387, 194])

        # point = np.array([0,0,0])
        # pz0=[point, point1, point, point2, point,point3,point]
        pz0 = centroid_points
        
        p_zframe = []
        start_idx = 0
        # 使用pz中一组三个点的比例，来求解中间一个点在Z框架坐标系下的坐标
        for i in range(3):
            pz = np.array(pz0[start_idx: start_idx + 3])
            # # 质心是二维向量，追加一维变为三维向量
            # pz = [np.append(i, 0) for i in pz]
            start_idx = start_idx + 2

            # 求解对角线交点在frame中的位置（z框架物理坐标）
            p_zframe.append(self.solve_z(pz, origin_vector[i], direction_vector[i]))

        print("{0:<30s}:\n".format("p_zframe"), p_zframe)

        #  以三个点在Z框架坐标系下的坐标来建立坐标系，以第三个点为建立坐标系的原点
        #  计算建立M坐标系到Z框架坐标系的旋转矩阵
        rotation_matrix_Z = utils.get_rotation_matrix_by_3points(p_zframe, 1)
        print("{0:<30s}:\n".format("M2Zframe rotation matrix"), rotation_matrix_Z)

        # 将旋转矩阵扩展为转换矩阵T
        T_matrix_Z = np.identity(4)
        T_matrix_Z[:3, :3] = rotation_matrix_Z
        T_matrix_Z[:3, 3] = p_zframe[2].T

        print("{0:<30s}:\n".format("M2Zframe transformation matrix"), T_matrix_Z)

        # 计算三个参考点建立的坐标系与RAS坐标系之间的转换关系
        # Compute RAS坐标系下三个点的坐标 并用空列表p_ras保存
        p_ras = []
        point1 = np.append(centroid_points[1], slice_idx)
        # # # # # re_point1 = point1[[2,0,1]]
        # # # # # re_point1[-1] = 480 - re_point1[-1]
        point2 = np.append(centroid_points[3], slice_idx)
        # # # # # re_point2 = point2[[2,0,1]]
        # # # # # re_point2[-1] = 480 - re_point2[-1]
        point3 = np.append(centroid_points[5], slice_idx)
        # re_point3 = point3[[2,0,1]]
        # re_point3[-1] = 480 - re_point3[-1]

        # point1 = np.array([256.,87.87729395,195.2273])
        # point2 = np.array([303.21551288,37., 195.2273])
        # point3 = np.array([350.,87.8772939, 195.2273])
        # point1 = np.array([257,123.60420716,194])
        # point2 = np.array([305.49108947 , 74.1098443, 194])
        # point3 = np.array([351.03978143, 124.49349387, 194])
        # point1 = np.array([257.00984545,114.18356522,203])
        # point3 = np.array([351.11129679, 113.60950655, 203])
        # point2 = np.array([307.97937407,  61., 203.])


        f_point1 = utils.ijk_to_ras(point1, self.origin, self.direction, self.space)
        f_point2 = utils.ijk_to_ras(point2, self.origin, self.direction, self.space)
        f_point3 = utils.ijk_to_ras(point3, self.origin, self.direction, self.space)
        # point1 = np.array( [-26.92480443,  92.65492786,  20.0628    ])
        # point2 = np.array( [-50.6270982,  104.27606806,  20.0628    ])
        # point3 = np.array( [-71.61129001,  91.58618618,  20.0628    ])
        # point1 = np.array( [-5.1, 5.3, 87.7])
        # point2 = np.array( [-4.6,  -18.7, 107.5])
        # point3 = np.array( [-4.6,  -39.7,  85.5])


        # point1 = np.array( [-26.92480443,  92.65492786, -29.9372    ]) [-26.92480443  92.65492786  20.0628    ]


        p_ras.append(f_point1)
        p_ras.append(f_point2)
        p_ras.append(f_point3)
        print("{0:<30s}:\n".format("p_ras"), p_ras)


        #  以三个点在RAS坐标系下的坐标来建立坐标系，以第三个点为建立坐标系的原点
        #  计算建立M坐标系到RAS坐标系的旋转矩阵
        rotation_matrix_RAS = utils.get_rotation_matrix_by_3points(p_ras, 1)
        print("{0:<30s}:\n".format("M2RAS rotation matrix"), rotation_matrix_RAS)

        # 将旋转矩阵扩展为转换矩阵T
        T_matrix_RAS = np.identity(4)
        T_matrix_RAS[:3, :3] = rotation_matrix_RAS
        T_matrix_RAS[:3, 3] = p_ras[2].T
        print("{0:<30s}:\n".format("M2Zframe transformation matrix"), T_matrix_RAS)


        # 计算Z框架坐标系相对于RAS坐标系的转移矩阵 T_Zframe2RAS = T_matrix_RAS * inverse(T_matrix_Z)
        # Zframe坐标系 到 RAS坐标系的变换矩阵
        T_Zframe2RAS = np.matmul(T_matrix_RAS, np.linalg.inv(T_matrix_Z))

        return T_Zframe2RAS, p_zframe, p_ras

    def solve_z(self, pz, origin_vector, direction_vector):
        # 方向向量归一化
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        D12 = np.linalg.norm(pz[0] - pz[1])
        D23 = np.linalg.norm(pz[1] - pz[2])

        Ld = float(self.zframe_configration['Ld']) * np.sqrt(float(2.0))
        Lc = Ld * D23 / (D12 + D23)

        pf = origin_vector + direction_vector * Lc

        return pf
