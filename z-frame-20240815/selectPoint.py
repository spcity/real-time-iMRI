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

def play_tviewer(images,index):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    img_display = ax.imshow(images[index], cmap='gray')
    plt.show()
# 主程序
folder_path = 'z-frame-20240815\Data\\tof3d_tra_2801'  # 替换为你的DICOM文件夹路径
images = load_dicom_images(folder_path)
# print(images)
# 先进行维度交换，使得新的顺序为 (640, 550, 524)
transposed_images = np.transpose(images, (2, 0, 1))

# 将结果 reshape 为 (640, 550, 524)，其中每个二维数组的形状为 (550, 524)
reshaped_images = transposed_images.reshape(524,550, 640)
create_viewer(reshaped_images)
# play_tviewer(reshaped_images,262)