import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

# 主程序
folder_path = 'z-frame-20240815\Data\\new501\ScalarVolume_21'  # 替换为你的DICOM文件夹路径
images = load_dicom_images(folder_path)
create_viewer(images)
