# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os



def sample_vision(data_path, picture_path, top_k=None):
    data = np.load(data_path)
    # 访问图像数组
    arr = data['arr_0']
    print(np.max(arr), np.min(arr))
    os.makedirs(picture_path, exist_ok=True)  
    # 可视化前几个图像
    num_images_to_show = len(arr) if top_k is None else top_k
    for i in range(num_images_to_show):
        im = plt.imshow(arr[i].squeeze(), cmap='coolwarm', vmin=-5, vmax=40, origin='lower')  # 使用自定义颜色映射  cmap='RdYlBu'反了  bwr
        plt.title(f"Sample {i}")
        plt.axis('off')  # 隐藏坐标轴

        # 自定义颜色条的刻度标签
        cbar = plt.colorbar(im, orientation='vertical')  # 添加颜色条
        cbar.set_label('Temperature (°C)')  # 设置颜色条标签
        cbar_ticks = np.linspace(-5, 40, num=10)  # 生成颜色条的刻度
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])  # 在真实值的基础上减去 5
        
        # 保存到指定路径
        save_path = os.path.join(picture_path, f"sample_{i}.png")  # 指定保存路径
        plt.savefig(save_path)
        plt.close()  # 关闭当前图形，防止重叠