# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import cmocean

# # 创建自定义颜色映射
# colors = [(0, 'blue'), (5, 'white'), (45, 'red')]  # 蓝色到白色再到红色

# # 归一化颜色映射点到 [0, 1] 范围
# min_val, max_val = 0, 45
# normalized_colors = [(float(i - min_val) / (max_val - min_val), color) for i, color in colors]

# cmap = LinearSegmentedColormap.from_list('custom_cmap', normalized_colors)

cmap = cmocean.cm.thermal # RdYlBu

# 加载 .npz 文件
# data = np.load('/home/bingxing2/ailab/scxlab0052/yysong/improved-diffusion-main/sample_test_log/317_2/samples_10x180x360x1.npz')
data = np.load('/home/bingxing2/ailab/scxlab0052/yysong/improved-diffusion-main/guided_result/sample_nzp/test_split/0327/grad_4.0_guided_0.075_loss_l2_sigma_False/185002_sample8x180x360x1.npz')

# 访问图像数组
arr = data['arr_0']

# 查看数组中的最大值和最小值
print(np.max(arr), np.min(arr))

# save_dir = "../sample_test_log/pictures/317_2_cmocean"
save_dir = '/home/bingxing2/ailab/scxlab0052/yysong/improved-diffusion-main/guided_result/sample_picture/tmp_test/grad_4.0/18002/'
os.makedirs(save_dir, exist_ok=True)  

# 可视化前几个图像
num_images_to_show = len(arr) 
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
    save_path = os.path.join(save_dir, f"sample_{i}.png")  # 指定保存路径
    plt.savefig(save_path)
    plt.close()  # 关闭当前图形，防止重叠