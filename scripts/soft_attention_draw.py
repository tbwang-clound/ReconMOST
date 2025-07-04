import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
def create_gaussian_kernel(size=5, sigma=1.0):
    """
    创建一个二维高斯核。
    """
    coords = torch.arange(size, dtype=torch.float32)
    coords -= (size - 1) / 2.0

    g = coords ** 2
    g = g.unsqueeze(0) + g.unsqueeze(1)  # 计算每个点到中心的距离的平方
    g = torch.exp(-g / (2 * sigma ** 2))  # 计算高斯函数值

    return g


def main():
    # --- 1. 创建一个15x15的稀疏矩阵 ---
    matrix_size = 15
    sparsity = 0.075

    num_nonzero = int(matrix_size * matrix_size * sparsity)

    sparse_matrix = np.zeros((matrix_size, matrix_size))

    # 首先，生成所有可能的坐标点 (0,0), (0,1), ..., (14,14)
    all_indices = np.arange(matrix_size * matrix_size)

    # 然后，从所有可能的坐标点中随机选择 num_nonzero 个
    chosen_indices = np.random.choice(all_indices, num_nonzero, replace=False)

    # 将一维索引转换为二维坐标
    rows, cols = np.unravel_index(chosen_indices, (matrix_size, matrix_size))

    # 在这些随机选择的位置上填充随机数字
    sparse_matrix[rows, cols] = np.random.rand(num_nonzero) * 0.5 + 0.5

    print("原始稀疏矩阵 (部分数据):")
    print(np.round(sparse_matrix[:5, :5], 2))
    print("-" * 30)

    # --- 2. 创建高斯卷积核 ---
    kernel_size = 5

    gaussian_kernel_unnormalized = create_gaussian_kernel(size=kernel_size, sigma=1.0)
    gaussian_kernel_normalized = gaussian_kernel_unnormalized / torch.sum(gaussian_kernel_unnormalized)

    print("未归一化的高斯核:")
    print(np.round(gaussian_kernel_unnormalized.numpy(), 3))
    print("\n归一化的高斯核:")
    print(np.round(gaussian_kernel_normalized.numpy(), 3))
    print("-" * 30)

    # --- 3. 执行卷积操作 ---
    input_tensor = torch.from_numpy(sparse_matrix).float().unsqueeze(0).unsqueeze(0)

    kernel_unnormalized = gaussian_kernel_unnormalized.unsqueeze(0).unsqueeze(0)
    kernel_normalized = gaussian_kernel_normalized.unsqueeze(0).unsqueeze(0)

    output_unnormalized = F.conv2d(input_tensor, kernel_unnormalized, stride=1, padding='same').squeeze().numpy()
    output_normalized = F.conv2d(input_tensor, kernel_normalized, stride=1, padding='same').squeeze().numpy()

    # --- 4. 可视化对比 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    vmin = 0
    vmax = np.max(sparse_matrix)

    im1 = axes[0].imshow(sparse_matrix, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('原始稀疏矩阵 (Original Sparse Matrix)', fontsize=14)
    axes[0].grid(False)

    im2 = axes[1].imshow(output_unnormalized, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('未归一化卷积后 (Unnormalized Conv)', fontsize=14)
    axes[1].grid(False)

    im3 = axes[2].imshow(output_normalized, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title('归一化卷积后 (Normalized Conv)', fontsize=14)
    axes[2].grid(False)

    fig.colorbar(im3, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

    plt.suptitle('高斯卷积对稀疏矩阵的影响对比', fontsize=18)
    plt.show()


if __name__ == '__main__':
    main()