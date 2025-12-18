# -*- coding: utf-8 -*-
"""
Aix_Evisual.py - 电场分布汇总可视化 (双栏多子图版)

功能:
- 批量读取 Lumerical .mat 文件
- 应用高斯滤波处理数据
- 将所有数据绘制在一张符合期刊双栏排版要求的组合图中
"""

import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
import traceback
import os

# ============================================================================
#                           导入样式配置模块
# ============================================================================
from plot_style import create_double_column_figure, PRINT_DPI, E_CMAP


class FieldVisualizer:
    """
    用于处理 Lumerical .mat 文件中电场数据的类。
    """

    def __init__(self, file_path, group_name='E'):
        self.file_path = file_path
        self.group_name = group_name

    def process_field_data(self, target_lambda, target_z, sigma_val):
        """
        加载并处理数据，返回绘图所需的字典。
        """
        # print(f"正在读取: {os.path.basename(self.file_path)}...")
        with h5py.File(self.file_path, 'r') as f:
            grp = f[self.group_name]

            x = np.array(grp['x']).flatten()
            y = np.array(grp['y']).flatten()
            z = np.array(grp['z']).flatten()
            f_data = np.array(grp['f']).flatten()
            Nx, Ny, Nz = len(x), len(y), len(z)

            lambda_data = 299792458 / f_data
            idx_f = np.argmin(np.abs(lambda_data - target_lambda))
            idx_z = np.argmin(np.abs(z - target_z))

            E_raw = grp['E'][idx_f]
            Ex = E_raw[0]['real'] + 1j * E_raw[0]['imag']
            Ey = E_raw[1]['real'] + 1j * E_raw[1]['imag']
            Ez = E_raw[2]['real'] + 1j * E_raw[2]['imag']

            E_total_complex = np.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)
            E_total_3D = E_total_complex.reshape((Nx, Ny, Nz), order='F')

            a_slice = np.abs(E_total_3D[:, :, idx_z])

            # 应用高斯滤波
            a_smooth = gaussian_filter(a_slice, sigma=sigma_val)

            # 转置以适配 imshow (origin='lower')
            plot_data = a_smooth.T

            extent = [x.min() * 1e6, x.max() * 1e6, y.min() * 1e6, y.max() * 1e6]

            # 计算数据的物理高宽比 (Height / Width)
            aspect_ratio = (y.max() - y.min()) / (x.max() - x.min())

            return {
                "plot_data": plot_data,
                "extent": extent,
                "max_val": np.max(a_smooth),  # 记录最大值用于全局归一化
                "aspect_ratio": aspect_ratio
            }


def plot_combined_grid(data_buffer, output_filename):
    """
    将收集到的数据绘制为双栏宽度的多子图。
    """
    if not data_buffer:
        print("无数据可绘图。")
        return

    n_plots = len(data_buffer)
    n_cols = 3  # 固定为3列
    n_rows = int(np.ceil(n_plots / n_cols))

    # 1. 计算全局最大值，确保所有子图颜色标度一致 (便于对比)
    global_max = max(item['max_val'] for item in data_buffer)
    if global_max == 0: global_max = 1
    print(f"全局最大电场强度 (用于归一化): {global_max:.4e}")

    # 2. 计算画布尺寸
    # 假设所有子图比例相似，取第一个作为参考
    ref_aspect = data_buffer[0]['aspect_ratio']
    # 估算 Figure 的高宽比: (单图高宽比 * 行数 / 列数) * 留白系数
    fig_aspect_ratio = (ref_aspect * (n_rows / n_cols)) * 1.2

    # 创建双栏画布 (7.0 英寸宽)
    fig = create_double_column_figure(aspect_ratio=fig_aspect_ratio)

    # 3. 配置 GridSpec (N行, 3列图 + 1列Colorbar)
    width_ratios = [1] * n_cols + [0.05]  # 最后一列给 colorbar
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=width_ratios, wspace=0.3, hspace=0.4)

    im = None

    # 4. 循环绘制
    for idx, item in enumerate(data_buffer):
        row = idx // n_cols
        col = idx % n_cols

        ax = fig.add_subplot(gs[row, col])

        im = ax.imshow(
            item['plot_data'],
            extent=item['extent'],
            cmap=E_CMAP,
            origin='lower',
            aspect='equal',
            vmin=0,
            vmax=global_max,  # 使用全局最大值
            interpolation='bilinear'  # 场分布建议保持 bilinear，若需完全一致可改为 nearest
        )

        # 设置标题 (使用文件名或索引)
        # 移除 fontsize=8，使用 plot_style 默认值
        ax.set_title(item['name'])

        # 设置轴标签 (使用 unicode 编码，与 Eps.py 保持一致)
        ax.set_xlabel(u'X (\u03bcm)')
        ax.set_ylabel(u'Y (\u03bcm)')

        # 移除手动刻度限制，与 Eps.py 保持一致
        # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ax.grid(False)

    # 5. 添加共享 Colorbar (跨越所有行，位于最右侧)
    cax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(im, cax=cax)

    # 场强是连续值，通常不需要像 Eps.py 那样手动设置 [0, 1] 刻度
    # 但保留字体大小设置
    cbar.ax.tick_params(labelsize=7)
    # cbar.set_label('|E| Field', rotation=270, labelpad=10)

    # 6. 保存
    plt.savefig(output_filename, dpi=PRINT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"汇总图像已保存: {output_filename}")


def main():
    # ================= 参数配置 =================
    GROUP_NAME = 'E'
    SIGMA_VAL = 1.5
    TARGET_LAMBDA = 1.55e-6
    TARGET_Z = 0.0
    DATA_DIR = r'E:\Postgraduate\Second\FDTD\VisualFunction\pro1_visual'
    OUTPUT_FILENAME = os.path.join(DATA_DIR, 'Combined_E_Field_Distribution.png')

    # 定义要处理的文件列表
    FILE_INDICES = range(0, 6)  # E0.mat 到 E5.mat

    data_buffer = []

    print(f"{'=' * 10} 开始批量处理 {'=' * 10}")

    for i in FILE_INDICES:
        filename = f'E{i}.mat'
        file_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(file_path):
            print(f"跳过缺失文件: {filename}")
            continue

        try:
            visualizer = FieldVisualizer(file_path=file_path, group_name=GROUP_NAME)

            # 处理数据
            data = visualizer.process_field_data(
                target_lambda=TARGET_LAMBDA,
                target_z=TARGET_Z,
                sigma_val=SIGMA_VAL
            )

            # 添加标识符
            data['name'] = f"E{i}"
            data_buffer.append(data)
            print(f"已加载: {filename}")

        except Exception as e:
            print(f"处理 {filename} 失败: {e}")
            traceback.print_exc()

    # 统一绘图
    if data_buffer:
        plot_combined_grid(data_buffer, OUTPUT_FILENAME)
    else:
        print("未收集到有效数据。")


if __name__ == "__main__":
    main()

