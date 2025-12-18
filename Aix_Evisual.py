import numpy as np
# cmap=''turbo,cividis，inferno，viridis / magma
# -*- coding: utf-8 -*-
"""
Aix_Evisual.py - 单个电场分布文件的可视化 (重构版)

功能:
- 读取 Lumerical .mat 文件
- 应用高斯滤波处理数据
- 使用 plot_style 模块生成符合期刊单栏排版要求的图片
"""

import numpy as np
import matplotlib as mpl

mpl.use('Agg')  # 必须在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import traceback
import os

# ============================================================================
#                           导入样式配置模块
# ============================================================================
from plot_style import create_single_column_figure, PRINT_DPI, E_CMAP


class FieldVisualizer:
    """
    一个用于处理和可视化 Lumerical 导出的 .mat 文件中电场数据的类。
    """

    def __init__(self, file_path, group_name='E'):
        """
        初始化 FieldVisualizer。

        :param file_path: .mat 文件的路径 (HDF5 格式)。
        :param group_name: 包含电场数据的 HDF5 组名。
        """
        self.file_path = file_path
        self.group_name = group_name
        # 移除内部样式设置，改用 plot_style 模块在绘图时动态应用

    def process_field_data(self, target_lambda, target_z, sigma_val):
        """
        加载、处理和准备用于绘图的电场数据。
        """
        print(f"正在处理文件: {self.file_path}...")
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
            a_smooth = gaussian_filter(a_slice, sigma=sigma_val)
            print(f"已应用高斯滤波 (Sigma={sigma_val})")

            plot_data = a_smooth.T
            extent = [x.min() * 1e6, x.max() * 1e6, y.min() * 1e6, y.max() * 1e6]

            # 计算数据的物理高宽比 (Height / Width)
            aspect_ratio = (y.max() - y.min()) / (x.max() - x.min())

            return {
                "plot_data": plot_data,
                "extent": extent,
                "vmin": np.min(a_slice),
                "vmax": np.max(a_slice),
                "aspect_ratio": aspect_ratio
            }

    def plot_and_save(self, data, output_filename):
        """
        根据处理后的数据进行绘图并保存。
        使用 plot_style.create_single_column_figure 确保单栏宽度。
        """
        # 1. 创建符合单栏宽度的 Figure (宽度固定为 3.5 英寸)
        # aspect_ratio 参数确保画布高度适应数据形状
        fig = create_single_column_figure(aspect_ratio=data['aspect_ratio'])

        # 添加子图
        ax = fig.add_subplot(111)

        # 调节 colorbar 范围：例如固定 vmin=0, vmax=数据最大值的 80%
        custom_vmin = 0  # 固定最小值
        custom_vmax = 1  # 调整最大值为原值的 80%

        # 2. 绘制图像
        im = ax.imshow(data['plot_data'],
                       extent=data['extent'],
                       cmap=E_CMAP,
                       origin='lower',
                       aspect='equal',
                       vmin=custom_vmin, vmax=custom_vmax,
                       interpolation='bilinear')

        # 3. 设置标签 (移除硬编码的字体和字号，使用 plot_style 的全局配置)
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')

        # 设置刻度密度
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        # 4. 添加 Colorbar
        # 使用 make_axes_locatable 可以在不改变主图宽高比太多的情况下附加 colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks(np.linspace(custom_vmin, custom_vmax, 5))
        # cbar.set_label('Electric Field Intensity') # 可选

        # 5. 保存图片
        # 使用 plot_style 定义的打印级 DPI
        plt.savefig(output_filename, dpi=PRINT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"图像已保存: {output_filename}")


def main():
    """主执行函数，循环处理多个文件。"""
    # ================= 参数配置 =================
    GROUP_NAME = 'E'
    SIGMA_VAL = 1.5
    TARGET_LAMBDA = 1.55e-6
    TARGET_Z = 0.0
    DATA_DIR = 'E:\Postgraduate\Second\FDTD\VisualFunction\pro1_visual'
    # 循环处理 E0.mat 到 E5.mat
    for i in range(0, 6):
        filename = f'E{i}.mat'
        file_path = os.path.join(DATA_DIR, filename)
        output_filename = f'E{i}_plot.png'

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 文件 '{file_path}' 不存在，已跳过。")
            continue

        print(f"\n--- 开始处理文件: {file_path} ---")
        try:
            # 1. 创建可视化工具实例
            visualizer = FieldVisualizer(file_path=file_path, group_name=GROUP_NAME)

            # 2. 处理数据
            plot_params = visualizer.process_field_data(
                target_lambda=TARGET_LAMBDA,
                target_z=TARGET_Z,
                sigma_val=SIGMA_VAL
            )

            # 3. 绘图并保存
            if plot_params:
                visualizer.plot_and_save(plot_params, output_filename)

        except Exception as e:
            print(f"处理文件 '{file_path}' 时出错:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
