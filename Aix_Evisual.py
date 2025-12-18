import numpy as np
# cmap='turbo',cividis，inferno，viridis / magma
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # 必须在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import traceback
import os

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
        self._setup_plot_style()

    @staticmethod
    def _setup_plot_style():
        """配置 Matplotlib 的学术风格。"""
        mpl.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9,
            'figure.figsize': (3.35, 2.36),
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'axes.linewidth': 1.2
        })

    def process_field_data(self, target_lambda, target_z, sigma_val):
        """
        加载、处理和准备用于绘图的电场数据。

        :param target_lambda: 目标波长 (单位: 米)。
        :param target_z: 目标 Z 切片位置 (单位: 米)。
        :param sigma_val: 高斯滤波的标准差。
        :return: 一个包含绘图所需数据的字典。
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

            E_total_complex = np.sqrt(Ex**2 + Ey**2 + Ez**2)
            E_total_3D = E_total_complex.reshape((Nx, Ny, Nz), order='F')

            a_slice = np.abs(E_total_3D[:, :, idx_z])
            a_smooth = gaussian_filter(a_slice, sigma=sigma_val)
            print(f"已应用高斯滤波 (Sigma={sigma_val})")

            plot_data = a_smooth.T
            extent = [x.min() * 1e6, x.max() * 1e6, y.min() * 1e6, y.max() * 1e6]
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

        :param data: process_field_data 方法返回的字典。
        :param output_filename: 输出图像的文件名。
        """
        fig, ax = plt.subplots(figsize=(8, 8 * data['aspect_ratio']))

        im = ax.imshow(data['plot_data'],
                       extent=data['extent'],
                       cmap='inferno',
                       origin='lower',
                       aspect='equal',
                       vmin=data['vmin'], vmax=data['vmax'],
                       interpolation='bilinear')

        ax.set_xlabel('X (μm)', fontsize=10, fontname='Times New Roman')
        ax.set_ylabel('Y (μm)', fontsize=10, fontname='Times New Roman')
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        plt.savefig(output_filename)
        plt.close(fig)
        print(f"图像已保存: {output_filename}")


def main():
    """主执行函数，循环处理多个文件。"""
    # ================= 参数配置 =================
    GROUP_NAME = 'E'
    SIGMA_VAL = 1.5
    TARGET_LAMBDA = 1.55e-6
    TARGET_Z = 0.0

    # 循环处理 E5.mat 到 E10.mat
    for i in range(0, 5):
        file_path = f'E{i}.mat'
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

