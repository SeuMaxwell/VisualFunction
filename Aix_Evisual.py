# -*- coding: utf-8 -*-
"""
Aix_Evisual.py - XY面电场分布可视化

本模块用于可视化Lumerical FDTD导出的.mat文件中的XY平面电场分布。
图像格式符合学术期刊单栏排版要求(宽度约89mm/3.5英寸)。

功能:
- 从HDF5格式的.mat文件加载电场数据
- 计算总电场强度并应用高斯滤波平滑
- 生成符合期刊要求的单栏单幅图

作者: [Your Name]
日期: 2024
"""

import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适用于服务器环境

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import traceback
import os
import argparse

# 导入本地绘图风格配置
from plot_style import (
    apply_journal_style,
    get_figure_size,
    SINGLE_COLUMN_WIDTH_INCH,
    PRINT_DPI,
    EFIELD_CMAP
)


# ============================================================================
#                           物理常量
# ============================================================================
SPEED_OF_LIGHT = 299792458  # 光速 (m/s)
MICRO_SCALE = 1e6           # 米转微米


# ============================================================================
#                           数据处理类
# ============================================================================

class FieldVisualizer:
    """
    用于处理和可视化 Lumerical 导出的 .mat 文件中电场数据的类。
    
    Attributes
    ----------
    file_path : str
        .mat 文件的路径 (HDF5 格式)
    group_name : str
        包含电场数据的 HDF5 组名
    """

    def __init__(self, file_path, group_name='E'):
        """
        初始化 FieldVisualizer。

        Parameters
        ----------
        file_path : str
            .mat 文件的路径 (HDF5 格式)
        group_name : str
            包含电场数据的 HDF5 组名，默认为 'E'
        """
        self.file_path = file_path
        self.group_name = group_name

    def process_field_data(self, target_lambda, target_z, sigma_val):
        """
        加载、处理和准备用于绘图的电场数据。

        Parameters
        ----------
        target_lambda : float
            目标波长 (单位: 米)
        target_z : float
            目标 Z 切片位置 (单位: 米)
        sigma_val : float
            高斯滤波的标准差

        Returns
        -------
        dict or None
            包含绘图所需数据的字典，失败时返回 None
        """
        print(f"正在处理文件: {self.file_path}...")
        
        try:
            with h5py.File(self.file_path, 'r') as f:
                grp = f[self.group_name]

                x = np.array(grp['x']).flatten()
                y = np.array(grp['y']).flatten()
                z = np.array(grp['z']).flatten()
                f_data = np.array(grp['f']).flatten()
                Nx, Ny, Nz = len(x), len(y), len(z)

                # 计算波长并找到目标索引
                lambda_data = SPEED_OF_LIGHT / f_data
                idx_f = np.argmin(np.abs(lambda_data - target_lambda))
                idx_z = np.argmin(np.abs(z - target_z))

                # 提取电场分量
                E_raw = grp['E'][idx_f]
                Ex = E_raw[0]['real'] + 1j * E_raw[0]['imag']
                Ey = E_raw[1]['real'] + 1j * E_raw[1]['imag']
                Ez = E_raw[2]['real'] + 1j * E_raw[2]['imag']

                # 计算总电场强度
                E_total_complex = np.sqrt(Ex**2 + Ey**2 + Ez**2)
                E_total_3D = E_total_complex.reshape((Nx, Ny, Nz), order='F')

                # 提取 XY 切片并应用高斯滤波
                a_slice = np.abs(E_total_3D[:, :, idx_z])
                a_smooth = gaussian_filter(a_slice, sigma=sigma_val)
                print(f"已应用高斯滤波 (Sigma={sigma_val})")

                # 准备绘图数据
                plot_data = a_smooth.T
                extent = [
                    x.min() * MICRO_SCALE, x.max() * MICRO_SCALE,
                    y.min() * MICRO_SCALE, y.max() * MICRO_SCALE
                ]
                aspect_ratio = (y.max() - y.min()) / (x.max() - x.min())

                return {
                    "plot_data": plot_data,
                    "extent": extent,
                    "vmin": np.min(a_slice),
                    "vmax": np.max(a_slice),
                    "aspect_ratio": aspect_ratio
                }
        except Exception as e:
            print(f"处理文件时出错: {e}")
            return None

    def plot_and_save(self, data, output_filename):
        """
        根据处理后的数据进行绘图并保存。

        Parameters
        ----------
        data : dict
            process_field_data 方法返回的字典
        output_filename : str
            输出图像的文件名
        """
        # 应用期刊风格
        apply_journal_style()

        # 计算单栏图像尺寸
        fig_size = get_figure_size('single', aspect_ratio=data['aspect_ratio'])
        
        fig, ax = plt.subplots(figsize=fig_size, dpi=PRINT_DPI)

        im = ax.imshow(
            data['plot_data'],
            extent=data['extent'],
            cmap=EFIELD_CMAP,
            origin='lower',
            aspect='equal',
            vmin=data['vmin'],
            vmax=data['vmax'],
            interpolation='bilinear'
        )

        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        # 添加 Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('|E|')

        plt.savefig(output_filename, bbox_inches='tight', dpi=PRINT_DPI)
        plt.close(fig)
        print(f"图像已保存: {output_filename}")


# ============================================================================
#                           配置参数
# ============================================================================

DEFAULT_CONFIG = {
    # 数据文件所在目录 (修改为实际路径)
    'data_dir': './data',
    
    # HDF5组名
    'group_name': 'E',
    
    # 高斯滤波标准差
    'sigma': 1.5,
    
    # 目标波长 (米)
    'target_lambda': 1.55e-6,
    
    # 目标Z切片位置 (米)
    'target_z': 0.0,
    
    # 文件索引范围
    'file_indices': list(range(0, 5)),
    
    # 文件名模板
    'file_template': 'E{}.mat',
    
    # 输出文件名模板
    'output_template': 'E{}_XY_field.png'
}


def run_visualization(config=None):
    """
    运行可视化流程。
    
    Parameters
    ----------
    config : dict, optional
        配置字典，如果为None则使用DEFAULT_CONFIG
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    data_dir = config['data_dir']
    
    for i in config['file_indices']:
        file_path = os.path.join(data_dir, config['file_template'].format(i))
        output_filename = os.path.join(data_dir, config['output_template'].format(i))
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 文件 '{file_path}' 不存在，已跳过。")
            continue

        print(f"\n{'='*50}")
        print(f"开始处理文件: {os.path.basename(file_path)}")
        print('='*50)
        
        try:
            # 创建可视化工具实例
            visualizer = FieldVisualizer(
                file_path=file_path,
                group_name=config['group_name']
            )

            # 处理数据
            plot_params = visualizer.process_field_data(
                target_lambda=config['target_lambda'],
                target_z=config['target_z'],
                sigma_val=config['sigma']
            )

            # 绘图并保存
            if plot_params:
                visualizer.plot_and_save(plot_params, output_filename)

        except Exception as e:
            print(f"处理文件 '{file_path}' 时出错:")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='XY面电场分布可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python Aix_Evisual.py --data-dir ./my_data
  python Aix_Evisual.py --data-dir /path/to/data --sigma 2.0
        '''
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=DEFAULT_CONFIG['data_dir'],
        help='数据文件目录路径'
    )
    parser.add_argument(
        '--sigma', '-s',
        type=float,
        default=DEFAULT_CONFIG['sigma'],
        help='高斯滤波标准差 (默认: 1.5)'
    )
    parser.add_argument(
        '--wavelength', '-w',
        type=float,
        default=DEFAULT_CONFIG['target_lambda'],
        help='目标波长，单位米 (默认: 1.55e-6)'
    )
    parser.add_argument(
        '--indices', '-i',
        type=int,
        nargs='+',
        default=DEFAULT_CONFIG['file_indices'],
        help='文件索引列表 (默认: 0 1 2 3 4)'
    )
    
    args = parser.parse_args()
    
    # 构建配置
    config = DEFAULT_CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['sigma'] = args.sigma
    config['target_lambda'] = args.wavelength
    config['file_indices'] = args.indices
    
    print("="*60)
    print("XY面电场分布可视化 - 学术期刊单栏格式")
    print("="*60)
    print(f"数据目录: {config['data_dir']}")
    print(f"文件索引: {config['file_indices']}")
    print(f"高斯滤波Sigma: {config['sigma']}")
    print(f"目标波长: {config['target_lambda']*1e9:.2f} nm")
    
    run_visualization(config)
