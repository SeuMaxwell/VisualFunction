# -*- coding: utf-8 -*-
"""
Radial_Evisual.py - 不同PCM厚度下YZ面电场分布对比可视化

本模块用于绘制不同相变材料(PCM)厚度下的YZ平面电场强度分布对比图。
图像格式符合学术期刊双栏排版要求(宽度约178mm/7英寸)。

功能:
- 从Lumerical FDTD导出的.mat文件加载电场数据
- 计算电场强度并进行归一化处理
- 使用 plot_style 模块生成符合期刊要求的双栏对比图

作者: [Ruan Zhiwei]
日期: 2024
"""

import matplotlib

matplotlib.use('Agg')

import argparse
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# 引入新的绘图封装函数
from plot_style import (
    create_double_column_figure,
    DOUBLE_COLUMN_WIDTH_INCH,
    E_CMAP,
    PRINT_DPI
)

# 本地定义 Colorbar 宽度比例 (解决 plot_style 中缺失定义的问题)
CBAR_WIDTH_RATIO = 0.05


def load_and_process_data(filename, target_freq_idx=0):
    """
    读取 .mat 文件并计算电场强度 (|E|^2)。
    """
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}，跳过。")
        return None
    try:
        with h5py.File(filename, 'r') as f:
            y = np.array(f['E_data']['y']).flatten()
            z = np.array(f['E_data']['z']).flatten()
            E_dataset = f['E_data']['E']

            num_freqs = E_dataset.shape[0]
            current_freq_idx = target_freq_idx
            if num_freqs <= 1:
                current_freq_idx = 0
            elif current_freq_idx >= num_freqs:
                print(f"警告: {filename} 中目标频率索引 {target_freq_idx} 超出范围，"
                      f"将使用中间频率索引 {num_freqs // 2}。")
                current_freq_idx = num_freqs // 2

            E_raw = E_dataset[current_freq_idx]
            # 计算复数电场
            E_complex = E_raw['real'] + 1j * E_raw['imag']
            # 计算光强 Intensity = |E|^2
            Intensity_flat = np.sum(np.abs(E_complex) ** 2, axis=0)

            Ny, Nz = len(y), len(z)
            if len(Intensity_flat) != Ny * Nz:
                print(f"错误: {filename} 数据点数不匹配，跳过。")
                return None

            # 重塑为 2D 网格 (F-order 对应 MATLAB 导出格式)
            Intensity_2D = Intensity_flat.reshape((Ny, Nz), order='F')

            return {
                'Intensity': Intensity_2D,
                'y_um': y * 1e6,
                'z_um': z * 1e6,
                'max_val': np.max(Intensity_2D)
            }
    except Exception as e:
        print(f"读取或处理 {filename} 失败: {e}")
        return None


def build_axes_grid(num_plots):
    """
    使用 create_double_column_figure 创建画布，并配置 GridSpec 布局。
    """
    # 目标高度为 2.5 英寸
    target_height = 2.5
    # 计算高宽比: Height / Width (7.0 inch)
    aspect_ratio = target_height / DOUBLE_COLUMN_WIDTH_INCH

    # 使用 plot_style 提供的工厂函数创建 Figure
    # 这会自动应用期刊样式 (字体、字号等) 并设置正确的双栏宽度
    fig = create_double_column_figure(aspect_ratio=aspect_ratio)

    # 配置子图布局: N 个数据图 + 1 个 Colorbar
    width_ratios = [1] * num_plots + [CBAR_WIDTH_RATIO]
    gs = gridspec.GridSpec(1, num_plots + 1, width_ratios=width_ratios)

    axes = [fig.add_subplot(gs[i]) for i in range(num_plots)]
    cbar_ax = fig.add_subplot(gs[-1])

    return fig, axes, cbar_ax


def plot_comparison(data_buffer, output_filename, suptitle= None):
    """
    绘制对比图主逻辑。
    """
    if not data_buffer:
        print("数据缓冲区为空，无法绘图。")
        return

    # 1. 计算全局最大值用于归一化
    global_max_intensity = max(
        (d['max_val'] for d in data_buffer if d['max_val'] > 0),
        default=0
    )
    if global_max_intensity <= 0:
        print("警告: 全局最大强度为0，无法进行归一化绘图。")
        return

    # 2. 创建画布和坐标轴 (不再需要手动调用 apply_journal_style)
    fig, axes, cbar_ax = build_axes_grid(len(data_buffer))

    # 3. 循环绘制子图
    pcm = None
    for idx, data in enumerate(data_buffer):
        ax = axes[idx]
        normalized_intensity = data['Intensity'].T / global_max_intensity

        pcm = ax.pcolormesh(
            data['y_um'],
            data['z_um'],
            normalized_intensity,
            shading='gouraud',
            cmap=E_CMAP,
            vmin=0,
            vmax=1
        )

        ax.set_title(f"{data['thickness']} nm")
        ax.set_xlabel("Y (μm)")

        # 仅第一个子图显示 Y 轴标签
        if idx == 0:
            ax.set_ylabel("Z (μm)")
        else:
            ax.set_yticklabels([])

        ax.set_aspect('auto')
        ax.set_xlim(data['y_um'].min(), data['y_um'].max())
        ax.set_ylim(data['z_um'].min(), data['z_um'].max())

    # 4. 添加 Colorbar
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_ticks(np.linspace(0, 1, 5))

    # # 5. 调整布局细节
    # plt.suptitle(suptitle, fontsize=9, y=0.98)

    # 手动微调边距以适应 GridSpec
    plt.subplots_adjust(
        left=0.07,
        right=0.95,
        top=0.95,               #标题0.85 无标题0.95
        bottom=0.20,
        wspace=0.08
    )

    # 6. 保存
    plt.savefig(output_filename, bbox_inches="tight", dpi=PRINT_DPI)
    plt.close(fig)
    print(f"对比图已成功保存为: {output_filename}")


# ============================================================================
#                           配置与主程序
# ============================================================================

DEFAULT_CONFIG = {
    'data_dir': 'E:\Postgraduate\Second\FDTD\VisualFunction\Project1_visual\Eyz',
    'thickness_list': [60, 80, 100, 120, 140],
    'target_freq_idx': 2,
    'states': [
        {
            'name': 'Amorphous',
            'file_template': 'A_thickness_{}nm.mat',
            'output_filename': 'Amorphous_Thickness_Comparison.png',
            # 'suptitle': 'Axial Optical Field Distribution (Amorphous State)'
        },
        {
            'name': 'Crystalline',
            'file_template': 'C_thickness_{}nm.mat',
            'output_filename': 'Crystalline_Thickness_Comparison.png',
            # 'suptitle': 'Axial Optical Field Distribution (Crystalline State)'
        }
    ]
}


def prepare_state_data(state, data_dir, thickness_list, target_freq_idx):
    buffer = []
    for thickness in thickness_list:
        filename = os.path.join(
            data_dir,
            state['file_template'].format(thickness)
        )
        processed_data = load_and_process_data(filename, target_freq_idx)
        if processed_data:
            processed_data['thickness'] = thickness
            buffer.append(processed_data)
            print(f"✓ 已加载: {os.path.basename(filename)}")
    return buffer


def run_visualization(config=None):
    if config is None:
        config = DEFAULT_CONFIG

    data_dir = config['data_dir']
    thickness_list = config['thickness_list']
    target_freq_idx = config['target_freq_idx']

    for state in config['states']:
        print(f"\n{'=' * 50}")
        print(f"开始处理 {state['name']} 状态数据")


        data_buffer = prepare_state_data(
            state,
            data_dir,
            thickness_list,
            target_freq_idx
        )

        if data_buffer:
            output_path = os.path.join(data_dir, state['output_filename'])
            # plot_comparison(data_buffer, output_path, state['suptitle'])
            plot_comparison(data_buffer, output_path)
        else:
            print(f"✗ 未找到 {state['name']} 状态的数据文件")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=DEFAULT_CONFIG['data_dir'],
        help='数据文件目录路径'
    )
    parser.add_argument(
        '--freq-idx', '-f',
        type=int,
        default=DEFAULT_CONFIG['target_freq_idx'],
        help='目标频率索引 (默认: 2)'
    )
    parser.add_argument(
        '--thicknesses', '-t',
        type=int,
        nargs='+',
        default=DEFAULT_CONFIG['thickness_list'],
        help='PCM厚度列表，单位nm (默认: 60 80 100 120 140)'
    )

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        'data_dir': args.data_dir,
        'target_freq_idx': args.freq_idx,
        'thickness_list': args.thicknesses
    })

    print(f"数据目录: {config['data_dir']}")
    print(f"厚度列表: {config['thickness_list']} nm")
    print(f"频率索引: {config['target_freq_idx']}")

    run_visualization(config)


if __name__ == '__main__':
    main()
