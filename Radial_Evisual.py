# -*- coding: utf-8 -*-
"""
Radial_Evisual.py - 不同PCM厚度下YZ面电场分布对比可视化

本模块用于绘制不同相变材料(PCM)厚度下的YZ平面电场强度分布对比图。
图像格式符合学术期刊双栏排版要求(宽度约178mm/7英寸)。

功能:
- 从Lumerical FDTD导出的.mat文件加载电场数据
- 计算电场强度并进行归一化处理
- 生成符合期刊要求的多子图对比图

作者: [Your Name]
日期: 2024
"""

import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适用于服务器环境

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# 导入本地绑图风格配置
from plot_style import (
    apply_journal_style,
    DOUBLE_COLUMN_WIDTH_INCH,
    PRINT_DPI,
    EFIELD_CMAP,
    COLORBAR_WIDTH_RATIO
)

def load_and_process_data(filename, target_freq_idx=0):
    """
    加载并处理单个 .mat 文件中的电场数据。

    :param filename: .mat 文件路径。
    :param target_freq_idx: 目标频率的索引。
    :return: 包含处理后数据的字典，如果失败则返回 None。
    """
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}，跳过。")
        return None
    try:
        with h5py.File(filename, 'r') as f:
            y = np.array(f['E_data']['y']).flatten()
            z = np.array(f['E_data']['z']).flatten()
            E_dataset = f['E_data']['E']

            # 确保频率索引有效
            num_freqs = E_dataset.shape[0]
            current_freq_idx = target_freq_idx
            if num_freqs <= 1:
                current_freq_idx = 0
            elif current_freq_idx >= num_freqs:
                print(f"警告: {filename} 中目标频率索引 {target_freq_idx} 超出范围，"
                      f"将使用中间频率索引 {num_freqs // 2}。")
                current_freq_idx = num_freqs // 2

            E_raw = E_dataset[current_freq_idx]
            E_complex = E_raw['real'] + 1j * E_raw['imag']
            Intensity_flat = np.sum(np.abs(E_complex) ** 2, axis=0)

            Ny, Nz = len(y), len(z)
            if len(Intensity_flat) != Ny * Nz:
                print(f"错误: {filename} 数据点数不匹配，跳过。")
                return None

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


def plot_comparison(data_buffer, output_filename, suptitle):
    """
    生成符合学术期刊要求的多子图电场分布对比图。
    
    Parameters
    ----------
    data_buffer : list[dict]
        包含多个数据集的列表，每个元素包含:
        - 'Intensity': 2D电场强度数组
        - 'y_um', 'z_um': 坐标数组 (微米)
        - 'max_val': 最大强度值
        - 'thickness': PCM厚度 (nm)
    output_filename : str
        输出图像文件路径
    suptitle : str
        图像总标题
    
    Notes
    -----
    - 图像宽度固定为7英寸(~178mm)，符合双栏期刊排版要求
    - 使用inferno色图，归一化显示电场强度
    - 所有子图共享同一colorbar
    """
    if not data_buffer:
        print("数据缓冲区为空，无法绘图。")
        return

    num_plots = len(data_buffer)
    global_max_intensity = max([d['max_val'] for d in data_buffer if d['max_val'] > 0])
    if global_max_intensity == 0:
        print("警告: 全局最大强度为0，无法进行归一化绘图。")
        return

    # 应用学术期刊风格
    apply_journal_style()

    # 计算图像尺寸：双栏宽度，适当高度
    fig_width = DOUBLE_COLUMN_WIDTH_INCH  # 7.0 英寸 ≈ 178 mm
    fig_height = 2.5  # 适合5个并排子图的高度

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=PRINT_DPI)

    # GridSpec布局：子图 + colorbar
    width_ratios = [1] * num_plots + [COLORBAR_WIDTH_RATIO]
    gs = gridspec.GridSpec(1, num_plots + 1, width_ratios=width_ratios)

    pcm = None  # 保存最后一个pcolormesh对象用于colorbar
    
    for i, data in enumerate(data_buffer):
        ax = fig.add_subplot(gs[i])
        normalized_intensity = data['Intensity'].T / global_max_intensity

        pcm = ax.pcolormesh(
            data['y_um'],
            data['z_um'],
            normalized_intensity,
            shading='gouraud',
            cmap=EFIELD_CMAP,
            vmin=0,
            vmax=1
        )

        # 子图标题：显示PCM厚度
        ax.set_title(f"{data['thickness']} nm")
        ax.set_xlabel("Y (μm)")
        
        # 只有第一个子图显示Y轴标签
        if i == 0:
            ax.set_ylabel("Z (μm)")
        else:
            ax.set_yticklabels([])

        ax.set_aspect('auto')
        ax.set_xlim(data['y_um'].min(), data['y_um'].max())
        ax.set_ylim(data['z_um'].min(), data['z_um'].max())

    # 添加共享Colorbar
    cbar_ax = fig.add_subplot(gs[-1])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label('Normalized Intensity')

    # 总标题和布局调整
    plt.suptitle(suptitle, fontsize=9, y=0.98)
    plt.subplots_adjust(
        left=0.07,
        right=0.95,
        top=0.85,
        bottom=0.20,
        wspace=0.08
    )

    plt.savefig(output_filename, bbox_inches="tight", dpi=PRINT_DPI)
    plt.close(fig)
    print(f"对比图已成功保存为: {output_filename}")


# ============================================================================
#                           配置参数
# ============================================================================

# 默认配置 - 可根据实际环境修改
DEFAULT_CONFIG = {
    # 数据文件所在目录 (修改为你的实际路径)
    'data_dir': './data/Eyz',
    
    # PCM厚度列表 (单位: nm)
    'thickness_list': [60, 80, 100, 120, 140],
    
    # 目标频率索引 (对应FDTD仿真的频率点)
    'target_freq_idx': 2,
    
    # 不同PCM状态的配置
    'states': [
        {
            'name': 'Amorphous',
            'file_template': 'A_thickness_{}nm.mat',
            'output_filename': 'Amorphous_Thickness_Comparison.png',
            'suptitle': 'Axial Optical Field Distribution (Amorphous State)'
        },
        {
            'name': 'Crystalline',
            'file_template': 'C_thickness_{}nm.mat',
            'output_filename': 'Crystalline_Thickness_Comparison.png',
            'suptitle': 'Axial Optical Field Distribution (Crystalline State)'
        }
    ]
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
    thickness_list = config['thickness_list']
    target_freq_idx = config['target_freq_idx']
    
    for state in config['states']:
        print(f"\n{'='*50}")
        print(f"开始处理 {state['name']} 状态数据")
        print('='*50)
        
        data_buffer = []
        
        for t in thickness_list:
            filename = os.path.join(data_dir, state['file_template'].format(t))
            processed_data = load_and_process_data(filename, target_freq_idx)
            
            if processed_data:
                processed_data['thickness'] = t
                data_buffer.append(processed_data)
                print(f"✓ 已加载: {os.path.basename(filename)}")
        
        # 绘图
        if data_buffer:
            output_path = os.path.join(data_dir, state['output_filename'])
            plot_comparison(data_buffer, output_path, state['suptitle'])
        else:
            print(f"✗ 未找到 {state['name']} 状态的数据文件")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='YZ面电场分布对比可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python Radial_Evisual.py --data-dir ./my_data
  python Radial_Evisual.py --data-dir /path/to/data --freq-idx 1
        '''
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
    
    # 构建配置
    config = DEFAULT_CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['target_freq_idx'] = args.freq_idx
    config['thickness_list'] = args.thicknesses
    
    print("="*60)
    print("YZ面电场分布对比可视化 - 学术期刊双栏格式")
    print("="*60)
    print(f"数据目录: {config['data_dir']}")
    print(f"厚度列表: {config['thickness_list']} nm")
    print(f"频率索引: {config['target_freq_idx']}")
    
    run_visualization(config)
