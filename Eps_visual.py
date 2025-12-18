# -*- coding: utf-8 -*-
"""
Eps_visual.py - 介电常数分布可视化

本模块用于可视化相变材料(PCM)层的介电常数/材料参数分布。
图像格式符合学术期刊单栏排版要求(宽度约89mm/3.5英寸)。

功能:
- 从.npz文件加载材料参数数据
- 二值化处理和形态学滤波
- 生成符合期刊要求的单栏单幅图

作者: [Your Name]
日期: 2024
"""

import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适用于服务器环境

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import argparse

# 导入本地绘图风格配置
from plot_style import (
    apply_journal_style,
    get_figure_size,
    SINGLE_COLUMN_WIDTH_INCH,
    PRINT_DPI
)


# ============================================================================
#                           常量定义
# ============================================================================
MICRO_SCALE = 1e6  # 米转微米

# 介电常数分布推荐色图 (二值材料分布)
EPS_CMAP = 'Greys'


# ============================================================================
#                           数据处理函数
# ============================================================================

def load_npz(path):
    """
    加载.npz格式的材料参数文件。
    
    Parameters
    ----------
    path : str or Path
        .npz文件路径
    
    Returns
    -------
    dict
        包含 'x', 'y', 'params' 数组的字典
    
    Raises
    ------
    FileNotFoundError
        文件不存在时抛出
    KeyError
        缺少必要数组时抛出
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'文件不存在: {path}')
    
    data = np.load(path)
    required_keys = ['x', 'y', 'params']
    
    if not all(k in data.files for k in required_keys):
        raise KeyError(f'缺少必要数组: {required_keys}')
    
    return {k: data[k] for k in required_keys}


def binarize(params, threshold=0.5):
    """
    将参数数组二值化。
    
    Parameters
    ----------
    params : ndarray
        原始参数数组
    threshold : float
        二值化阈值，默认0.5
    
    Returns
    -------
    ndarray
        二值化后的数组 (0.0 或 1.0)
    """
    return np.where(params > threshold, 1.0, 0.0)


def refine_structure(params, size=3):
    """
    使用中值滤波去除孤立像素并平滑边界。
    
    Parameters
    ----------
    params : ndarray
        二值化后的参数数组
    size : int
        中值滤波窗口大小
    
    Returns
    -------
    ndarray
        滤波后的数组
    """
    print(f"正在进行形态学滤波 (Size={size})...")
    cleaned = median_filter(params, size=size)
    diff = np.sum(np.abs(params - cleaned))
    print(f"已修正像素点数量: {int(diff)}")
    return cleaned


def process_orientation(params, x, y, no_transpose=False):
    """
    处理数组方向以匹配坐标系。
    
    Parameters
    ----------
    params : ndarray
        参数数组
    x, y : ndarray
        坐标数组
    no_transpose : bool
        是否禁用自动转置
    
    Returns
    -------
    ndarray
        处理后的数组
    """
    if no_transpose:
        return params
    
    if params.shape == (len(x), len(y)):
        return params.T
    elif params.shape == (len(y), len(x)):
        return params
    
    raise ValueError(f'维度不匹配: params.shape={params.shape}, x={len(x)}, y={len(y)}')


# ============================================================================
#                           绘图函数
# ============================================================================

def plot_eps_distribution(params, x, y, output_path, cmap=EPS_CMAP):
    """
    绘制介电常数/材料分布图。
    
    Parameters
    ----------
    params : ndarray
        处理后的参数数组
    x, y : ndarray
        坐标数组 (单位: 米)
    output_path : str or Path
        输出图像路径
    cmap : str
        色图名称
    """
    # 应用期刊风格
    apply_journal_style()
    
    # 计算范围
    extent = [
        x.min() * MICRO_SCALE, x.max() * MICRO_SCALE,
        y.min() * MICRO_SCALE, y.max() * MICRO_SCALE
    ]
    
    # 计算宽高比
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    aspect_ratio = y_range / x_range if x_range > 0 else 1.0
    
    # 获取单栏图像尺寸
    fig_size = get_figure_size('single', aspect_ratio=aspect_ratio)
    
    fig, ax = plt.subplots(figsize=fig_size, dpi=PRINT_DPI)
    
    im = ax.imshow(
        params,
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='equal',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )
    
    # 添加 Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['a-PCM', 'c-PCM'])
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.grid(False)
    
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    plt.savefig(output_path, bbox_inches='tight', dpi=PRINT_DPI)
    print(f'图像已保存: {output_path}')
    plt.close(fig)


# ============================================================================
#                           配置参数
# ============================================================================

DEFAULT_CONFIG = {
    # 数据文件所在目录 (修改为实际路径)
    'data_dir': './data/Eps',
    
    # 文件索引范围
    'file_indices': list(range(5, 11)),
    
    # 文件名模板
    'file_template': '{}parameters.npz',
    
    # 输出文件名模板
    'output_template': '{}eps_distribution.png',
    
    # 二值化阈值
    'threshold': 0.5,
    
    # 是否进行二值化
    'binarize': True,
    
    # 是否进行形态学滤波
    'clean_islands': True,
    
    # 中值滤波窗口大小
    'filter_size': 6,
    
    # 色图
    'cmap': 'Greys',
    
    # 是否禁用自动转置
    'no_transpose': False
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
    
    data_dir = Path(config['data_dir'])
    
    for i in config['file_indices']:
        file_path = data_dir / config['file_template'].format(i)
        output_path = data_dir / config['output_template'].format(i)
        
        print(f"\n{'='*50}")
        print(f"开始处理文件: {file_path.name}")
        print('='*50)
        
        try:
            # 加载数据
            data = load_npz(file_path)
            x, y, params = data['x'], data['y'], data['params']
            
            # 二值化处理
            if config['binarize']:
                params = binarize(params, config['threshold'])
                print(f'二值化完成: 阈值 {config["threshold"]}')
                
                # 形态学滤波
                if config['clean_islands']:
                    params = refine_structure(params, size=config['filter_size'])
            
            # 处理数组方向
            params = process_orientation(params, x, y, config['no_transpose'])
            
            # 绘图
            plot_eps_distribution(params, x, y, output_path, config['cmap'])
            
        except FileNotFoundError as e:
            print(f"错误: {e}。跳过此文件。")
            continue
        except Exception as e:
            print(f"处理文件 {file_path.name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("所有文件处理完成!")
    print('='*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='介电常数分布可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python Eps_visual.py --data-dir ./my_data
  python Eps_visual.py --data-dir /path/to/data --threshold 0.6
        '''
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=DEFAULT_CONFIG['data_dir'],
        help='数据文件目录路径'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=DEFAULT_CONFIG['threshold'],
        help='二值化阈值 (默认: 0.5)'
    )
    parser.add_argument(
        '--filter-size', '-f',
        type=int,
        default=DEFAULT_CONFIG['filter_size'],
        help='中值滤波窗口大小 (默认: 6)'
    )
    parser.add_argument(
        '--indices', '-i',
        type=int,
        nargs='+',
        default=DEFAULT_CONFIG['file_indices'],
        help='文件索引列表 (默认: 5 6 7 8 9 10)'
    )
    parser.add_argument(
        '--no-binarize',
        action='store_true',
        help='禁用二值化处理'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='禁用形态学滤波'
    )
    
    args = parser.parse_args()
    
    # 构建配置
    config = DEFAULT_CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['threshold'] = args.threshold
    config['filter_size'] = args.filter_size
    config['file_indices'] = args.indices
    config['binarize'] = not args.no_binarize
    config['clean_islands'] = not args.no_filter
    
    print("="*60)
    print("介电常数分布可视化 - 学术期刊单栏格式")
    print("="*60)
    print(f"数据目录: {config['data_dir']}")
    print(f"文件索引: {config['file_indices']}")
    print(f"二值化: {'是' if config['binarize'] else '否'}")
    if config['binarize']:
        print(f"  阈值: {config['threshold']}")
        print(f"  滤波: {'是' if config['clean_islands'] else '否'}")
        if config['clean_islands']:
            print(f"  窗口大小: {config['filter_size']}")
    
    run_visualization(config)
