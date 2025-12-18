# -*- coding: utf-8 -*-
"""
main.py - 可视化主程序

本模块作为统一入口，调用各可视化模块生成FDTD仿真数据的可视化图像。

功能:
- 介电常数分布可视化 (Eps_visual)
- XY面电场分布可视化 (Aix_Evisual)
- YZ面电场对比可视化 (Radial_Evisual)

使用方法:
    python main.py --data-dir ./data --all
    python main.py --eps --xy
    python main.py --yz --thicknesses 60 80 100

作者: [Your Name]
日期: 2024
"""

import argparse
import os
import sys
from pathlib import Path

# 导入各可视化模块
try:
    from Eps_visual import run_visualization as run_eps_visualization
    from Eps_visual import DEFAULT_CONFIG as EPS_DEFAULT_CONFIG
except ImportError as e:
    print(f"警告: 无法导入 Eps_visual 模块: {e}")
    run_eps_visualization = None
    EPS_DEFAULT_CONFIG = {}

try:
    from Aix_Evisual import run_visualization as run_xy_visualization
    from Aix_Evisual import DEFAULT_CONFIG as XY_DEFAULT_CONFIG
except ImportError as e:
    print(f"警告: 无法导入 Aix_Evisual 模块: {e}")
    run_xy_visualization = None
    XY_DEFAULT_CONFIG = {}

try:
    from Radial_Evisual import run_visualization as run_yz_visualization
    from Radial_Evisual import DEFAULT_CONFIG as YZ_DEFAULT_CONFIG
except ImportError as e:
    print(f"警告: 无法导入 Radial_Evisual 模块: {e}")
    run_yz_visualization = None
    YZ_DEFAULT_CONFIG = {}


# ============================================================================
#                           默认配置
# ============================================================================

MASTER_CONFIG = {
    # 数据根目录
    'data_root': './data',
    
    # 各模块子目录
    'eps_subdir': 'Eps',
    'xy_subdir': '',          # XY面电场数据在根目录
    'yz_subdir': 'Eyz',
    
    # 通用参数
    'file_indices': list(range(5, 11)),  # 分光比索引
    'thickness_list': [60, 80, 100, 120, 140],  # PCM厚度列表 (nm)
}


# ============================================================================
#                           可视化执行函数
# ============================================================================

def visualize_eps_distribution(config):
    """
    执行介电常数分布可视化。
    
    Parameters
    ----------
    config : dict
        配置字典，包含 data_root, eps_subdir, file_indices 等
    """
    print("\n" + "="*70)
    print("【1】介电常数分布可视化 (单栏格式)")
    print("="*70)
    
    eps_config = EPS_DEFAULT_CONFIG.copy()
    eps_config['data_dir'] = os.path.join(config['data_root'], config['eps_subdir'])
    eps_config['file_indices'] = config.get('file_indices', EPS_DEFAULT_CONFIG['file_indices'])
    
    run_eps_visualization(eps_config)


def visualize_xy_field(config):
    """
    执行XY面电场分布可视化。
    
    Parameters
    ----------
    config : dict
        配置字典
    """
    print("\n" + "="*70)
    print("【2】XY面电场分布可视化 (单栏格式)")
    print("="*70)
    
    xy_config = XY_DEFAULT_CONFIG.copy()
    # 处理空子目录的情况
    xy_subdir = config.get('xy_subdir', '')
    if xy_subdir:
        xy_config['data_dir'] = os.path.join(config['data_root'], xy_subdir)
    else:
        xy_config['data_dir'] = config['data_root']
    xy_config['file_indices'] = config.get('file_indices', XY_DEFAULT_CONFIG['file_indices'])
    
    run_xy_visualization(xy_config)


def visualize_yz_comparison(config):
    """
    执行YZ面电场对比可视化。
    
    Parameters
    ----------
    config : dict
        配置字典
    """
    print("\n" + "="*70)
    print("【3】YZ面电场分布对比可视化 (双栏格式)")
    print("="*70)
    
    yz_config = YZ_DEFAULT_CONFIG.copy()
    yz_config['data_dir'] = os.path.join(config['data_root'], config['yz_subdir'])
    yz_config['thickness_list'] = config.get('thickness_list', YZ_DEFAULT_CONFIG['thickness_list'])
    
    run_yz_visualization(yz_config)


def run_all_visualizations(config):
    """
    执行所有可视化任务。
    
    Parameters
    ----------
    config : dict
        主配置字典
    """
    print("\n" + "#"*70)
    print("#" + " "*20 + "FDTD仿真数据可视化工具" + " "*20 + "#")
    print("#"*70)
    print(f"\n数据根目录: {config['data_root']}")
    print(f"分光比索引: {config['file_indices']}")
    print(f"PCM厚度列表: {config['thickness_list']} nm")
    
    # 1. 介电常数分布
    visualize_eps_distribution(config)
    
    # 2. XY面电场分布
    visualize_xy_field(config)
    
    # 3. YZ面电场对比
    visualize_yz_comparison(config)
    
    print("\n" + "#"*70)
    print("#" + " "*22 + "所有可视化任务完成!" + " "*22 + "#")
    print("#"*70)


# ============================================================================
#                           命令行入口
# ============================================================================

def main():
    """主函数：解析命令行参数并执行可视化。"""
    
    parser = argparse.ArgumentParser(
        description='FDTD仿真数据可视化工具 - 统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
可视化类型说明:
  --eps     介电常数分布 (单栏, 3.5 in)
  --xy      XY面电场分布 (单栏, 3.5 in)
  --yz      YZ面电场对比 (双栏, 7.0 in)
  --all     执行所有可视化任务

示例:
  # 执行所有可视化
  python main.py --data-dir ./my_data --all
  
  # 只生成介电常数分布图
  python main.py --data-dir ./my_data --eps
  
  # 生成XY和YZ电场图
  python main.py --data-dir ./my_data --xy --yz
  
  # 指定分光比索引和厚度
  python main.py --data-dir ./data --all -i 5 6 7 -t 60 80 100
        '''
    )
    
    # 数据路径参数
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=MASTER_CONFIG['data_root'],
        help='数据根目录路径 (默认: ./data)'
    )
    
    # 可视化类型选择
    parser.add_argument(
        '--eps',
        action='store_true',
        help='生成介电常数分布图'
    )
    parser.add_argument(
        '--xy',
        action='store_true',
        help='生成XY面电场分布图'
    )
    parser.add_argument(
        '--yz',
        action='store_true',
        help='生成YZ面电场对比图'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='执行所有可视化任务'
    )
    
    # 其他参数
    parser.add_argument(
        '--indices', '-i',
        type=int,
        nargs='+',
        default=MASTER_CONFIG['file_indices'],
        help='分光比文件索引列表 (默认: 5 6 7 8 9 10)'
    )
    parser.add_argument(
        '--thicknesses', '-t',
        type=int,
        nargs='+',
        default=MASTER_CONFIG['thickness_list'],
        help='PCM厚度列表，单位nm (默认: 60 80 100 120 140)'
    )
    
    args = parser.parse_args()
    
    # 构建配置
    config = MASTER_CONFIG.copy()
    config['data_root'] = args.data_dir
    config['file_indices'] = args.indices
    config['thickness_list'] = args.thicknesses
    
    # 确定要执行的任务
    run_eps = args.eps or args.all
    run_xy = args.xy or args.all
    run_yz = args.yz or args.all
    
    # 如果没有指定任何任务，显示帮助
    if not (run_eps or run_xy or run_yz):
        parser.print_help()
        print("\n错误: 请至少指定一个可视化任务 (--eps, --xy, --yz, 或 --all)")
        sys.exit(1)
    
    # 打印配置信息
    print("\n" + "#"*70)
    print("#" + " "*20 + "FDTD仿真数据可视化工具" + " "*20 + "#")
    print("#"*70)
    print(f"\n数据根目录: {config['data_root']}")
    print(f"分光比索引: {config['file_indices']}")
    print(f"PCM厚度列表: {config['thickness_list']} nm")
    print(f"\n待执行任务:")
    if run_eps:
        print("  ✓ 介电常数分布可视化")
    if run_xy:
        print("  ✓ XY面电场分布可视化")
    if run_yz:
        print("  ✓ YZ面电场对比可视化")
    
    # 执行可视化任务
    if run_eps:
        visualize_eps_distribution(config)
    
    if run_xy:
        visualize_xy_field(config)
    
    if run_yz:
        visualize_yz_comparison(config)
    
    print("\n" + "#"*70)
    print("#" + " "*22 + "所有任务执行完成!" + " "*22 + "#")
    print("#"*70)


if __name__ == '__main__':
    main()
