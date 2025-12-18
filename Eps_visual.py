# -*- coding: utf-8 -*-
# 该脚本用于可视化介电常数分布 (适配 plot_style 学术风格版)
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable  # 用于精确控制colorbar

# 导入自定义样式模块
import plot_style

# ================= 配置参数 =================
MICRO_SCALE = 1e6

# 1. 定义文件所在的基础目录
BASE_DIR = Path(r"E:\Postgraduate\Second\FDTD\VisualFunction\Project1_visual\Eps")
# 2. 定义要处理的文件编号范围
FILE_INDICES = range(5, 11)

THRESHOLD = 0.5
NO_BINARIZE = False
CLEAN_ISLANDS = True
FILTER_SIZE = 6
CMAP = "Greys"
TITLE = "Material Distribution"
NO_AUTO_TRANSPOSE = False


def load_npz(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'文件不存在: {path}')
    data = np.load(path)
    if not all(k in data.files for k in ['x', 'y', 'params']):
        raise KeyError('缺少必要数组: x, y, params')
    return {k: data[k] for k in ['x', 'y', 'params']}


def binarize(params, threshold):
    return np.where(params > threshold, 1.0, 0.0)


def refine_structure(params, size=3):
    """
    使用中值滤波去除孤立像素，并平滑边界
    """
    print(f"正在进行形态学滤波 (Size={size})...")
    cleaned = median_filter(params, size=size)
    diff = np.sum(np.abs(params - cleaned))
    print(f"已修正像素点数量: {int(diff)}")
    return cleaned


def process_orientation(params, x, y, no_transpose):
    if no_transpose:
        return params
    if params.shape == (len(x), len(y)):
        return params.T
    elif params.shape == (len(y), len(x)):
        return params
    raise ValueError(f'维度不匹配')


def plot_params(params, x, y, cmap, out_path):
    """
    绘制并保存图像，使用 plot_style 规范。
    """
    # 计算物理范围
    extent = [x.min() * MICRO_SCALE, x.max() * MICRO_SCALE,
              y.min() * MICRO_SCALE, y.max() * MICRO_SCALE]

    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]

    # 计算高宽比，用于生成正确的画布尺寸
    aspect_ratio = y_range / x_range

    # === 修改点 1: 使用 plot_style 创建单栏画布 ===
    # 这会自动应用 8pt 字体、细线条等期刊风格
    fig = plot_style.create_single_column_figure(aspect_ratio=aspect_ratio)
    ax = fig.add_subplot(111)

    # 绘制图像
    im = ax.imshow(
        params,
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='equal',
        interpolation='nearest',
        vmin=0, vmax=1
    )

    # === 修改点 2: 移除硬编码字体设置，使用全局样式 ===
    ax.set_xlabel(u'X (\u03bcm)')  # 自动使用 8pt, normal weight
    ax.set_ylabel(u'Y (\u03bcm)')
    ax.grid(False)  # 确保无网格

    # 设置坐标轴范围
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # === 修改点 3: 规范化 Colorbar ===
    # 使用 make_axes_locatable 确保 colorbar 宽度适中且不挤压主图
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, 1])
    # 确保 colorbar 刻度字体大小与坐标轴一致 (7pt)
    cbar.ax.tick_params(labelsize=7)

    # 保存图片
    # 使用 plot_style 定义的打印级 DPI (300)
    plt.savefig(out_path, dpi=plot_style.PRINT_DPI, bbox_inches='tight')
    print(f'图像已保存: {out_path}')
    plt.close(fig)


def main():
    """
    主函数，循环处理指定范围内的所有文件。
    """
    # 确保应用样式 (虽然 create_single_column_figure 会自动调用，但显式调用是个好习惯)
    plot_style.apply_journal_style()

    for i in FILE_INDICES:
        # 动态生成输入和输出文件路径
        file_path = BASE_DIR / f"{i}parameters.npz"
        out_path = BASE_DIR / f"{i}eps_distribution.png"

        print(f"\n{'=' * 20} Processing file: {file_path.name} {'=' * 20}")

        try:
            data = load_npz(file_path)
            x, y, params = data['x'], data['y'], data['params']

            if not NO_BINARIZE:
                params = binarize(params, THRESHOLD)
                print(f'二值化完成: 阈值 {THRESHOLD}')

                if CLEAN_ISLANDS:
                    params = refine_structure(params, size=FILTER_SIZE)

            params = process_orientation(params, x, y, NO_AUTO_TRANSPOSE)

            # 调用绘图函数 (不再需要传递 DPI，内部使用 plot_style 配置)
            plot_params(params, x, y, CMAP, out_path)

        except FileNotFoundError as e:
            print(f"错误: {e}。跳过此文件。")
            continue
        except Exception as e:
            print(f"处理文件 {file_path.name} 时发生未知错误: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 20} 所有文件处理完成! {'=' * 20}")


if __name__ == '__main__':
    main()
