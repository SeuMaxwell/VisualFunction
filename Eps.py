# -*- coding: utf-8 -*-
# 该脚本用于可视化介电常数分布 (适配 plot_style 学术风格版 - 双栏多子图模式)
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
# 移除 make_axes_locatable，改用 GridSpec 布局
# from mpl_toolkits.axes_grid1 import make_axes_locatable

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
OUTPUT_FILENAME = "Combined_Eps_Distribution.png"  # 汇总输出文件名


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
    # print(f"正在进行形态学滤波 (Size={size})...") # 减少日志输出
    cleaned = median_filter(params, size=size)
    return cleaned


def process_orientation(params, x, y, no_transpose):
    if no_transpose:
        return params
    if params.shape == (len(x), len(y)):
        return params.T
    elif params.shape == (len(y), len(x)):
        return params
    raise ValueError(f'维度不匹配')


def plot_combined_grid(data_buffer, out_path):
    """
    将收集到的所有数据绘制在一张双栏宽度的图中。
    """
    if not data_buffer:
        print("没有数据可绘图。")
        return

    n_plots = len(data_buffer)
    n_cols = 3  # 固定为3列，适合双栏布局
    n_rows = int(np.ceil(n_plots / n_cols))

    # 1. 计算画布尺寸
    # 假设所有子图的物理尺寸比例相似，取第一个数据的比例作为参考
    ref_extent = data_buffer[0]['extent']
    # 单个子图的数据高宽比 (Height / Width)
    data_aspect = (ref_extent[3] - ref_extent[2]) / (ref_extent[1] - ref_extent[0])

    # 计算整个 Figure 的高宽比
    # Figure 宽度固定为 7.0 英寸 (Double Column)
    # 估算：Figure 高度 ≈ (宽度 / 列数) * 数据高宽比 * 行数
    # 乘以 1.2 是为了给标题和轴标签留出空间
    fig_aspect_ratio = (data_aspect * (n_rows / n_cols)) * 1.2

    # 创建双栏画布
    fig = plot_style.create_double_column_figure(aspect_ratio=fig_aspect_ratio)

    # 2. 配置 GridSpec 布局
    # 前 n_cols 列用于放图，最后一列用于放 Colorbar (宽度较小)
    width_ratios = [1] * n_cols + [0.05]
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=width_ratios, wspace=0.3, hspace=0.4)

    im = None  # 用于保存最后一次 imshow 对象以生成 colorbar

    # 3. 循环绘制子图
    for idx, item in enumerate(data_buffer):
        row = idx // n_cols
        col = idx % n_cols

        ax = fig.add_subplot(gs[row, col])

        im = ax.imshow(
            item['params'],
            cmap=CMAP,
            origin='lower',
            extent=item['extent'],
            aspect='equal',
            interpolation='nearest',
            vmin=0, vmax=1
        )

        # 设置标题 (使用文件索引)
        ax.set_title(f"Index: {item['index']}")

        # 设置轴标签
        ax.set_xlabel(u'X (\u03bcm)')
        ax.set_ylabel(u'Y (\u03bcm)')

        # 优化：为了整洁，可以只在最左侧和最下侧显示标签（可选）
        # if col > 0: ax.set_ylabel('')
        # if row < n_rows - 1: ax.set_xlabel('')

        ax.grid(False)

    # 4. 添加共享 Colorbar (放在最右侧一列，跨越所有行)
    cax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=7)
    # cbar.set_label('Permittivity (a.u.)') # 可选

    # 5. 保存
    plt.savefig(out_path, dpi=plot_style.PRINT_DPI, bbox_inches='tight')
    print(f"汇总图像已保存: {out_path}")
    plt.close(fig)


def main():
    """
    主函数：收集数据并统一绘图。
    """
    plot_style.apply_journal_style()

    data_buffer = []  # 用于存储处理后的数据

    print(f"开始处理文件范围: {FILE_INDICES}")

    for i in FILE_INDICES:
        file_path = BASE_DIR / f"{i}parameters.npz"

        try:
            # 1. 加载数据
            data = load_npz(file_path)
            x, y, params = data['x'], data['y'], data['params']

            # 2. 数据处理
            if not NO_BINARIZE:
                params = binarize(params, THRESHOLD)
                if CLEAN_ISLANDS:
                    params = refine_structure(params, size=FILTER_SIZE)

            params = process_orientation(params, x, y, NO_AUTO_TRANSPOSE)

            # 3. 计算物理范围
            extent = [x.min() * MICRO_SCALE, x.max() * MICRO_SCALE,
                      y.min() * MICRO_SCALE, y.max() * MICRO_SCALE]

            # 4. 存入缓冲区
            data_buffer.append({
                'index': i,
                'params': params,
                'extent': extent
            })
            print(f"已加载数据: {file_path.name}")

        except FileNotFoundError:
            print(f"跳过缺失文件: {file_path.name}")
            continue
        except Exception as e:
            print(f"处理 {file_path.name} 出错: {e}")
            continue

    # 5. 统一绘图
    if data_buffer:
        out_path = BASE_DIR / OUTPUT_FILENAME
        plot_combined_grid(data_buffer, out_path)
    else:
        print("未收集到有效数据，无法绘图。")

    print(f"\n{'=' * 20} 处理完成 {'=' * 20}")


if __name__ == '__main__':
    main()
