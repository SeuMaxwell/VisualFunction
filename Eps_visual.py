# 该脚本用于可视化介电常数分布 (增加形态学滤波版)
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import median_filter, binary_opening, binary_closing

# ================= 学术风格配置 =================
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
# ===============================================

MICRO_SCALE = 1e6

# ================= 配置参数 (已修改) =================
# 1. 定义文件所在的基础目录
BASE_DIR = Path(r"E:\Postgraduate\Second\FDTD\VisualFunction\Project1_visual\Eps")
# 2. 定义要处理的文件编号范围 (5 到 10)
FILE_INDICES = range(5, 11)

# --- 其他参数保持不变 ---
THRESHOLD = 0.5
NO_BINARIZE = False
CLEAN_ISLANDS = True
FILTER_SIZE = 6
CMAP = "Greys"
TITLE = "Material Distribution"
NO_AUTO_TRANSPOSE = False
DPI = 400


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


def plot_params(params, x, y, cmap, dpi, out_path):
    extent = [x.min() * MICRO_SCALE, x.max() * MICRO_SCALE,
              y.min() * MICRO_SCALE, y.max() * MICRO_SCALE]

    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]

    base_width = 8
    fig_height = base_width * (y_range / x_range) * 0.9
    if fig_height < 3: fig_height = 3

    fig, ax = plt.subplots(figsize=(base_width, fig_height))

    im = ax.imshow(
        params,
        cmap=cmap,
        origin='lower',
        extent=extent,
        aspect='equal',
        interpolation='nearest',
        vmin=0, vmax=1
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    # cbar.set_label('Material State', fontsize=12)
    cbar.set_ticks([0, 1])

    ax.set_xlabel(u'X (\u03bcm)', fontsize=12, fontweight='bold')
    ax.set_ylabel(u'Y (\u03bcm)', fontsize=12, fontweight='bold')
    ax.grid(False)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'图像已保存: {out_path}')
    plt.close(fig)


def main():
    """
    主函数，循环处理指定范围内的所有文件。
    """
    for i in FILE_INDICES:
        # 动态生成输入和输出文件路径
        file_path = BASE_DIR / f"{i}parameters.npz"
        out_path = BASE_DIR / f"{i}eps_distribution.png"

        print(f"\n{'='*20} Processing file: {file_path.name} {'='*20}")

        try:
            data = load_npz(file_path)
            x, y, params = data['x'], data['y'], data['params']

            if not NO_BINARIZE:
                params = binarize(params, THRESHOLD)
                print(f'二值化完成: 阈值 {THRESHOLD}')

                if CLEAN_ISLANDS:
                    params = refine_structure(params, size=FILTER_SIZE)

            params = process_orientation(params, x, y, NO_AUTO_TRANSPOSE)
            plot_params(params, x, y, CMAP, DPI, out_path)

        except FileNotFoundError as e:
            print(f"错误: {e}。跳过此文件。")
            continue # 如果文件不存在，打印错误并继续处理下一个
        except Exception as e:
            print(f"处理文件 {file_path.name} 时发生未知错误: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*20} 所有文件处理完成! {'='*20}")


if __name__ == '__main__':
    main()
