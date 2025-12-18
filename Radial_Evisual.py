#绘制不同厚度pcm下电场分布的对比图  径向E场分布
import matplotlib
# 建议使用 'Agg' 后端以避免在无图形界面的服务器上出错
matplotlib.use('Agg')

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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


# def plot_comparison(data_buffer, output_filename, suptitle):
#     """
#     根据提供的数据缓冲区生成并保存对比图。
#
#     :param data_buffer: 包含多个数据集的列表，每个元素是一个字典。
#     :param output_filename: 输出图片的文件名。
#     :param suptitle: 图像的总标题。
#     """
#     if not data_buffer:
#         print("数据缓冲区为空，无法绘图。")
#         return
#
#     num_plots = len(data_buffer)
#     # 找到所有数据中的全局最大强度值，用于归一化
#     global_max_intensity = max([d['max_val'] for d in data_buffer if d['max_val'] > 0])
#     if global_max_intensity == 0:
#         print("警告: 全局最大强度为0，无法进行归一化绘图。")
#         return
#     #非归一化
#     # global_max_intensity = max([d['max_val'] for d in data_buffer])
#
#     # 创建画布和子图网格
#     fig = plt.figure(figsize=(3 * num_plots + 1, 5), dpi=300)
#     gs = gridspec.GridSpec(1, num_plots + 1, width_ratios=[1] * num_plots + [0.05])
#
#     for i, data in enumerate(data_buffer):
#         ax = plt.subplot(gs[i])
#         # 将强度数据除以全局最大值进行归一化
#         normalized_intensity = data['Intensity'].T / global_max_intensity
#
#         pcm = ax.pcolormesh(
#             data['y_um'],
#             data['z_um'],
#             normalized_intensity,  # 使用归一化后的数据
#             shading='gouraud',
#             cmap='inferno',
#             vmin=0,
#             vmax=1  # 归一化后的数据范围是 0 到 1
#         )
#         # # 交换 x, y 输入，并转置 Intensity 矩阵
#         # pcm = ax.pcolormesh(
#         #     data['y_um'],
#         #     data['z_um'],
#         #     data['Intensity'].T, # 转置强度矩阵以匹配新的坐标轴
#         #     shading='gouraud',
#         #     cmap='inferno',
#         #     vmin=0,
#         #     vmax=global_max_intensity
#         # )
#
#         ax.set_title(f"{data['thickness']} nm", fontsize=12, fontweight='bold')
#         # 交换坐标轴标签
#         ax.set_xlabel(u'Y (\u03bcm)', fontsize=10)
#         ax.tick_params(axis='both', direction='in', labelsize=8)
#
#         if i == 0:
#             ax.set_ylabel(u'Z (\u03bcm)', fontsize=10)
#         else:
#             ax.set_yticklabels([]) # 隐藏非首个子图的Y轴刻度标签
#
#         ax.set_aspect('auto')
#         ax.set_xlim(data['y_um'].min(), data['y_um'].max())
#         ax.set_ylim(data['z_um'].min(), data['z_um'].max())
#
#     # 添加共享的 Colorbar
#     cbar_ax = plt.subplot(gs[-1])
#     cbar = fig.colorbar(pcm, cax=cbar_ax)
#     cbar.set_label('Electric Field Intensity', fontsize=10)
#     cbar.formatter.set_powerlimits((0, 0))
#
#     plt.suptitle(suptitle, fontsize=14, y=0.98)
#     # wspace 控制子图的水平间距，可以根据需要调整此值
#     plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.92, wspace=0.15)
#
#     plt.savefig(output_filename)
#     plt.close(fig) # 关闭图像，释放内存
#     print(f"对比图已成功保存为: {output_filename}")

def plot_comparison(data_buffer, output_filename, suptitle):
    if not data_buffer:
        print("数据缓冲区为空，无法绘图。")
        return

    num_plots = len(data_buffer)
    global_max_intensity = max([d['max_val'] for d in data_buffer if d['max_val'] > 0])
    if global_max_intensity == 0:
        print("警告: 全局最大强度为0，无法进行归一化绘图。")
        return

    # ===== 1. 全局风格（接近顶刊排版） =====
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        "font.size": 8,              # 默认字体 8 pt
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
    })

    # ===== 2. 图像物理尺寸：宽 178 mm（约 7 in），高 2.5 in =====
    fig_width_inch = 7.0     # ≈ 178 mm
    fig_height_inch = 2.5    # 可根据需要调到 3.0
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=300)

    # gridspec 宽度比例：num_plots 个子图 + 1 个窄 colorbar
    width_ratios = [1] * num_plots + [0.15]  # 最右侧 colorbar 稍窄
    gs = gridspec.GridSpec(1, num_plots + 1, width_ratios=width_ratios)

    for i, data in enumerate(data_buffer):
        ax = fig.add_subplot(gs[i])
        normalized_intensity = data['Intensity'].T / global_max_intensity

        pcm = ax.pcolormesh(
            data['y_um'],
            data['z_um'],
            normalized_intensity,
            shading='gouraud',
            cmap='inferno',
            vmin=0,
            vmax=1
        )

        ax.set_title(f"{data['thickness']} nm")
        ax.set_xlabel(u"Y (\u03bcm)")
        if i == 0:
            ax.set_ylabel(u"Z (\u03bcm)")
        else:
            ax.set_yticklabels([])

        ax.set_aspect('auto')
        ax.set_xlim(data['y_um'].min(), data['y_um'].max())
        ax.set_ylim(data['z_um'].min(), data['z_um'].max())

    # colorbar 单独占最后一列
    cbar_ax = fig.add_subplot(gs[-1])
    fig.colorbar(pcm, cax=cbar_ax)


    # ===== 3. 调整边距：适配 7 in 宽图 =====
    plt.suptitle(suptitle, fontsize=9, y=0.98)
    plt.subplots_adjust(
        left=0.08,   # 给 y 轴标签留点空间
        right=0.97,  # 紧一点以容纳 colorbar
        top=0.85,
        bottom=0.22,
        wspace=0.05  # 子图之间尽量紧凑
    )

    plt.savefig(output_filename, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已成功保存为: {output_filename}")
# plot_params = {
#     "DATA_DIR": r"E:\Postgraduate\Second\FDTD\VisualFunction\Project1_visual\Eyz",  # 数据目录（.mat 文件所在路径）
#     "TARGET_FREQ_IDX": 2,  # 目标频率索引，用于选择 E 数据的频带
#     "THICKNESS_LIST": [60, 80, 100, 120, 140],  # 厚度列表（单位：nm）
#     "num_plots": len([60, 80, 100, 120, 140]),  # 子图数量，与 THICKNESS_LIST 一致
#     "figsize": (3 * len([60, 80, 100, 120, 140]) + 1, 5),  # 图像尺寸：每个子图宽 3 英寸，外加 1 英寸给 colorbar
#     "dpi": 300,  # 输出图片分辨率
#     "gridspec_width_ratios": [1] * len([60, 80, 100, 120, 140]) + [0.05],  # gridspec 宽度比，最后一个为 colorbar
#     "subplots_adjust": {"top":0.85, "bottom":0.15, "left":0.05, "right":0.92, "wspace":0.15},  # 子图边距与间距设置
#     "pcolormesh": {"shading":"gouraud", "cmap":"inferno", "vmin":0, "vmax":1},  # pcolormesh 默认参数（着色、色图、值域）
#     "xlabel": "Y (µm)",  # x 轴标签文本
#     "ylabel": "Z (µm)",  # y 轴标签文本
#     "title_fontsize": 12,  # 子图标题字体大小
#     "suptitle_fontsize": 14,  # 总标题字体大小
#     "colorbar_label": "Electric Field Intensity",  # colorbar 标签文本
#     "intensity_reshape_order": "F",  # 重塑强度数组时使用的顺序（Fortran 风格）
#     "units_scale": 1e6  # 单位缩放因子（m -> µm）
# }


if __name__ == '__main__':
    # ================= 1. 通用配置区域 =================
    # 数据文件所在目录
    DATA_DIR = 'E:\Postgraduate\Second\FDTD\VisualFunction\Project1_visual\Eyz'
    # 厚度列表 (nm)
    THICKNESS_LIST = [60, 80, 100, 120, 140]
    # 目标频率索引
    TARGET_FREQ_IDX = 2

    # ================= 2. 定义不同状态的配置 =================
    configurations = [
        {
            "state_name": "Amorphous",
            "file_template": "A_thickness_{}nm.mat",
            "output_filename": "Amorphous_Thickness_Comparison.png",
            "suptitle": "Axial Optical Field Distribution (Amorphous State)"
        },
        {
            "state_name": "Crystalline",
            "file_template": "C_thickness_{}nm.mat",
            "output_filename": "Crystalline_Thickness_Comparison.png",
            "suptitle": "Axial Optical Field Distribution (Crystalline State)"
        }
    ]

    # ================= 3. 主程序流程 =================
    for config in configurations:
        print(f"\n--- 开始处理 {config['state_name']} 状态数据 ---")
        data_buffer = []

        for t in THICKNESS_LIST:
            filename = os.path.join(DATA_DIR, config["file_template"].format(t))
            processed_data = load_and_process_data(filename, TARGET_FREQ_IDX)

            if processed_data:
                processed_data['thickness'] = t
                data_buffer.append(processed_data)
                print(f"成功加载并处理: {filename}")

        # 绘图
        if data_buffer:
            plot_comparison(data_buffer, config["output_filename"], config["suptitle"])
        else:
            print(f"未找到或处理任何 {config['state_name']} 状态的数据文件，跳过绘图。")
