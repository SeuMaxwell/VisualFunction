import h5py
import numpy as np
import os
from typing import Optional, Dict, Any, List

class MatFileHandler:
    """
    一个用于处理和检查 .mat (HDF5 格式) 文件的类。
    """

    def __init__(self, file_path: str):
        """
        初始化 MatFileHandler。

        :param file_path: .mat 文件的路径。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"错误: 文件 '{file_path}' 不存在。")
        self.file_path = file_path
        self.file = None

    def __enter__(self):
        """上下文管理器入口：打开文件。"""
        self.file = h5py.File(self.file_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口：关闭文件。"""
        if self.file:
            self.file.close()

    def summarize_contents(self):
        """
        以简洁的方式打印文件顶层组及其直接包含的数据集。
        """
        print(f"--- 文件 '{self.file_path}'  ---")
        for group_name in self.file.keys():
            group = self.file[group_name]
            if isinstance(group, h5py.Group):
                print(f"\n组: '{group_name}'")
                for key, item in group.items():
                    if isinstance(item, h5py.Dataset):
                        print(f"  - 数据集: {key:<10} | 维度: {item.shape} | 类型: {item.dtype}")

    def load_data(self, group_name: str, fields: List[str]) -> Optional[Dict[str, np.ndarray]]:
        """
        从指定的组中加载多个数据集。

        :param group_name: 包含数据集的组名。
        :param fields: 需要加载的数据集名称列表。
        :return: 包含加载数据的字典，如果失败则返回 None。
        """
        try:
            if group_name not in self.file:
                print(f"错误: 在 '{self.file_path}' 中未找到组 '{group_name}'。")
                return None

            data_group = self.file[group_name]
            loaded_data = {}
            for field in fields:
                if field in data_group:
                    loaded_data[field] = np.array(data_group[field])
            return loaded_data
        except KeyError as e:
            print(f"错误: 读取数据时键错误: {e}。")
            return None


def main():
    """主执行函数"""
    file_name = r'E:\Postgraduate\Second\FDTD\VisualFunction\pro1_visual\E5.mat'

    try:
        with MatFileHandler(file_name) as handler:
            # 1. 显示文件内容摘要
            handler.summarize_contents()

            # 2. 加载指定数据
            group_to_load = 'E_data'
            fields_to_load = ['y', 'z', 'E']
            print(f"\n--- 正在加载 '{group_to_load}' 组的数据 ---")
            e_data = handler.load_data(group_to_load, fields_to_load)

            # 3. 使用加载的数据
            if e_data:
                print("成功加载数据:")
                for field, data in e_data.items():
                    print(f" - '{field}' 维度: {data.shape}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        import traceback
        print(f"程序执行时发生未知错误: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()

