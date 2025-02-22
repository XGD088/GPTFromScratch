import os


def get_current_dir():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件的目录
    current_dir = os.path.dirname(current_file_path)

    return current_dir
