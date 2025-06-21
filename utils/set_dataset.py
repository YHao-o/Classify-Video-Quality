import os
import random
import shutil


def read_folder(folder_path):
    black_list=[]
    distort_list=[]
    normal_list=[]
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path=os.path.join(root, file)
            if file.endswith('.jpg'):
                if 'black' in file_path:
                    black_list.append(file_path)
                elif 'distort' in file_path:
                    distort_list.append(file_path)
                else:
                    normal_list.append(file_path)
    return black_list, distort_list, normal_list

# 创建必要的文件夹
def create_directories():
    for folder in ['train', 'val', 'test']:
        for subfolder in ['black', 'distort', 'normal']:
            path = os.path.join(folder, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

# 分配数据
def split_data(data_list, train_ratio=0.9):
    random.shuffle(data_list)
    split_index = int(len(data_list) * train_ratio)
    train_data = data_list[:split_index]
    val_data = data_list[split_index:]
    return train_data, val_data


# 保存数据
def save_data(data, folder, subfolder):
    for item in data:
        src_path = item  # 假设 item 是源文件路径
        dst_path = os.path.join(folder, subfolder, os.path.basename(item))
        shutil.copy(src_path, dst_path)
        print(f"复制文件 {src_path} 到 {dst_path}")

# 主函数
def main(black_list, distort_list, normal_list):
    create_directories()

    # 分配 black_list
    black_train, black_val = split_data(black_list)
    print("开始保存 black 训练集...")
    save_data(black_train, 'train', 'black')
    print("开始保存 black 验证集...")
    save_data(black_val, 'val', 'black')

    # 分配 distort_list
    distort_train, distort_val = split_data(distort_list)
    print("开始保存 distort 训练集...")
    save_data(distort_train, 'train', 'distort')
    print("开始保存 distort 验证集...")
    save_data(distort_val, 'val', 'distort')

    # 分配 normal_list
    normal_train, normal_val = split_data(normal_list)
    print("开始保存 normal 训练集...")
    save_data(normal_train, 'train', 'normal')
    print("开始保存 normal 验证集...")
    save_data(normal_val, 'val', 'normal')

    print("数据保存完成")


if __name__ == "__main__":
    black_list, distort_list, normal_list = read_folder('../dataset_pre')
    main(black_list, distort_list, normal_list)
