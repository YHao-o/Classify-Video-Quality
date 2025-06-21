import argparse
import os
import cv2
from ultralytics import YOLO
import concurrent.futures

# 读取文件夹中的所有视频文件
def read_folder(folder_path):
    """
    遍历指定文件夹，获取所有以 .mp4 结尾的视频文件路径。

    Args:
        folder_path (str): 文件夹路径。

    Returns:
        list: 包含所有视频文件路径的列表。
    """
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mp4')]

# 处理单个视频文件
def process_video(file_path, model):
    """
    对单个视频进行处理并分类。

    Args:
        file_path (str): 视频文件路径。
        model (YOLO): YOLO 模型实例。
    """
    cap = cv2.VideoCapture(file_path)
    sum_ad = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if sum_ad < 2:
        print(f"视频 {file_path} 帧数过少，跳过处理。")
        return

    num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        num += 1
        results = model(frame, device=0,verbose=False)
        class_name = ''
        for result in results:
            name_dict = result.names
            top1_index = result.probs.top1
            class_name = name_dict[top1_index]

        # 保存到相应的目录
        output_dir = f'../dataset_pre/{class_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(os.path.join(output_dir, f"{os.path.basename(file_path)[:5]}_{num}.jpg"), frame)
        print(f"视频 {file_path}, 帧 {num}/{sum_ad}, 分类: {class_name}")

# 使用线程池并发处理多个视频文件
def main(folder_path, model):
    """
    使用线程池处理视频。

    Args:
        folder_path (str): 视频文件夹路径。
        model (YOLO): YOLO 模型实例。
    """
    video_files = read_folder(folder_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_video, file_path, model): file_path for file_path in video_files}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # 确保捕获任务中出现的异常
            except Exception as e:
                print(f"处理视频 {futures[future]} 时发生错误: {e}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="视频质量检测程序")
    parser.add_argument("-p", "--folder_path", default='../test', type=str, help="视频文件夹路径")
    parser.add_argument("-m", "--model", type=str, default='../models/best.pt', help="模型路径")
    args = parser.parse_args()

    # 确保分类目录存在
    for category in ['black', 'distort', 'normal']:
        os.makedirs(f'../dataset_pre/{category}', exist_ok=True)

    # 加载 YOLO 模型
    model = YOLO(args.model)

    # 调用主函数
    main(args.folder_path, model)
