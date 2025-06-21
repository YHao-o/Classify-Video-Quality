import os
import cv2
from ultralytics import YOLO
import argparse
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
    arrs = []
    for file in os.listdir(folder_path):
        if file.endswith('.mp4'):
            arrs.append(os.path.join(folder_path, file))
    return arrs

# 处理单个视频文件
def task(arr):
    """
    对单个视频进行质量检测。

    Args:
        arr (str): 视频文件路径。

    Returns:
        dict: 检测结果，包括状态码、信息和分类。
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(arr)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

        # 判断视频是否可用
        if frame_count == 0:
            return {"code": "1", "msg": f"{arr} 检测异常", "class": "无法解析/无法播放"}
        if frame_count == -1:
            return {"code": "1", "msg": f"{arr} 检测异常", "class": "无画面"}

        # 初始化变量
        test_arr = []
        frame_num = 0
        flag = True
        video_type = ''
        while flag:
            ret, frame = cap.read()
            if not ret:  # 视频读取结束
                break
            frame_num += 1
            if frame_count > args.frame_empty*22:
                # print(f'{frame_num}帧',frame_num % args.frame_empty)
                if frame_num % args.frame_empty != 0:  # 跳帧检测
                    continue
            if args.compressed:
                # 缩小分辨率，例如 50%
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # 对每一帧进行模型推理
            results = model(frame, verbose=False, device=args.device)
            # 获取模型分类结果
            for result in results:
                name_dict = result.names
                top1_index = result.probs.top1
                class_name = name_dict[top1_index]
                # 更新检测结果数组，保留最近20帧的结果
                if len(test_arr) < 21:
                    test_arr.append(class_name)
                else:
                    test_arr.pop(0)

            # 检测异常类型
            if len(test_arr) >= 19:
                if 'normal' in test_arr:
                    continue
                else:
                    black_num = test_arr.count('black')
                    distort_num = test_arr.count('distort')
                    if black_num > distort_num - 5:
                        video_type = 'black'
                        test_arr.append('err')
                        break
                    else:
                        video_type = 'distort'
                        test_arr.append('err')
                        break

        # 判断检测结果
        if 'err' not in test_arr and flag:
            return {"code": "1", "msg": f"{arr} 播放正常", "class": 'normal'}
        else:
            if video_type=='black':
                return {"code": "1", "msg": f"{arr} 检测到黑屏异常", "class": video_type}
            else:
                return {"code": "1", "msg": f"{arr} 检测到花屏/闪屏/噪点等故障", "class": video_type}

    except Exception as e:
        print(e)
        # 捕获异常并返回结果
        return {"code": "0", "msg": "检测程序异常", "class": "未知异常"}

# 多线程处理多个视频
def main(arrs):
    """
    使用线程池并发处理多个视频文件。

    Args:
        arrs (list): 视频文件路径列表。

    Yields:
        dict: 每个视频的检测结果。
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(task, arr) for arr in arrs]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
            yield future.result()

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="视频质量检测程序")
    parser.add_argument("-p", "--folder_path", default='videos', type=str, help="视频文件夹路径")
    parser.add_argument("-u", "--url", type=str, help="视频 URL 数组")
    parser.add_argument("-m", "--model", type=str, default='./models/best.pt', help="模型路径")
    parser.add_argument("-f", "--frame_empty", type=int, default=4, help="跳帧检测间隔")
    parser.add_argument('-d', "--device",type=str, default='0', help='设备类型')
    parser.add_argument('-c', "--compressed",type=bool, default=True, help='是否压缩')
    args = parser.parse_args()

    # 加载 YOLO 模型
    model = YOLO(args.model)
    # 如果指定了文件夹路径，则从文件夹读取视频
    if args.folder_path and args.url is None:
        video_arr = read_folder(args.folder_path)
    else:
        video_arr = args.url.split(",")
    # 调用主函数并输出结果
    results = list(main(video_arr))

