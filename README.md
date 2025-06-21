# 视频质量检测工具

## 简介

本项目包含两个主要脚本，用于实现视频的分类和帧截取任务：

1. **`cls_video.py`**：对指定目录下的视频进行质量检测，识别异常状态（如黑屏、花屏/闪屏/噪点、无法播放、无法解析、无画面等）。
2. **`crop_video.py`**：将视频逐帧分类并将结果存储为图片，用于进一步的数据分析或训练。

这些工具使用 [YOLO](https://github.com/ultralytics/ultralytics) 深度学习框架实现模型推理，并支持批量处理和多线程优化。

---

## 主要功能

### 1. 视频质量检测（`cls_video.py`）

- 支持批量读取视频文件。
- 跳帧检测以优化推理速度。
- 可开启画质压缩参数以进一步优化处理效率。
- 识别视频中的以下异常状态：
  - 黑屏（`black`）
  - 花屏、闪屏、噪点等故障（`distort`）
- 多线程支持，提升多视频处理效率。
- 可选择显卡以加速处理。
- 支持通过命令行指定参数。

### 2. 视频帧截取（`crop_video.py`）

- 针对视频逐帧检测，分类并存储到指定文件夹中。
- 输出以下类别的帧：
  - 黑屏（`black`）
  - 花屏（`distort`）
  - 正常（`normal`）
- 生成的图片可用于构建数据集或后续处理。

---

## 环境依赖

1. **Python** >= 3.8
2. 依赖库：
   - `opencv-python`
   - `ultralytics`
   - `argparse`
   - `concurrent.futures`

安装依赖项：

```bash
conda create -n video_cls python==3.8
conda activate video_cls
pip install ultralytics
#根据情况安装torch
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## 快速开始

### 视频质量检测

1. **运行 `cls_video.py`**：
   ```bash
   python cls_video.py -p <视频文件夹路径> -m <模型路径> -f <跳帧间隔> -d <设备选择> -c <是否启用视频质量压缩>
   ```
   示例：
   ```bash
   python cls_video.py -p ./videos -m ./models/best.pt -f 4 -d 0 -c True
   ```
   示例：
   ```bash
   #传入视频url地址检测，可传入多个，以逗号分隔
   python cls_video.py -u  "https://prod-streaming-video-msn-com.akamaized.net/3d6f4af0-79ab-46fe-9d33-e191be5a878e/b4fa3f3e-a582-4bb5-9115-a82652e45b65.mp4","https://prod-streaming-video-msn-com.akamaized.net/3d6f4af0-79ab-46fe-9d33-e191be5a878e/b4fa3f3e-a582-4bb5-9115-a82652e45b65.mp4" -d 0 -c True
   ```
2. **参数说明**：
   - `-p` / `--folder_path`：待检测视频文件夹路径（默认：`./videos`）。
   - `-u` / `--url`：视频文件的 URL 数组（以逗号分隔，替代文件夹路径）。
   - `-m` / `--model`：YOLO 模型权重路径（默认：`./models/best.pt`）。
   - `-f` / `--frame_empty`：跳帧间隔，减少处理帧数提高速度（默认：`4`）。
   - `-d` / `--device`: 设备选择，(默认为 `cpu`)。
   - `-c` / `--compress`：是否启用视频质量压缩，以减少处理时间（默认：`False`）。
3. **输出**：
   检测结果会以 JSON 格式打印到控制台，包含以下字段：
   - `code`：状态码（`1` 表示正常，`0` 表示异常）。
   - `msg`：检测结果信息。
   - `class`：检测类别（`normal`、`black` 或 `distort`）。

---

### 视频帧截取

1. **运行 `crop_video.py`**：

   ```bash
   python crop_video.py -p <视频文件夹路径> -m <模型路径>
   ```

   示例：

   ```bash
   python crop_video.py -p ./videos -m ./models/best.pt
   ```

2. **参数说明**：

   - `-p` / `--folder_path`：待处理视频文件夹路径（默认：`./视频`）。
   - `-m` / `--model`：YOLO 模型权重路径（默认：`./models/best.pt`）。

3. **输出**：
   - 将每帧分类为 `black`、`distort` 或 `normal`。
   - 分类结果存储在以下文件夹中：
     - `./dataset_pre/black/`
     - `./dataset_pre/distort/`
     - `./dataset_pre/normal/`

---

## 文件结构

```
.
├── cls_video.py              # 视频质量检测脚本
├── utils/
│   └── crop_video.py         # 视频帧截取脚本
├── models/                   # 存储 YOLO 模型权重
│   └── best.pt               # 示例模型权重
├── videos/                   # 待检测视频文件夹
└── dataset_pre/              # 帧截取结果保存路径
    ├── black/                # 黑屏帧
    ├── distort/              # 花屏帧
    └── normal/               # 正常帧
```

---

## 注意事项

1. 确保 YOLO 模型权重文件与使用的分类类别匹配。
2. 如果视频帧较多，建议调整 `frame_empty` 参数以提升检测效率。
3. 截取结果的保存路径需提前存在，或在脚本中自动创建。

---

## 作者信息

如果对该项目有任何问题或建议，请联系开发者。

- **作者**: [YHao-o]
- **邮箱**: [woyuanhaoa@gamil.com]
- **版本**: 1.0.0

---
