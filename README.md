# FireRed-OCR 中文说明

本项目基于 FireRed-OCR（Qwen3-VL 架构），用于将文档图像/PDF 转换为结构化 Markdown。

## 功能概览

- 支持图片 OCR 推理（基于 `transformers`）。
- 支持 Gradio Web 前端上传 PDF。
- 支持 PDF 页内混合提取：
  - 页面中的矢量文字直接提取；
  - 页面中的图片区域单独 OCR；
  - 按版面顺序合并输出 Markdown。
- 支持识别结果 `.md` 文件下载。
- 支持 CUDA 自动加速（可用 GPU 时自动使用）。
- 支持本地模型目录加载（不从远程拉取）。

## 环境准备

建议使用 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- `torch`
- `transformers`
- `qwen-vl-utils`
- `gradio`
- `pymupdf`
- `pillow`
- `tqdm`

## 本地模型目录

请先把模型文件放到本地目录（示例）：

```text
./models/FireRed-OCR
```

程序默认读取：

- `--model_dir ./models/FireRed-OCR`
- `--processor_dir ./models/FireRed-OCR`

如果目录不存在，程序会直接报错。

## 启动 Gradio 前端（推荐）

```bash
python3 gradio_app.py \
  --model_dir ./models/FireRed-OCR \
  --processor_dir ./models/FireRed-OCR \
  --host 0.0.0.0 \
  --port 7860
```

启动后访问：

- `http://127.0.0.1:7860`

前端支持：

- 上传 PDF
- 设置最大处理页数
- 设置 PDF 渲染 DPI
- 设置每页最大生成 Token
- 查看处理状态
- 查看 OCR 页面预览
- 下载 Markdown 结果文件

## PDF 处理逻辑（当前实现）

对每一页执行以下步骤：

1. 提取页面块（文本块、图片块）并按坐标排序。
2. 文本块：直接提取矢量文字。
3. 图片块：按块裁剪后调用 OCR。
4. 将文本提取结果和 OCR 结果按版面顺序合并。
5. 若块级结果为空，回退为整页 OCR。

最终输出为按页组织的 Markdown：

```markdown
## Page 1
...

## Page 2
...
```

## CUDA 加速说明

程序会在启动时自动判断：

- `torch.cuda.is_available() == True`：使用 CUDA
- 否则使用 CPU

日志会打印设备信息，示例：

```text
[Init] Device: cuda; cuda_available=True; cuda_devices=1
```

## 命令行脚本（原项目）

仓库中保留了原始推理脚本：

- `qwen3_hf_infer.py`：基于 transformers
- `qwen3_vllm_infer.py`：基于 vLLM（需额外安装）

## 常见问题

1. 报错 `local_files_only=True` 找不到模型  
原因：本地模型目录不完整或路径错误。  
处理：检查 `--model_dir`、`--processor_dir` 是否指向有效目录。

2. 显示使用 CPU  
原因：当前环境未检测到可用 CUDA。  
处理：检查 CUDA 驱动、PyTorch CUDA 版本、`torch.cuda.is_available()`。

3. PDF 中一页同时有矢量文字和图片文字  
当前已支持页内混合提取，不会只走单一路径。

## 相关文件

- Web 前端：`gradio_app.py`
- 提示词构造：`conv_for_infer.py`
- 依赖清单：`requirements.txt`

