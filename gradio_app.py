import argparse
import os
import tempfile
from typing import List, Tuple

import fitz
import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from conv_for_infer import generate_conv


MODEL = None
PROCESSOR = None
DEVICE = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models/FireRed-OCR")
    parser.add_argument("--processor_dir", type=str, default="./models/FireRed-OCR")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max_pages", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    return parser.parse_args()


def load_model_once(model_dir: str, processor_dir: str):
    global MODEL, PROCESSOR, DEVICE
    if MODEL is not None and PROCESSOR is not None:
        return

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir 不存在或不是目录: {model_dir}")
    if not os.path.isdir(processor_dir):
        raise FileNotFoundError(f"processor_dir 不存在或不是目录: {processor_dir}")

    if torch.cuda.is_available():
        DEVICE = "cuda"
        dtype = torch.bfloat16
    else:
        DEVICE = "cpu"
        dtype = torch.float32

    PROCESSOR = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
    MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        MODEL.to("cpu")
    MODEL.eval()
    print(f"[Init] Device: {DEVICE}; cuda_available={torch.cuda.is_available()}; cuda_devices={torch.cuda.device_count()}")


def extract_page_blocks(page: fitz.Page):
    data = page.get_text("dict")
    mixed_blocks = []
    for block in data.get("blocks", []):
        block_type = block.get("type")
        bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        if block_type == 0:
            lines = []
            for line in block.get("lines", []):
                spans = []
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text and text.strip():
                        spans.append(text)
                if spans:
                    lines.append("".join(spans).strip())
            text_content = "\n".join(lines).strip()
            if text_content:
                mixed_blocks.append({"kind": "text", "bbox": bbox, "text": text_content})
        elif block_type == 1:
            mixed_blocks.append({"kind": "image", "bbox": bbox})

    mixed_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return mixed_blocks


def render_page_image(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def crop_image_by_bbox(page_image: Image.Image, bbox, dpi: int):
    scale = dpi / 72.0
    width, height = page_image.size
    x0, y0, x1, y1 = bbox
    left = max(0, min(width, int(x0 * scale)))
    top = max(0, min(height, int(y0 * scale)))
    right = max(0, min(width, int(x1 * scale)))
    bottom = max(0, min(height, int(y1 * scale)))
    if right - left < 20 or bottom - top < 20:
        return None
    return page_image.crop((left, top, right, bottom))


@torch.inference_mode()
def infer_single_image(image: Image.Image, max_new_tokens: int) -> str:
    messages = generate_conv(image)
    inputs = PROCESSOR.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(MODEL.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = MODEL.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text.strip()


def run_pdf_ocr(
    pdf_file,
    max_pages: int,
    dpi: int,
    max_new_tokens: int,
    model_dir: str,
    processor_dir: str,
):
    if pdf_file is None:
        return "请先上传 PDF 文件。", [], "", None

    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    page_count = min(total_pages, max_pages)
    if page_count <= 0:
        doc.close()
        return "PDF 无可处理页面。", [], "", None

    markdown_parts = []
    preview_images = []
    vector_page_count = 0
    ocr_page_count = 0
    mixed_page_count = 0
    ocr_region_count = 0

    for idx in range(page_count):
        page = doc[idx]
        blocks = extract_page_blocks(page)
        page_parts = []
        page_has_vector = False
        page_has_ocr = False
        page_image = None

        for block in blocks:
            if block["kind"] == "text":
                page_parts.append(block["text"])
                page_has_vector = True
                continue

            if MODEL is None or PROCESSOR is None:
                load_model_once(model_dir=model_dir, processor_dir=processor_dir)
            if page_image is None:
                page_image = render_page_image(page, dpi=dpi)
            cropped = crop_image_by_bbox(page_image, block["bbox"], dpi=dpi)
            if cropped is None:
                continue
            ocr_text = infer_single_image(cropped, max_new_tokens=max_new_tokens)
            if ocr_text:
                page_parts.append(ocr_text)
                page_has_ocr = True
                ocr_region_count += 1

        if not page_parts:
            if MODEL is None or PROCESSOR is None:
                load_model_once(model_dir=model_dir, processor_dir=processor_dir)
            if page_image is None:
                page_image = render_page_image(page, dpi=dpi)
            fallback_text = infer_single_image(page_image, max_new_tokens=max_new_tokens)
            if fallback_text:
                page_parts.append(fallback_text)
                page_has_ocr = True
                ocr_region_count += 1

        if page_has_vector and page_has_ocr:
            mixed_page_count += 1
        elif page_has_vector:
            vector_page_count += 1
        elif page_has_ocr:
            ocr_page_count += 1

        if page_has_ocr and page_image is not None:
            preview_images.append(page_image)

        page_text = "\n\n".join(page_parts).strip()
        markdown_parts.append(f"## Page {idx + 1}\n\n{page_text}")

    doc.close()

    final_markdown = "\n\n".join(markdown_parts)
    status = (
        f"处理完成：共 {total_pages} 页，已处理 {page_count} 页。"
        f" 纯矢量页 {vector_page_count}，纯OCR页 {ocr_page_count}，混合页 {mixed_page_count}，"
        f"OCR区域 {ocr_region_count} 个。"
    )

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0] or "ocr_result"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        prefix=f"{pdf_name}_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(final_markdown)
        download_path = f.name

    return status, preview_images, final_markdown, download_path


def build_demo(args):
    def run_with_args(pdf_file, max_pages, dpi, max_new_tokens):
        return run_pdf_ocr(
            pdf_file,
            max_pages=max_pages,
            dpi=dpi,
            max_new_tokens=max_new_tokens,
            model_dir=args.model_dir,
            processor_dir=args.processor_dir,
        )

    with gr.Blocks(title="FireRed-OCR PDF Frontend") as demo:
        gr.Markdown("# FireRed-OCR PDF 识别")
        gr.Markdown("上传 PDF 后自动处理：同一页内会混合提取矢量文本和图片区域 OCR，并按版面顺序合并。")

        with gr.Row():
            pdf_input = gr.File(label="上传 PDF", file_types=[".pdf"], type="filepath")
            with gr.Column():
                max_pages = gr.Slider(1, 400, value=args.max_pages, step=1, label="最大处理页数")
                dpi = gr.Slider(100, 300, value=args.dpi, step=10, label="PDF 渲染 DPI")
                max_new_tokens = gr.Slider(
                    256, 8192, value=args.max_new_tokens, step=256, label="每页最大生成 Token"
                )
                run_btn = gr.Button("开始识别", variant="primary")

        status = gr.Textbox(label="状态", interactive=False)
        page_gallery = gr.Gallery(label="PDF 页面预览", columns=2, height=420)
        markdown_output = gr.Textbox(label="识别结果（Markdown）", lines=24)
        download_file = gr.File(label="下载识别结果（Markdown）")

        run_btn.click(
            fn=run_with_args,
            inputs=[pdf_input, max_pages, dpi, max_new_tokens],
            outputs=[status, page_gallery, markdown_output, download_file],
        )

    return demo


def main():
    args = parse_args()
    demo = build_demo(args)
    demo.queue(max_size=8).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
