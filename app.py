from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import uvicorn
import asyncio
import json
import os
import uuid
import cv2
import numpy as np
import shutil
import pickle
from skimage.segmentation import find_boundaries

from domain import suggest_mix, generate_steps_from_mix # 依赖domain/__init__.py 显式导出
from domain import segment_hue_masks, generate_hue_mean_colored_masks
from domain.line_art import line_art
from domain.light import process_light_color_blocks_directory
from domain.dark import process_dark_color_blocks_directory


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(BASE_DIR, "static", "originals")
PROCESSED_DIR = os.path.join(BASE_DIR, "static", "processed")
SEGMENTED_DIR = os.path.join(BASE_DIR, "static", "segmented")
SEGMENTED_DATA_DIR = os.path.join(BASE_DIR, "static", "segmented_data")
LINE_ART_DIR = os.path.join(BASE_DIR, "static", "line_art")
GRAY_FADE_DIR = os.path.join(BASE_DIR, "static", "gray_fade")
COLOR_BLOCKS_DIR = os.path.join(BASE_DIR, "static", "color_blocks")
COLOR_MASKS_DIR = os.path.join(BASE_DIR, "static", "color_masks")
LIGHT_DETAILS_DIR = os.path.join(BASE_DIR, "static", "light_details")
LIGHT_DIR = os.path.join(BASE_DIR, "static", "light")
DARK_DETAILS_DIR = os.path.join(BASE_DIR, "static", "dark_details")
DARK_DIR = os.path.join(BASE_DIR, "static", "dark")

# 后端对外访问的基础 URL，需要与你前端配置的 BACKEND_BASE_URL 保持一致
# 示例：前端配置为 export const BACKEND_BASE_URL = 'http://172.16.25.51:8000'
# 这里就写成相同的地址，便于返回完整的 projectorImageUrl
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://172.16.20.50:8000")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LINE_ART_DIR, exist_ok=True)
os.makedirs(COLOR_BLOCKS_DIR, exist_ok=True)
os.makedirs(COLOR_MASKS_DIR, exist_ok=True)
os.makedirs(LIGHT_DETAILS_DIR, exist_ok=True)
os.makedirs(LIGHT_DIR, exist_ok=True)
os.makedirs(DARK_DETAILS_DIR, exist_ok=True)
os.makedirs(DARK_DIR, exist_ok=True)

app = FastAPI(title="Painting Helper Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录，确保 /static/... 可以通过 HTTP 直接访问
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


class RegisterOriginalRequest(BaseModel):
    imageUrl: str
    width: int
    height: int


class PixelCoord(BaseModel):
    x: int
    y: int


class RatioCoord(BaseModel):
    x: float
    y: float


class ColorMixFromClickByIdRequest(BaseModel):
    imageId: str
    pixel: PixelCoord
    ratio: Optional[RatioCoord] = None
    layer: str = "segment_large"  # Default to large segment for better UX



class ColorMixFromClickByUrlRequest(BaseModel):
    imageUrl: str
    pixel: PixelCoord
    layer: Optional[str] = None  # "original", "segment_large", "segment_small"


class LineArtRequest(BaseModel):
    imageUrl: str
    mode: Optional[str] = "sketch"



@app.post("/api/painting/upload-original-file")
async def upload_original_file(file: UploadFile = File(...)):
    """上传原始图片文件，保存到 static/originals 下并返回可访问的 imageUrl。

    对齐 BACKEND_API_README.md 中的 /api/painting/upload-original-file 规范：
    - 入参：multipart/form-data, 字段名为 file
    - 出参：data.imageUrl 为后端可直接访问的完整 HTTP URL
    """
    # 固定命名为 0，保留原始扩展名（例如 0.jpg / 0.png）
    ext = os.path.splitext(file.filename or "")[1] or ".png"
    filename = f"0{ext}"
    save_path = os.path.join(ORIGINAL_DIR, filename)

    # 保存上传的文件内容
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # 生成静态相对路径和完整 URL
    rel_static_path = os.path.relpath(save_path, BASE_DIR).replace("\\", "/")
    image_url = _build_public_url_from_static_path(rel_static_path)

    # --- 清理旧的原图 ---
    # 只保留本次上传的 filename
    _clear_directory_except(ORIGINAL_DIR, [filename])

    # --- 清理旧的色相掩码（color_masks）---
    # 每次上传新图，先清空 static/color_masks 目录，避免遗留上一次的掩码文件
    _clear_directory_except(COLOR_MASKS_DIR, [])

    # 读取图片以供后续处理 (OpenCV BGR -> RGB)
    img_bgr = cv2.imread(save_path)
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to read saved image after upload")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    filename_base = os.path.splitext(filename)[0]

    # --- 生成线稿图 (Line Art) ---
    # 新的 line_art 返回素描风格灰度图（uint8，白底黑线），可直接保存
    line_art_img = line_art(img_rgb)
    
    # 线稿固定命名为 0.png
    filename_lineart = "0.png"
    path_lineart = os.path.join(LINE_ART_DIR, filename_lineart)
    cv2.imwrite(path_lineart, line_art_img)

    # --- 生成原图色块图（按色块大小降序）---
    try:
        # 每次上传新图，清空 static/color_blocks 目录
        _clear_directory_except(COLOR_BLOCKS_DIR, [])
        masks, blocks = segment_hue_masks(
            img_rgb,
            input_bgr=False,
            min_saturation=0.1,
            min_value=0.1,
            bin_size_deg=2,
            smooth_sigma_deg=6,
            min_prominence_ratio=0.01,  # 放宽峰显著性，避免遗漏真实色块
            min_ratio=0.0,              # 不过滤小块，交给形态学 min_area 处理
            open_size=3,
            close_size=7,
            iterations_open=1,
            iterations_close=1,
            min_area=None,
        )
        color_block_files = []
        for i, mask in enumerate(masks):
            # 用原图 BGR 与掩码相与，保留该色块的原始颜色
            block_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
            fname = f"{i}.png"
            fpath = os.path.join(COLOR_BLOCKS_DIR, fname)
            cv2.imwrite(fpath, block_img)
            color_block_files.append(fname)
        # 清理目录中非本次生成的文件
        _clear_directory_except(COLOR_BLOCKS_DIR, color_block_files)
        color_block_urls = [
            _build_public_url_from_static_path(
                os.path.relpath(os.path.join(COLOR_BLOCKS_DIR, f), BASE_DIR).replace("\\", "/")
            )
            for f in color_block_files
        ]

        # --- 生成按色相均值着色的掩码图（color_masks）---
        color_mask_files = []
        try:
            if isinstance(masks, list) and len(masks) > 0:
                value_scale = float(os.environ.get("HUE_MASK_VALUE_SCALE", "0.8"))
                color_mask_files = generate_hue_mean_colored_masks(
                    img_rgb=img_rgb,
                    masks=masks,
                    output_dir=COLOR_MASKS_DIR,
                    value_scale=value_scale,
                )
        except Exception as e:
            # 不影响主流程
            print(f"generate hue mean-color masks failed: {e}")

    # --- light_details 亮部细节---
        # 可通过环境变量调整阈值（便于细节处理）：LIGHT_V_PERCENTILE / LIGHT_S_PERCENTILE
        v_thr = float(os.environ.get("LIGHT_V_PERCENTILE", "0.80"))
        s_thr = float(os.environ.get("LIGHT_S_PERCENTILE", "0.10"))

        # 清空旧的 light_details
        _clear_directory_except(LIGHT_DETAILS_DIR, [])
        _ = process_light_color_blocks_directory(
            input_dir=COLOR_BLOCKS_DIR,
            output_dir=LIGHT_DETAILS_DIR,
            v_percentile=v_thr,
            s_percentile=s_thr,
            morph_kernel_size=1,
            min_region_area=10,
        )
        # 仅返回与当前 color_block_files 对应的亮部掩码 URL（按同名文件）
        light_detail_urls = [
            _build_public_url_from_static_path(
                os.path.relpath(os.path.join(LIGHT_DETAILS_DIR, f), BASE_DIR).replace("\\", "/")
            )
            for f in color_block_files
            if os.path.exists(os.path.join(LIGHT_DETAILS_DIR, f))
        ]

    # --- light 亮部---
        v_thr = float(os.environ.get("LIGHT_V_PERCENTILE", "0.70"))
        s_thr = float(os.environ.get("LIGHT_S_PERCENTILE", "0.50"))

        # 清空旧的 light
        _clear_directory_except(LIGHT_DIR, [])
        _ = process_light_color_blocks_directory(
            input_dir=COLOR_BLOCKS_DIR,
            output_dir=LIGHT_DIR,
            v_percentile=v_thr,
            s_percentile=s_thr,
            morph_kernel_size=3,
            min_region_area=10,
        )
        # 仅返回与当前 color_block_files 对应的亮部掩码 URL（按同名文件）
        light_urls = [
            _build_public_url_from_static_path(
                os.path.relpath(os.path.join(LIGHT_DIR, f), BASE_DIR).replace("\\", "/")
            )
            for f in color_block_files
            if os.path.exists(os.path.join(LIGHT_DIR, f))
        ]

    # --- dark_details 暗部细节---
        # 通过环境变量调整暗部阈值：DARK_V_PERCENTILE（暗部用更低的V分位）
        v_thr_dark = float(os.environ.get("DARK_V_PERCENTILE", "0.1"))

        # 清空旧的 dark_details
        _clear_directory_except(DARK_DETAILS_DIR, [])
        _ = process_dark_color_blocks_directory(
            input_dir=COLOR_BLOCKS_DIR,
            output_dir=DARK_DETAILS_DIR,
            v_percentile=v_thr_dark,
            morph_kernel_size=1,
            min_region_area=10,
        )
        # 与当前 color_block_files 对应的暗部掩码 URL（按同名文件）
        dark_detail_urls = [
            _build_public_url_from_static_path(
                os.path.relpath(os.path.join(DARK_DETAILS_DIR, f), BASE_DIR).replace("\\", "/")
            )
            for f in color_block_files
            if os.path.exists(os.path.join(DARK_DETAILS_DIR, f))
        ]
    # --- dark 暗部---
        # 通过环境变量调整暗部阈值：DARK_V_PERCENTILE（暗部用更低的V分位）
        v_thr_dark = float(os.environ.get("DARK_V_PERCENTILE", "0.30"))

        # 清空旧的 dark
        _clear_directory_except(DARK_DIR, [])
        _ = process_dark_color_blocks_directory(
            input_dir=COLOR_BLOCKS_DIR,
            output_dir=DARK_DIR,
            v_percentile=v_thr_dark,
            morph_kernel_size=3,
            min_region_area=10,
        )
        # 与当前 color_block_files 对应的暗部掩码 URL（按同名文件）
        dark_urls = [
            _build_public_url_from_static_path(
                os.path.relpath(os.path.join(DARK_DIR, f), BASE_DIR).replace("\\", "/")
            )
            for f in color_block_files
            if os.path.exists(os.path.join(DARK_DIR, f))
        ]
    
    except Exception as e:
        print(f"color blocks generation failed: {e}")
        color_block_urls = []
        light_urls = []
        dark_detail_urls = []
        dark_urls = []
        color_mask_files = []

    # --- 清理旧的派生数据 ---
    # 保留线稿与本次生成的 color_blocks，清理多余文件
    _clear_directory_except(LINE_ART_DIR, [filename_lineart])
    _clear_directory_except(COLOR_BLOCKS_DIR, [os.path.basename(url.split('/static/')[-1]) for url in color_block_urls])
    _clear_directory_except(LIGHT_DETAILS_DIR, [os.path.basename(url.split('/static/')[-1]) for url in light_detail_urls])
    _clear_directory_except(LIGHT_DIR, [os.path.basename(url.split('/static/')[-1]) for url in light_urls])
    _clear_directory_except(DARK_DETAILS_DIR, [os.path.basename(url.split('/static/')[-1]) for url in dark_detail_urls])
    _clear_directory_except(DARK_DIR, [os.path.basename(url.split('/static/')[-1]) for url in dark_urls])
    _clear_directory_except(COLOR_MASKS_DIR, color_mask_files)


    return {
        "code": 0,
        "message": "ok",
        "data": {
            "imageUrl": image_url,
            "colorBlockUrls": color_block_urls,
            "lightDetailUrls": light_detail_urls,
            "lightUrls": light_urls,
            "darkDetailUrls": dark_detail_urls,
            "darkUrls": dark_urls,
        },
    }


@app.get("/api/projector/gray-fade-stream")
async def gray_fade_stream(interval: float = 0.5):
    """Server-Sent Events stream of gray_fade images.

    Behavior:
    - Emits JSON events {"imageUrl": <full-url>, "index": <int>} every `interval` seconds.
    - Order: 0..N-1, then N-1..0, and repeats in a loop.
    - Uses files present under static/gray_fade (numeric PNGs).
    """

    # Discover available frames under static/gray_fade as integers
    if not os.path.exists(GRAY_FADE_DIR):
        raise HTTPException(status_code=404, detail={"code": 4004, "message": "gray_fade directory not found", "data": None})

    indices: List[int] = []
    for name in os.listdir(GRAY_FADE_DIR):
        base, ext = os.path.splitext(name)
        if ext.lower() == ".png":
            try:
                idx = int(base)
                indices.append(idx)
            except ValueError:
                continue

    if not indices:
        raise HTTPException(status_code=404, detail={"code": 4005, "message": "no gray_fade frames available", "data": None})

    indices = sorted(indices)

    # Precompute full URLs for discovered frames
    rel_paths = [os.path.relpath(os.path.join(GRAY_FADE_DIR, f"{i}.png"), BASE_DIR).replace("\\", "/") for i in indices]
    urls = [_build_public_url_from_static_path(p) for p in rel_paths]

    async def event_generator():
        # forward indices then backward
        forward = list(range(len(indices)))
        backward = list(reversed(forward))
        while True:
            for pos in forward:
                payload = {"code": 0, "message": "ok", "data": {"imageUrl": urls[pos], "index": indices[pos]}}
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(interval)
            for pos in backward:
                payload = {"code": 0, "message": "ok", "data": {"imageUrl": urls[pos], "index": indices[pos]}}
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(interval)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# 简单的内存映射，生产环境可替换为数据库
IMAGE_REGISTRY: Dict[str, Dict] = {}


def _build_public_url_from_static_path(static_rel_path: str) -> str:
    """将 static 下的相对路径转换为完整可访问 URL。

    例如: static_rel_path = "static/originals/xxx.png"
    返回: f"{BACKEND_BASE_URL}/static/originals/xxx.png"
    """
    rel_path = static_rel_path.lstrip("/")
    return f"{BACKEND_BASE_URL.rstrip('/')}/{rel_path}"


def _load_image_from_url_or_path(image_url: str) -> Tuple[np.ndarray, str]:
    """从本地静态目录或绝对路径读取图片。

    当前实现主要支持两类 imageUrl:
    - 形如 "static/originals/xxx.png" 的相对路径（推荐）
    - 绝对文件路径（仅用于本地调试）
    """

    # 以 http 开头的 URL 暂不直接下载，前端应先通过 upload-original-file 获取本地 static 路径
    if image_url.startswith("http://") or image_url.startswith("https://"):
        # 提示前端按协议先上传原图，获得 static 路径
        raise FileNotFoundError(f"remote http(s) urls are not supported directly: {image_url}")

    # 如果是绝对路径直接用；否则认为是相对当前工程的路径
    if not os.path.isabs(image_url):
        image_path = os.path.join(BASE_DIR, image_url.lstrip("/"))
    else:
        image_path = image_url
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    # OpenCV BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, image_path


def _get_pixel_rgb(image: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w, _ = image.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        raise IndexError("pixel out of range")
    return image[y, x, :]


def _apply_segmentation_boundaries(image: np.ndarray, labels: np.ndarray, color: Tuple[int, int, int] = (128, 128, 128)):
    """在图像上叠加分割线（灰色虚线效果）。"""
    # 找出边界 (boolean mask)
    boundaries = find_boundaries(labels, mode='thick')
    
    # 生成虚线掩码：利用坐标奇偶性模拟
    # 这里使用 (x + y) % 4 < 2 来生成类似虚线的效果
    h, w = labels.shape
    y_grid, x_grid = np.indices((h, w))
    dashed_mask = ((x_grid + y_grid) % 4 < 2)
    
    # 最终掩码：是边界 且 符合虚线规则
    final_mask = boundaries & dashed_mask
    
    # 应用颜色
    image[final_mask] = color
    return image


def _create_masked_image(image: np.ndarray, x: int, y: int, radius: int = 10, labels: Optional[np.ndarray] = None) -> np.ndarray:
    """生成只有点击区域保留颜色、其他区域置为黑色的图。
    
    如果提供了 labels，则会叠加灰色虚线分割线。
    """
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
    masked = np.zeros_like(image)
    masked[mask == 255] = image[mask == 255]
    
    if labels is not None:
        _apply_segmentation_boundaries(masked, labels)
        
    return masked


@app.post("/api/painting/register-original")
async def register_original(req: RegisterOriginalRequest):
    # 简化实现：不真正下载图片，只记录其 URL 和尺寸
    image_id = f"img_{uuid.uuid4().hex[:8]}"
    IMAGE_REGISTRY[image_id] = {
        "imageUrl": req.imageUrl,
        "width": req.width,
        "height": req.height,
    }
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "imageId": image_id,
            "normalizedUrl": req.imageUrl,
        },
    }


@app.post("/api/painting/show-original")
async def show_original(req: RegisterOriginalRequest):
    """打开绘画界面时调用：
    - 记录原图信息
    - 返回给前端一个 projectorImageUrl，用于让投影仪直接展示整张原图
    """
    image_id = f"img_{uuid.uuid4().hex[:8]}"
    # 解析并归一化 imageUrl：
    # - registry 中存相对 static 的路径，便于本地读取；
    # - 返回给前端的是完整 HTTP URL。

    raw_url = req.imageUrl

    # 处理完整 HTTP URL：尝试去掉 BACKEND_BASE_URL 前缀，得到相对路径
    if raw_url.startswith("http://") or raw_url.startswith("https://"):
        prefix = BACKEND_BASE_URL.rstrip("/") + "/"
        if raw_url.startswith(prefix):
            # 例如 raw_url = http://host:8000/static/originals/xxx.jpg
            # 得到 rel_static_path = static/originals/xxx.jpg
            rel_static_path = raw_url[len(prefix):]
        else:
            # 域名与当前 BACKEND_BASE_URL 不一致，暂时直接保留原始 URL 作为 "相对路径"
            rel_static_path = raw_url.lstrip("/")
        projector_url = raw_url
    else:
        # 非 http(s) 的情况：
        # 如果是绝对文件系统路径，认为是不合法的入参
        if os.path.isabs(raw_url):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": 1003,
                    "message": "imageUrl must be a HTTP url or relative static path",
                    "data": None,
                },
            )

        rel_static_path = raw_url.lstrip("/")
        projector_url = _build_public_url_from_static_path(rel_static_path)

    # 在 registry 中存的是相对 static 的路径，用于后续 color-mix 读取本地文件
    IMAGE_REGISTRY[image_id] = {
        "imageUrl": rel_static_path,
        "width": req.width,
        "height": req.height,
    }

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "imageId": image_id,
            "projectorImageUrl": projector_url,
            "normalizedUrl": projector_url,
        },
    }


def _build_mix_plan(target_rgb: np.ndarray):
    """调用已有算法生成调色方案与分步数据。此处 palette_source 需替换为你的实际颜料库。"""
    # 示例：使用本地 JSON 调色盘文件，如不存在则返回空方案
    palette_path = os.path.join(BASE_DIR, "my_palette.json")
    if not os.path.exists(palette_path):
        return None, None, None

    top_colors, weights = suggest_mix(target_rgb.tolist(), palette_path)
    if not top_colors or weights is None:
        return None, None, None

    # 生成 mixPlan（百分比）
    w = weights / weights.sum()
    mix_plan = []
    for (name, rgb), ratio in zip(top_colors, w):
        r, g, b = rgb
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        mix_plan.append({
            "name": name,
            "color": hex_color,
            "ratio": float(round(ratio * 100, 2)),
        })

    # 分步调色数据
    steps = generate_steps_from_mix(top_colors, weights)

    # 目标颜色 hex
    tr, tg, tb = target_rgb.astype(int).tolist()
    target_hex = f"#{tr:02x}{tg:02x}{tb:02x}"

    return target_hex, mix_plan, steps


def _clear_processed_dir():
    """清空 processed 目录下的已有文件，避免累积旧的处理结果。"""
    if not os.path.exists(PROCESSED_DIR):
        return
    for entry in os.listdir(PROCESSED_DIR):
        path = os.path.join(PROCESSED_DIR, entry)
        if os.path.isfile(path) or os.path.islink(path):
            try:
                os.remove(path)
            except OSError:
                pass
        elif os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except OSError:
                pass


def _clear_directory_except(directory: str, keep_files: List[str]):
    """清空指定目录下除了 keep_files 以外的所有文件。"""
    if not os.path.exists(directory):
        return
    
    # 规范化 keep_files 路径，只保留文件名
    keep_filenames = {os.path.basename(f) for f in keep_files}
    
    for entry in os.listdir(directory):
        if entry in keep_filenames:
            continue
            
        path = os.path.join(directory, entry)
        if os.path.isfile(path) or os.path.islink(path):
            try:
                os.remove(path)
            except OSError:
                pass
        elif os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except OSError:
                pass


def _save_processed_image(masked_image: np.ndarray, image_id: str, x: int, y: int) -> str:
    # 每次生成新处理图前，先清空旧的 processed 内容
    # _clear_processed_dir()

    filename = f"{image_id}_{x}_{y}.png"
    save_path = os.path.join(PROCESSED_DIR, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # RGB -> BGR for OpenCV
    bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr)
    # 返回完整的可访问 URL，前端和投影仪可以直接使用
    rel_path = os.path.relpath(save_path, BASE_DIR).replace("\\", "/")
    # 确保不重复斜杠
    return _build_public_url_from_static_path(rel_path)


@app.post("/api/color-mix/from-click")
async def color_mix_from_click_by_id(req: ColorMixFromClickByIdRequest):
    info = IMAGE_REGISTRY.get(req.imageId)
    if not info:
        raise HTTPException(status_code=404, detail={"code": 1001, "message": "image not found", "data": None})

    # 1. 确定要处理的图片源和取色逻辑
    # 默认情况：从原图取色，生成圆形遮罩
    target_rgb = None
    masked_image = None
    projector_url = None
    
    # 尝试解析原图文件名，以便寻找对应的分割数据
    # info["imageUrl"] 类似 "static/originals/origin_xxx.png"
    original_rel_path = info["imageUrl"]
    original_filename = os.path.basename(original_rel_path)
    filename_base = os.path.splitext(original_filename)[0]

    if req.layer == "segment_large" or req.layer == "segment_small":
        # --- 分割模式 ---
        suffix = "large" if req.layer == "segment_large" else "small"
        pkl_path = os.path.join(SEGMENTED_DATA_DIR, f"{filename_base}_{suffix}.pkl")
        
        if not os.path.exists(pkl_path):
             # Fallback to original if segmentation data is missing (e.g. old images)
             # This prevents errors for images uploaded before this feature was added
             print(f"Warning: Segmentation data not found at {pkl_path}, falling back to original.")
             masked_image = None
        else:
            try:
                with open(pkl_path, "rb") as f:
                    seg_data = pickle.load(f)
                labels = seg_data["labels"]
                palette = seg_data["palette"]
                
                h, w = labels.shape
                if req.pixel.x < 0 or req.pixel.x >= w or req.pixel.y < 0 or req.pixel.y >= h:
                     raise HTTPException(status_code=400, detail={"code": 1002, "message": "pixel out of range", "data": None})

                label_id = labels[req.pixel.y, req.pixel.x]
                
                # 如果点击的是背景(0)或者无效区域
                if label_id not in palette:
                     # Fallback or error? Let's error to be clear
                     raise HTTPException(status_code=400, detail={"code": 1006, "message": "clicked on invalid segment (background)", "data": None})

                target_rgb = np.array(palette[label_id])
                
                if req.layer == "segment_small":
                    masked_image = np.zeros((h, w, 3), dtype=np.uint8)  # 生成黑色背景图
                else:
                    # masked_image = np.zeros((h, w, 3), dtype=np.uint8)
                    # mask = (labels == label_id)
                    # masked_image[mask] = target_rgb
                    # # 叠加分割线
                    # _apply_segmentation_boundaries(masked_image, labels)

                    masked_image = np.zeros((h, w, 3), dtype=np.uint8)

            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                # Log error and fallback
                print(f"Error loading segmentation data: {e}")
                masked_image = None

    else:
        # --- 原图模式 (默认) ---
        pass

    if masked_image is None and projector_url is None:
        # Fallback logic: load original image and create circle mask
        try:
            img, image_path = _load_image_from_url_or_path(info["imageUrl"])
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail={"code": 1001, "message": "image not found", "data": None})

        try:
            target_rgb = _get_pixel_rgb(img, req.pixel.x, req.pixel.y)
        except IndexError:
            raise HTTPException(status_code=400, detail={"code": 1002, "message": "pixel out of range", "data": None})

        # 原图模式下，返回线稿URL而不是原图URL（线稿固定为 static/line_art/0.png）
        filename_lineart = "0.png"
        path_lineart = os.path.join(LINE_ART_DIR, filename_lineart)
        # 2. 检查线稿是否存在，不存在则生成
        if not os.path.exists(path_lineart):
            line_art_img = line_art(img)
            cv2.imwrite(path_lineart, line_art_img)
        # 3. 返回线稿URL
        rel_path = os.path.relpath(path_lineart, BASE_DIR).replace("\\", "/")
        projector_url = _build_public_url_from_static_path(rel_path)

    # 2. 计算调色配方
    target_hex, mix_plan, steps = _build_mix_plan(target_rgb)
    if mix_plan is None:
        raise HTTPException(status_code=500, detail={"code": 2001, "message": "mix plan failed", "data": None})

    # 3. 保存掩膜图并返回 (如果 projector_url 还没定下来，说明是分割模式生成的 masked_image)
    if projector_url is None:
        projector_url = _save_processed_image(masked_image, req.imageId, req.pixel.x, req.pixel.y)

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "targetColor": {
                "hex": target_hex,
                "name": "",  # 暂无颜色命名，可后续扩展色名表
            },
            "mixPlan": mix_plan,
            "steps": steps,
            "projectorImageUrl": projector_url,
        },
    }


@app.post("/api/painting/line-art")
async def generate_line_art(req: LineArtRequest):
    try:
        # 线稿固定为 static/line_art/0.png
        filename_lineart = "0.png"
        path_lineart = os.path.join(LINE_ART_DIR, filename_lineart)

        # 2. 检查文件是否存在
        if not os.path.exists(path_lineart):
            # 如果不存在（可能是旧图片），则现场生成一次
            # Fallback logic
            img, _ = _load_image_from_url_or_path(req.imageUrl)
            line_art_img = line_art(img)
            cv2.imwrite(path_lineart, line_art_img)

        # 3. 返回原图 URL（保持字段名 lineArtUrl，值为原图地址）
        if req.imageUrl.startswith("http://") or req.imageUrl.startswith("https://"):
            original_url = req.imageUrl
        else:
            rel_original = req.imageUrl.lstrip("/")
            original_url = _build_public_url_from_static_path(rel_original)

        return {
            "code": 0,
            "message": "ok",
            "data": {
                "lineArtUrl": original_url
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": 3001, "message": f"failed to get line art: {str(e)}", "data": None})


def _reconstruct_image_from_labels(labels: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    """根据分割的 labels 和 palette 重建 RGB 图像。"""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for lab, color in palette.items():
        # color is (r, g, b)
        out[labels == lab] = color
    return out


if __name__ == "__main__":
    # Allow overriding port via environment variable PORT to avoid collisions
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
