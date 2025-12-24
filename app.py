from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import uvicorn
import os
import uuid
import cv2
import numpy as np
import shutil
import pickle

from domain import suggest_mix, generate_steps_from_mix
from domain.segmentation import coarse_color_blocks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(BASE_DIR, "static", "originals")
PROCESSED_DIR = os.path.join(BASE_DIR, "static", "processed")
SEGMENTED_DIR = os.path.join(BASE_DIR, "static", "segmented")
SEGMENTED_DATA_DIR = os.path.join(BASE_DIR, "static", "segmented_data")

# 后端对外访问的基础 URL，需要与你前端配置的 BACKEND_BASE_URL 保持一致
# 示例：前端配置为 export const BACKEND_BASE_URL = 'http://172.16.25.51:8000'
# 这里就写成相同的地址，便于返回完整的 projectorImageUrl
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://172.16.25.165:8000")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(SEGMENTED_DATA_DIR, exist_ok=True)

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



@app.post("/api/painting/upload-original-file")
async def upload_original_file(file: UploadFile = File(...)):
    """上传原始图片文件，保存到 static/originals 下并返回可访问的 imageUrl。

    对齐 BACKEND_API_README.md 中的 /api/painting/upload-original-file 规范：
    - 入参：multipart/form-data, 字段名为 file
    - 出参：data.imageUrl 为后端可直接访问的完整 HTTP URL
    """
    # 简单按随机前缀避免重名覆盖
    ext = os.path.splitext(file.filename or "")[1] or ".png"
    filename = f"origin_{uuid.uuid4().hex[:8]}{ext}"
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

    # --- 生成分割图 (大色块 & 小色块) ---
    # 读取图片 (OpenCV BGR -> RGB)
    img_bgr = cv2.imread(save_path)
    if img_bgr is None:
        # 理论上不应该发生，除非文件写入失败
        raise HTTPException(status_code=500, detail="Failed to read saved image for segmentation")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. 生成大色块图 (Coarse / Large Segments)
    # 使用较少的 segments 和较大的 absorb_area_thresh
    labels_large, palette_large = coarse_color_blocks(
        img_rgb,
        slic_segments=50,       # 分割数少
        absorb_area_thresh=10  # 小区域合并阈值大
    )
    img_large = _reconstruct_image_from_labels(labels_large, palette_large)
    
    filename_base = os.path.splitext(filename)[0]
    filename_large = f"{filename_base}_large.png"
    path_large = os.path.join(SEGMENTED_DIR, filename_large)
    # 保存为 BGR
    cv2.imwrite(path_large, cv2.cvtColor(img_large, cv2.COLOR_RGB2BGR))
    url_large = _build_public_url_from_static_path(os.path.relpath(path_large, BASE_DIR).replace("\\", "/"))

    # 保存 Large 数据的 labels 和 palette
    data_large_path = os.path.join(SEGMENTED_DATA_DIR, f"{filename_base}_large.pkl")
    with open(data_large_path, "wb") as f:
        pickle.dump({"labels": labels_large, "palette": palette_large}, f)

    # 2. 生成小色块图 (Fine / Small Segments)
    # 使用较多的 segments 和较小的 absorb_area_thresh
    labels_small, palette_small = coarse_color_blocks(
        img_rgb,
        slic_segments=3000,      # 分割数多
        absorb_area_thresh=500   # 小区域合并阈值小
    )
    img_small = _reconstruct_image_from_labels(labels_small, palette_small)

    filename_small = f"{filename_base}_small.png"
    path_small = os.path.join(SEGMENTED_DIR, filename_small)
    cv2.imwrite(path_small, cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR))
    url_small = _build_public_url_from_static_path(os.path.relpath(path_small, BASE_DIR).replace("\\", "/"))

    # 保存 Small 数据的 labels 和 palette
    data_small_path = os.path.join(SEGMENTED_DATA_DIR, f"{filename_base}_small.pkl")
    with open(data_small_path, "wb") as f:
        pickle.dump({"labels": labels_small, "palette": palette_small}, f)

    # --- 清理旧数据 ---
    # 只保留本次生成的4个文件：
    # 1. SEGMENTED_DIR: filename_large, filename_small
    # 2. SEGMENTED_DATA_DIR: filename_base_large.pkl, filename_base_small.pkl
    
    _clear_directory_except(SEGMENTED_DIR, [filename_large, filename_small])
    _clear_directory_except(SEGMENTED_DATA_DIR, [os.path.basename(data_large_path), os.path.basename(data_small_path)])

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "imageUrl": image_url,
            "segmentedLargeUrl": url_large,
            "segmentedSmallUrl": url_small,
        },
    }


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


def _create_masked_image(image: np.ndarray, x: int, y: int, radius: int = 10) -> np.ndarray:
    """生成只有点击区域保留颜色、其他区域置为黑色的图。"""
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
    masked = np.zeros_like(image)
    masked[mask == 255] = image[mask == 255]
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
    _clear_processed_dir()

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
                masked_image = np.zeros((h, w, 3), dtype=np.uint8)
                mask = (labels == label_id)
                masked_image[mask] = target_rgb

            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                # Log error and fallback
                print(f"Error loading segmentation data: {e}")
                masked_image = None

    else:
        # --- 原图模式 (默认) ---
        pass

    if masked_image is None:
        # Fallback logic: load original image and create circle mask
        try:
            img, image_path = _load_image_from_url_or_path(info["imageUrl"])
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail={"code": 1001, "message": "image not found", "data": None})

        try:
            target_rgb = _get_pixel_rgb(img, req.pixel.x, req.pixel.y)
        except IndexError:
            raise HTTPException(status_code=400, detail={"code": 1002, "message": "pixel out of range", "data": None})
        
        masked_image = _create_masked_image(img, req.pixel.x, req.pixel.y, radius=10)

    # 2. 计算调色配方
    target_hex, mix_plan, steps = _build_mix_plan(target_rgb)
    if mix_plan is None:
        raise HTTPException(status_code=500, detail={"code": 2001, "message": "mix plan failed", "data": None})

    # 3. 保存掩膜图并返回
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


@app.post("/api/color-mix/from-click-by-url")
async def color_mix_from_click_by_url(req: ColorMixFromClickByUrlRequest):
    # 1. 尝试加载图片
    try:
        img, image_path = _load_image_from_url_or_path(req.imageUrl)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail={"code": 1001, "message": "image not found", "data": None})

    # 2. 确定 layer 和 filename_base
    # 尝试从 URL/Path 中推断 layer
    # 假设 segmented url 格式: .../origin_xxx_large.png 或 .../origin_xxx_small.png
    filename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    # Normalize filename_base by stripping known suffixes
    if name_no_ext.endswith("_large"):
        real_base = name_no_ext[:-6]
        inferred_layer = "segment_large"
    elif name_no_ext.endswith("_small"):
        real_base = name_no_ext[:-6]
        inferred_layer = "segment_small"
    else:
        real_base = name_no_ext
        inferred_layer = "original"

    # Use requested layer if present, otherwise inferred
    layer = req.layer if req.layer else inferred_layer
    filename_base = real_base

    target_rgb = None
    masked_image = None

    if layer == "segment_large" or layer == "segment_small":
        # --- 分割模式 ---
        suffix = "large" if layer == "segment_large" else "small"
        pkl_path = os.path.join(SEGMENTED_DATA_DIR, f"{filename_base}_{suffix}.pkl")
        
        if not os.path.exists(pkl_path):
            # 既然明确识别出了是分割图模式，如果数据不存在，应该报错而不是回退
            raise HTTPException(status_code=404, detail={"code": 1004, "message": f"segmentation data pkl not found: {pkl_path}", "data": None})

        try:
            with open(pkl_path, "rb") as f:
                seg_data = pickle.load(f)
            labels = seg_data["labels"]
            palette = seg_data["palette"]
            
            h, w = labels.shape
            # 简单校验尺寸
            if h != img.shape[0] or w != img.shape[1]:
                # 尺寸不匹配，可能是缩略图？
                raise HTTPException(status_code=400, detail={"code": 1005, "message": "segmentation data dimension mismatch", "data": None})

            if req.pixel.x < 0 or req.pixel.x >= w or req.pixel.y < 0 or req.pixel.y >= h:
                raise HTTPException(status_code=400, detail={"code": 1002, "message": "pixel out of range", "data": None})

            label_id = labels[req.pixel.y, req.pixel.x]
            
            # 如果点击的是背景(0)或者无效区域
            if label_id not in palette:
                 raise HTTPException(status_code=400, detail={"code": 1006, "message": "clicked on invalid segment (background)", "data": None})

            target_rgb = np.array(palette[label_id])
            masked_image = np.zeros((h, w, 3), dtype=np.uint8)
            mask = (labels == label_id)
            masked_image[mask] = target_rgb

        except Exception as e:
            # 除非是上面主动抛出的 HTTPException，否则捕获并报 500
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail={"code": 2002, "message": f"failed to process segmentation data: {str(e)}", "data": None})
    
    # 如果 masked_image 依然是 None (说明不是分割模式)，则走默认逻辑
    if masked_image is None:
        # 如果用户显式请求了 layer 但没走进去（理论上上面会报错），这里再次检查
        if layer and layer != "original":
             # 如果 layer 不是 original 且 masked_image 为空，说明上面逻辑有漏网之鱼或者 fallback 了
             # 但我们现在不允许 fallback，所以这里应该是一个异常状态
             pass

        try:
            target_rgb = _get_pixel_rgb(img, req.pixel.x, req.pixel.y)
        except IndexError:
            raise HTTPException(status_code=400, detail={"code": 1002, "message": "pixel out of range", "data": None})
        masked_image = _create_masked_image(img, req.pixel.x, req.pixel.y, radius=10)

    # 构造一个临时 imageId 方便生成文件名
    image_id = f"tmp_{uuid.uuid4().hex[:8]}"

    target_hex, mix_plan, steps = _build_mix_plan(target_rgb)
    if mix_plan is None:
        raise HTTPException(status_code=500, detail={"code": 2001, "message": "mix plan failed", "data": None})

    projector_url = _save_processed_image(masked_image, image_id, req.pixel.x, req.pixel.y)

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


def _reconstruct_image_from_labels(labels: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    """根据分割的 labels 和 palette 重建 RGB 图像。"""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for lab, color in palette.items():
        # color is (r, g, b)
        out[labels == lab] = color
    return out


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
