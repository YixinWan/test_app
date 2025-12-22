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

from domain import suggest_mix, generate_steps_from_mix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(BASE_DIR, "static", "originals")
PROCESSED_DIR = os.path.join(BASE_DIR, "static", "processed")

# 后端对外访问的基础 URL，需要与你前端配置的 BACKEND_BASE_URL 保持一致
# 示例：前端配置为 export const BACKEND_BASE_URL = 'http://172.16.25.51:8000'
# 这里就写成相同的地址，便于返回完整的 projectorImageUrl
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://172.16.25.165:8000")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

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


class ColorMixFromClickByUrlRequest(BaseModel):
    imageUrl: str
    pixel: PixelCoord


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

    return {
        "code": 0,
        "message": "ok",
        "data": {
            "imageUrl": image_url,
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
    IMAGE_REGISTRY[image_id] = {
        "imageUrl": req.imageUrl,
        "width": req.width,
        "height": req.height,
    }

    # 构造整张原图的完整 URL（假设 imageUrl 指向 static/originals/...）
    # 如果是绝对路径，就提示前端改成相对 static 路径或 HTTP URL
    if os.path.isabs(req.imageUrl):
        raise HTTPException(
            status_code=400,
            detail={
                "code": 1003,
                "message": "imageUrl must be a HTTP path or relative static path",
                "data": None,
            },
        )

    # 去掉开头的斜杠，拼成完整的可访问 URL
    rel_path = req.imageUrl.lstrip("/")
    projector_url = _build_public_url_from_static_path(rel_path)

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


def _save_processed_image(masked_image: np.ndarray, image_id: str, x: int, y: int) -> str:
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

    try:
        img, image_path = _load_image_from_url_or_path(info["imageUrl"])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail={"code": 1001, "message": "image not found", "data": None})

    try:
        target_rgb = _get_pixel_rgb(img, req.pixel.x, req.pixel.y)
    except IndexError:
        raise HTTPException(status_code=400, detail={"code": 1002, "message": "pixel out of range", "data": None})

    target_hex, mix_plan, steps = _build_mix_plan(target_rgb)
    if mix_plan is None:
        raise HTTPException(status_code=500, detail={"code": 2001, "message": "mix plan failed", "data": None})

    masked = _create_masked_image(img, req.pixel.x, req.pixel.y, radius=10)
    projector_url = _save_processed_image(masked, req.imageId, req.pixel.x, req.pixel.y)

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
    try:
        img, image_path = _load_image_from_url_or_path(req.imageUrl)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail={"code": 1001, "message": "image not found", "data": None})

    try:
        target_rgb = _get_pixel_rgb(img, req.pixel.x, req.pixel.y)
    except IndexError:
        raise HTTPException(status_code=400, detail={"code": 1002, "message": "pixel out of range", "data": None})

    # 构造一个临时 imageId 方便生成文件名
    image_id = f"tmp_{uuid.uuid4().hex[:8]}"

    target_hex, mix_plan, steps = _build_mix_plan(target_rgb)
    if mix_plan is None:
        raise HTTPException(status_code=500, detail={"code": 2001, "message": "mix plan failed", "data": None})

    masked = _create_masked_image(img, req.pixel.x, req.pixel.y, radius=10)
    projector_url = _save_processed_image(masked, image_id, req.pixel.x, req.pixel.y)

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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
