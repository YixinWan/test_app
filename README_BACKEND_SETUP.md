# 油画辅助工具后端运行说明

本后端服务基于 **FastAPI + Uvicorn**，提供与前端 App 和投影仪配合使用的调色与图像处理 API。

## 1. 代码结构

- `domain/`
  - `color_mixing.py`：颜色空间转换、调色建议算法
  - `segmentation.py`：图像分割算法（SLIC、区域合并）
  - `line_art.py`：线稿生成算法
- `app.py`
  - FastAPI 后端入口，提供以下核心接口：
    - `POST /api/painting/register-original`：登记原图，返回 `imageId`
    - `POST /api/color-mix/from-click`：基于 `imageId + pixel` 计算调色建议并生成局部高亮图
    - `POST /api/color-mix/from-click-by-url`：直接基于 `imageUrl + pixel` 计算（无需提前登记）
- `BACKEND_API_README.md`
  - 与前端对接的正式 API 规范（App ⇄ 后端）
- `PROJECTOR_API_README.md`
  - 与投影仪对接的 HTTP 协议说明（App ⇄ 投影仪）

## 2. 环境准备

建议使用 Python 3.9+。

### 2.1 快速安装

#### 方式一：使用 pip（推荐轻量级部署）

项目根目录下已提供 `requirements.txt`，可直接运行：

```powershell
pip install -r requirements.txt
```

#### 方式二：使用 Conda（推荐开发环境）

如果你使用 Anaconda 或 Miniconda，可以使用 `environment.yml` 创建独立的虚拟环境，避免依赖冲突：

```powershell
# 1. 创建环境
conda env create -f environment.yml

# 2. 激活环境
conda activate painting-helper
```

### 2.2 手动安装（可选）

若需手动控制依赖版本，可参考：

```powershell
pip install fastapi uvicorn[standard] opencv-python-headless numpy scikit-image scipy
```

> **注意**：
> - 若本机无 GUI（如服务器环境），推荐使用 `opencv-python-headless`（默认配置）。
> - 若在本地开发且需要 `cv2.imshow` 等窗口功能，可将 `requirements.txt` 中的 `opencv-python-headless` 改为 `opencv-python`。

## 3. 运行后端服务

在项目目录 `test_app` 下执行：

```powershell
python app.py
```

启动后，服务默认监听：

- `http://127.0.0.1:8000`

你可以通过浏览器访问 `http://127.0.0.1:8000/docs` 查看自动生成的 Swagger API 文档，并进行联调测试。

## 4. 与前端 App 的对接方式

### 4.1 原图登记（可选）

前端在进入绘画页时，可按 `BACKEND_API_README.md`：

- 调用 `POST /api/painting/register-original`
- Body 例：

```json
{
  "imageUrl": "static/originals/origin_001.png",
  "width": 1920,
  "height": 1080
}
```

> 当前实现中，`imageUrl` 被当作**后端本地可访问的路径或相对路径**，即：
> - 若为相对路径，例如 `static/originals/origin_001.png`，会自动拼接为 `BASE_DIR/static/originals/origin_001.png`；
> - 若为绝对路径（如 `D:/images/origin_001.png`），将直接按此路径读取。

返回示例：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "imageId": "img_xxxxxx",
    "normalizedUrl": "static/originals/origin_001.png"
  }
}
```

前端在画布点击时使用该 `imageId` 即可。

### 4.2 点击取色 + 调色建议（核心）

前端在用户点击画布时：

1. 将点击点转换为原图的像素坐标 `(pixel.x, pixel.y)`；
2. 调用 `POST /api/color-mix/from-click`（推荐使用 `imageId` 方案）：

```json
{
  "imageId": "img_xxxxxx",
  "pixel": { "x": 530, "y": 240 }
}
```

3. 后端根据该像素位置：
   - 读取原图；
   - 取出该点 RGB 颜色；
   - 调用 `suggest_mix` + `generate_steps_from_mix` 生成：
     - `targetColor.hex`
     - `mixPlan`（颜料 + 比例）
     - `steps`（分步调色数据）
   - 使用 `_create_masked_image` 将除点击区域外的像素全部置黑，生成新的 PNG 图片；
   - 将该图片保存到 `static/processed/<imageId>_<x>_<y>.png`，并返回相对 URL。

返回数据结构示例：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "targetColor": {
      "hex": "#ff6b6b",
      "name": ""
    },
    "mixPlan": [
      { "name": "钛白", "color": "#ffffff", "ratio": 40.0 },
      { "name": "镉红", "color": "#dc143c", "ratio": 35.0 }
    ],
    "steps": [
      {
        "step_num": 1,
        "parts": [3, 1],
        "names": ["钛白", "镉红"],
        "rgbs": [[255, 255, 255], [220, 20, 60]],
        "mixed_hex": "#f2b5c5"
      }
    ],
    "projectorImageUrl": "/static/processed/img_xxxxxx_530_240.png"
  }
}
```

前端只需：

- 使用 `targetColor`、`mixPlan`、`steps` 更新 UI；
- 取出 `projectorImageUrl`，与投影仪的 Base URL 拼成完整地址：
  - 例如投影仪地址为 `http://192.168.1.50:9000`，则：
    - `imageUrl = http://your-backend-host:8000/static/processed/img_xxxxxx_530_240.png`
  - 按 `PROJECTOR_API_README.md` 的规范，调用投影仪 `POST /show-image`。

### 4.3 直接基于 imageUrl 方案（无需登记）

若不想使用 `imageId`，前端也可以直接调用：

- `POST /api/color-mix/from-click-by-url`

Body：

```json
{
  "imageUrl": "static/originals/origin_001.png",
  "pixel": { "x": 530, "y": 240 }
}
```

其他行为与 `/api/color-mix/from-click` 类似，只是后端内部不维护 `imageId` 映射。

## 5. 颜料调色盘配置说明

当前后端从工程根目录下的 `my_palette.json` 读取颜料库：

- 文件路径：`./my_palette.json`
- 示例结构：

```json
{
  "钛白": [255, 255, 255],
  "镉红": [220, 20, 60],
  "镉黄": [255, 215, 0],
  "象牙黑": [0, 0, 0]
}
```

你可以根据真实颜料库扩展或替换此文件：

- 若文件不存在，当前实现会返回 `mix plan failed` 错误；
- 建议在部署时务必提供一个完整的 `my_palette.json`。

## 6. 建议的后续优化方向

1. **图片存储与访问 URL**
   - 目前实现中，`imageUrl` 被当作本地文件路径使用；
   - 正式环境中建议：
     - 将原图与处理后图片统一存储到对象存储（如 OSS、S3、静态资源服务器）；
     - 后端只负责生成访问 URL（带签名 / 过期时间），不直接读写本地磁盘。
2. **颜色命名表**
   - 现在 `targetColor.name` 为空字符串；
   - 可增加一个颜色查表模块，根据 Lab/HSV 空间匹配到最近的人类可读色名（中文）。
3. **分块/区域级辅助**
   - `slic_color_blocks`、`coarse_color_blocks` 已经实现了较完整的超像素与粗分块逻辑；
   - 可进一步增加接口：
     - 一次性生成全图的铺色指导图；
     - 为每个色块生成单独的投影图片与调色方案。
4. **持久化与多用户支持**
   - 当前 `IMAGE_REGISTRY` 为进程内存字典，服务重启会丢失；
   - 正式环境建议使用 Redis / 数据库来存储 `imageId → 原图信息` 的映射。

如你需要，我可以进一步：

- 帮你补充 `my_palette.json` 示例文件；
- 或增加新的 API，例如：一次性生成某个区域的投影图与调色方案、批量处理等。
