# 后端调色服务 API 规范

本文件说明 **App ⇄ 调色后端** 的 HTTP API 约定，方便与前端、投影端配合。

## 通用约定

- 协议：HTTP/HTTPS + JSON
- 字符编码：UTF-8
- Base URL 例：`https://your-backend.example.com`
- 所有响应建议使用统一包装：

```json
{
  "code": 0,
  "message": "ok",
  "data": { /* 具体数据 */ }
}
```

- `code = 0` 表示成功；非 0 表示失败。
- 前端会直接使用 `data` 字段中的内容。

---

## 0. 原图文件上传接口（App 从相册/相机选图后调用）

> 目的：前端从本地相册/相机选择图片后，将**原始文件**上传到调色后端，后端返回一个自己可直接访问的 HTTP 图片地址，供后续所有接口使用。

### URL

- `POST /api/painting/upload-original-file`

### 请求方式

- `Content-Type: multipart/form-data`
- 表单字段：

| 字段名 | 类型   | 是否必填 | 说明                             |
|--------|--------|----------|----------------------------------|
| file   | file   | 是       | 图片文件本身（原图），单个文件 |

### Response Body

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "imageUrl": "http://your-backend/static/originals/origin_001.png"
  }
}
```

字段说明：

- `imageUrl` `string`：
  - 后端保存后的**完整可访问 URL**（推荐 `http://` 或 `https://` 开头）；
  - 前端后续会：
    - 在进入绘画页时，将此 URL 作为 `imageUrl` 传给 `show-original`；
    - 在未拿到 `imageId` 时，将此 URL 作为 `imageUrl` 传给 `from-click-by-url` 接口。

> 约定：前端不会把本地路径（如 `blob:...`、`file://...`）传给后续接口，**所有后续接口都只使用这里返回的 `imageUrl`**。

---

## 1. （可选）原图登记接口

> 目的：前端进入绘画页时，将原图信息告知后端，后端返回一个 `imageId`，后续点击取色统一使用此 ID。

### URL

- `POST /api/painting/register-original`

### Request Body

```json
{
  "imageUrl": "https://your-backend/static/originals/origin_001.png",
  "width": 1920,
  "height": 1080
}
```

字段说明：

- `imageUrl` `string`：前端绘画页使用的原图 URL。
- `width` `number`：原图像素宽度（前端在图片加载完成后传入）。
- `height` `number`：原图像素高度。

### Response Body

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "imageId": "img_001",
    "normalizedUrl": "https://your-backend/static/originals/origin_001.png"
  }
}
```

字段说明：

- `imageId` `string`：后端为该原图生成的唯一 ID，后续点击取色接口使用。
- `normalizedUrl` `string`：标准化后的图片 URL（可与入参相同）。

> 注：如果后端不需要登记原图，也可以跳过本接口，改为在下文接口中直接使用 `imageUrl` 字段。

---

## 2. 原图展示 + 原图投影接口

> 目的：前端进入绘画页时，将原图信息告知后端，后端生成 `imageId` 并返回一张适合直接投影显示的整图 `projectorImageUrl`。

### URL

- `POST /api/painting/show-original`

### Request Body

```json
{
  "imageUrl": "http://your-backend/static/originals/origin_001.png",
  "width": 1920,
  "height": 1080
}
```

字段说明：

- `imageUrl` `string`：
  - 来自第 0 步上传接口返回的 URL；
  - 或其他后端可直接访问的 HTTP/HTTPS 图片地址。
- `width` / `height` `number`：
  - 原图像素尺寸（前端在图片加载完成后传入，用于边界检查、调试）。

### Response Body

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "imageId": "img_001",
    "projectorImageUrl": "http://your-backend/static/originals/origin_001_projector.png",
    "normalizedUrl": "http://your-backend/static/originals/origin_001.png"
  }
}
```

字段说明：

- `imageId` `string`：后端为该原图生成的唯一 ID，后续点击取色接口使用；
- `projectorImageUrl` `string`：
  - 可供投影仪直接显示的**整图 URL**；
  - 前端会原样转发给投影仪的 `/show-image` 接口；
  - 若不需要特殊处理，也可以与 `normalizedUrl` 相同。
- `normalizedUrl` `string`：
  - 标准化后的原图 URL，用于回显或调试；
  - 当前前端实现会在 `projectorImageUrl` 为空时回退使用此字段，再回退到 `imageUrl`。

调用时序（与前端约定）：

1. 前端从相册/相机选图 → 通过 `/api/painting/upload-original-file` 上传，拿到 `imageUrl`；
2. 进入绘画页后，前端在图片加载完成时调用 `POST /api/painting/show-original`；
3. 后端返回 `imageId` + `projectorImageUrl`；
4. 前端：
   - 保存 `imageId`；
   - 立即调用投影仪 `/show-image`，把 `projectorImageUrl` 原样转发过去，让投影仪投整张原图。

---

## 3. 点击取色 + 调色建议接口（核心）

> 目的：前端在用户点击画布时，将点击位置（在原图中的像素坐标）发送给后端，后端返回：
>
> - 该位置的颜色信息；
> - 调色方案；
> - 一张 **已处理好的新图片 URL**，供前端转发给投影仪。

### URL

- `POST /api/color-mix/from-click`

### Request Body

**方案 A：使用 `imageId` 标识原图（推荐）**

```json
{
  "imageId": "img_001",
  "pixel": {
    "x": 530,
    "y": 240
  },
  "ratio": {
    "x": 0.27,
    "y": 0.35
  },
  "layer": "segment_large"
}
```

字段说明：

- `imageId` `string`：原图登记时返回的 ID。
- `pixel` `object`：点击点在 **原图像素坐标系** 下的位置：
  - `pixel.x` `number`：0 ≤ x < width。
  - `pixel.y` `number`：0 ≤ y < height。
- `ratio` `object`（可选）：点击点在原图上的相对位置，便于后端做容错或其他算法：
  - `ratio.x` `number`：0.0 ~ 1.0。
  - `ratio.y` `number`：0.0 ~ 1.0。
- `layer` `string`（可选）：指定取色和分割的层级，默认为 `segment_large`。
  - `segment_large`：大色块模式，适合起稿和铺大色（默认）；
  - `segment_small`：小色块模式，适合深入刻画和丰富细节；
  - `original`：原图模式，不使用分割逻辑，直接取像素点颜色。

**方案 B：直接使用 `imageUrl`（如果无需登记）**

```json
{
  "imageUrl": "https://your-backend/static/originals/origin_001.png",
  "pixel": { "x": 530, "y": 240 }
}
```

后端可按需支持其中一种或两种方式，与前端约定即可。当前前端实现：

- 正常情况下：进入绘画页时会先调用 `/api/painting/show-original` 获取 `imageId`，点击取色时优先使用 `imageId` 方案；
- 仅在调色后端明确支持且协商好的前提下，才会退回使用 `imageUrl` 方案（同样使用第 0 步上传接口返回的 URL）。

### Response Body

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "targetColor": {
      "hex": "#FF6B6B",
      "name": "珊瑚红"
    },
    "mixPlan": [
      { "name": "钛白", "color": "#FFFFFF", "ratio": 40 },
      { "name": "镉红", "color": "#DC143C", "ratio": 35 },
      { "name": "镉黄", "color": "#FFD700", "ratio": 20 },
      { "name": "象牙黑", "color": "#000000", "ratio": 5 }
    ],
    "projectorImageUrl": "https://your-backend/processed/img_001_530_240.png",
    "maskImageUrl": "https://your-backend/masks/img_001_530_240_mask.png"
  }
}
```

字段说明：

- `targetColor` `object`：点击位置的目标颜色信息：
  - `hex` `string`：颜色十六进制值（前端会直接用于背景色展示）。
  - `name` `string`：颜色名称（中文名，如“珊瑚红”）。
- `mixPlan` `array<object>`：调色方案列表，每一项对应一种颜料：
  - `name` `string`：颜料名称（如“钛白”、“镉红”）。
  - `color` `string`：该颜料代表色的十六进制值，用于前端渲染色条。
  - `ratio` `number`：该颜料在配方中的比例，单位为百分比（0~100）。
- `projectorImageUrl` `string`：
  - 一张**已经处理好的新图片**的 HTTP URL（例如在点击区域做了高亮或其他效果）；
  - **视觉效果**：
    - 背景为全黑；
    - 仅保留当前点击所在的色块颜色；
    - **叠加全图的灰色虚线网格**，标示出所有色块的边界，便于用户定位。
  - 前端不会再加工，只会把这个 URL 通过 HTTP 转发给投影仪；
  - 要求投影仪所在网络环境可以直接访问该 URL（局域网或公网均可）。
- `maskImageUrl` `string`（可选）：
  - 一张**已经生成好的掩膜/高亮图**的 HTTP URL；
  - 前端同样不会再加工，而是将该 URL 通过 HTTP 转发给投影仪，由投影仪负责如何叠加或呈现掩膜效果；
  - 要求与 `projectorImageUrl` 一样可被投影仪访问（通常与调色后端同一域名/IP）。

### 使用约定

前端在点击某一点后，会按以下顺序调用：

1. 将点击位置换算为原图像素坐标 `pixel.x / pixel.y`；
2. 调用 `POST /api/color-mix/from-click`，传入 `imageId` 与 `pixel`；
3. 根据返回的 `targetColor`、`mixPlan` 更新 App 内的颜色分析与调色建议 UI；
4. 取出 `projectorImageUrl`，**直接转发给投影仪的 HTTP 接口**（见投影仪规范文档），由投影仪实际拉取并显示该图片。

---

## 3. 错误码与异常处理建议

- 推荐错误码示例：

```json
{
  "code": 1001,
  "message": "image not found",
  "data": null
}
```

- 常见错误场景：
  - `1001`：`imageId` 或 `imageUrl` 不存在或已过期；
  - `1002`：`pixel` 越界（前端传入的坐标超出图片范围）；
  - `2001`：内部处理错误（颜色分析或调色算法异常）；
  - `2002`：生成处理后图片失败。

前端在收到 `code != 0` 时，会根据 `message` 做统一的 Toast 提示或降级处理；`projectorImageUrl` 为空时，不会尝试通知投影仪。

---

## 4. 线稿生成接口（辅助起形）

> 目的：生成素描或轮廓风格的线稿，辅助用户在画布上起形。

### URL

- `POST /api/painting/line-art`

### Request Body

```json
{
  "imageUrl": "static/originals/origin_001.png",
  "mode": "sketch"
}
```

字段说明：

- `imageUrl` `string`：原图的 URL（推荐使用 `upload-original-file` 返回的地址）。
- `mode` `string`（可选）：
  - `"sketch"`：素描风格（默认），保留明暗关系，适合观察素描调子；
  - `"outline"`：轮廓风格，只有线条，适合投影描边。

### Response Body

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "lineArtUrl": "http://your-backend/static/processed/origin_001_sketch.png"
  }
}
```

字段说明：

- `lineArtUrl` `string`：生成的线稿图片 URL，前端可直接传给投影仪显示。

---

## 5. 面向后端同事的实现指引与进度说明

本节是专门给后端同事看的，从“当前前端行为 + 后端需要做什么”的角度，梳理一遍实现重点，方便排期和联调。

### 5.1 当前前端整体行为总结

- 严格模式：
  - App 只在 `upload-original-file` + `show-original` 都成功、且拿到 `imageId` 之后，才允许：
    - 把原图投到投影仪上；
    - 响应用户点击进行取色与调色。
  - 任何一步失败（例如后端 500 / 超时），前端会提示错误，不再偷偷用本地图片 URL 兜底投影。

- 点击取色链路：
  - 用户每点击一次画面，前端会：
    1. 根据点击位置计算原图像素坐标 `pixel.x / pixel.y`；
    2. 计算相对坐标 `ratio.x / ratio.y`（0~1，便于后端容错）；
    3. 调用 `/api/color-mix/from-click`，优先携带 `imageId`；
    4. 用响应中的 `targetColor` / `mixPlan` 更新 App 内调色建议面板；
    5. 依次使用 `projectorImageUrl`、`maskImageUrl` 调用投影仪 HTTP 接口 `/show-image`，由投影仪实际显示结果图/掩膜图。

- 投影仪交互：
  - App 不做任何图像合成和掩膜计算，只负责：
    - 把后端返回的 `projectorImageUrl`、`maskImageUrl` 原样转发给投影仪；
    - 保证 URL 是完整的 HTTP/HTTPS 地址，可被投影仪所在网络访问。

### 5.2 各接口当前状态 & 后端待办

> 说明中的“已接入前端”指：前端已经按本文件约定发请求，并基于响应结构完成 UI 行为；“待后端实现/完善”指：仍需后端按协议补齐逻辑。

#### 0. `POST /api/painting/upload-original-file`（必做）

- 前端状态：
  - 已接入前端，用于从相册/相机选图后上传原始文件；
  - 前端严格依赖本接口返回的 `imageUrl` 作为后续所有接口的图片地址来源。

- 后端需要保证：
  - 接收 `multipart/form-data`，字段名为 `file`；
  - 将原图存储到后端可控的文件系统或对象存储中；
  - 返回的 `data.imageUrl`：
    - 必须是完整的 HTTP/HTTPS URL（如 `http://172.16.25.165:8000/static/...`）；
    - 在调色后端进程和投影仪所在网络都可直接访问；
  - 建议为静态资源目录配置合理的缓存、访问控制和 CORS（若 App 直接访问）。

#### 2. `POST /api/painting/show-original`（推荐，当前前端已使用）

- 前端状态：
  - 已接入前端，进入绘画页并完成原图加载后自动调用；
  - 成功后会：
    - 记录 `imageId`，作为后续 `/api/color-mix/from-click` 的必备参数；
    - 使用 `projectorImageUrl` 投整张原图到投影仪；
  - 若本接口失败，前端不会再尝试兜底投原图，也不会允许用户取色。

- 后端需要保证：
  - 能根据入参 `imageUrl`、`width`、`height` 登记/缓存原图信息，生成一个唯一 `imageId`；
  - 返回：
    - `data.imageId`：字符串 ID，后续所有点击接口依赖此 ID；
    - `data.projectorImageUrl`：
      - 可直接用于投影显示的整图 URL；
      - 若暂无特殊处理，可与规范中的 `normalizedUrl` 或入参 `imageUrl` 相同；
    - `data.normalizedUrl`：标准化后的原图地址，用于调试和回显；
  - 建议做的校验：
    - 检查 `imageUrl` 是否可访问、格式是否受支持；
    - 如发现异常，返回非 0 `code` 和明确的 `message`（例如 `image not found`）。

#### 3. `POST /api/color-mix/from-click`（核心，当前前端已按协议调用）

- 前端状态：
  - 已接入前端，每次点击画面都会调用本接口；
  - 默认使用“方案 A：`imageId + pixel + ratio`”，仅在双方协商后才会用 `imageUrl` 方案；
  - 收到成功响应后，会：
    - 使用 `targetColor` / `mixPlan` 渲染调色建议面板；
    - 先后将 `projectorImageUrl`、`maskImageUrl` 发送给投影仪 `/show-image` 接口。

- 后端需要保证（最小可用版本）：
  - 能根据 `imageId` 找到对应原图（或其内部表示）；
  - 校验 `pixel` 是否在图片范围内（越界时返回错误码 1002）；
  - 至少返回：
    - 一个有效的 `targetColor.hex`（例如点击像素的平均色）和 `name`（可先用占位名，如“颜色 A”）；
    - 一个简单但结构正确的 `mixPlan` 数组（比例总和建议约等于 100）；
    - 一个有效的 `projectorImageUrl`，可以在后端先返回原图或简单处理后的图；
  - 在算法尚未完全就绪时，可以先返回“假数据/占位图”，优先打通链路。

- 后端进一步优化空间（算法成熟后）：
  - 使用 `ratio` 参数进行坐标容错（应对前端缩放、裁剪等差异）；
  - 真正实现调色算法，生成可靠的 `mixPlan`；
  - 生成带有高亮/描边等效果的 `projectorImageUrl`；
  - 如有需要，生成单独的 `maskImageUrl`，用于投影端做特殊显示。

#### 4. `POST /api/painting/line-art`（可选）

- 前端状态：
  - 已接入前端，辅助用户在画布上起形；
  - 成功后会将线稿图直接传给投影仪显示。

- 后端需要保证：
  - 能根据入参 `imageUrl` 生成对应的线稿图；
  - 返回：
    - `data.lineArtUrl`：生成的线稿图 URL，前端可直接用于投影仪；
  - 支持的线稿风格：
    - `sketch`：素描风格，保留明暗关系；
    - `outline`：轮廓风格，仅保留线条。

### 5.3 掩膜图 `maskImageUrl` 的推荐用法

- 生成位置：
  - 完全由后端决定是否生成掩膜图以及其视觉样式（透明背景、半透明覆盖、描边等）；
  - 若不需要掩膜，可不返回 `maskImageUrl` 字段或设为 `null`。

- 前端与投影端职责划分：
  - 前端：
    - 不在 App 画面上叠加掩膜；
    - 仅在收到 `maskImageUrl` 时，将其作为新的图片 URL 转发给投影仪 `/show-image`；
  - 投影仪：
    - 负责如何展示这张图（例如直接显示“原图+掩膜已合成好的整图”，或单独显示掩膜图）。

- 实现建议：
  - 一种简单做法是在后端直接生成“原图+掩膜已合成好的整图”，把它作为 `projectorImageUrl` 返回；
  - 若需要更复杂的分层效果，可以：
    - `projectorImageUrl` 返回整图；
    - `maskImageUrl` 返回只包含掩膜的图，由投影仪端负责二者的叠加方式。

### 5.4 联调建议与示例请求

- 建议顺序：先打通“上传 → show-original → 投影原图”，再实现“from-click → 调色面板 + 投影处理图/掩膜图”。

- 示例：`/api/painting/show-original`（当前联调环境的伪示例）

```bash
curl -X POST "http://<BACKEND_BASE_URL>/api/painting/show-original" \
  -H "Content-Type: application/json" \
  -d '{
    "imageUrl": "http://<BACKEND_BASE_URL>/static/originals/origin_001.png",
    "width": 1920,
    "height": 1080
  }'
```

- 示例：`/api/color-mix/from-click`

```bash
curl -X POST "http://<BACKEND_BASE_URL>/api/color-mix/from-click" \
  -H "Content-Type: application/json" \
  -d '{
    "imageId": "img_001",
    "pixel": { "x": 530, "y": 240 },
    "ratio": { "x": 0.27, "y": 0.35 }
  }'
```

- 联调时建议：
  - 后端在日志中打印收到的 `imageId`、`pixel`、`ratio`，方便与前端点击位置核对；
  - 先用简单/固定的调色方案和占位图片 URL 返回，验证前端 UI 和投影仪能否正常反应；
  - 确认所有返回的 URL（尤其是 `projectorImageUrl`、`maskImageUrl`）在投影仪网络中可以访问。