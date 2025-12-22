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
  }
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

**方案 B：直接使用 `imageUrl`（如果无需登记）**

```json
{
  "imageUrl": "https://your-backend/static/originals/origin_001.png",
  "pixel": { "x": 530, "y": 240 }
}
```

后端可按需支持其中一种或两种方式，与前端约定即可。当前前端实现：

- 若已成功调用 `/api/painting/show-original`，会优先使用 `imageId` 方案；
- 否则退回使用 `imageUrl` 方案（同样使用第 0 步上传接口返回的 URL）。

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
    "projectorImageUrl": "https://your-backend/processed/img_001_530_240.png"
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
  - 一张**已经处理好的新图片**的 HTTP URL；
  - 前端不会再加工，只会把这个 URL 通过 HTTP 转发给投影仪；
  - 要求投影仪所在网络环境可以直接访问该 URL（局域网或公网均可）。

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

## 4. 对投影仪侧的要求（供参考）

- `projectorImageUrl` 必须是投影仪能访问到的 HTTP/HTTPS 资源：
  - 若在局域网：建议使用局域网 IP 或域名；
  - 若在公网：确保投影仪可访问外网。
- 如果有权限控制，建议使用 **带签名的短时效 URL**，避免投影仪还要做复杂认证流程。
- 图片格式建议使用常见的 `jpg/png/webp`，分辨率不宜远大于投影仪本身分辨率。

### 与“先投原图”流程的关系（补充说明）

- App 在完成投影仪扫码绑定后（获取到 `deviceBaseUrl`），进入绘画页时会先将**原图本身**发送给投影仪显示：
  - 这一步不经过调色后端，只是调用投影仪的 `POST /show-image` 接口，并将 `imageUrl = 原图 URL`。
  - 目的是让用户一进入绘画界面，投影上就直接显示参考原图。
- 用户点击画面任意位置时，才会调用本文件定义的调色后端接口 `/api/color-mix/from-click`：
  - 后端返回 `targetColor`、`mixPlan` 以及处理后的 `projectorImageUrl`；
  - App 再次调用投影仪 `POST /show-image`，这次 `imageUrl = projectorImageUrl`，覆盖掉原图显示。
- 换句话说：
  - **“先投原图”只依赖投影仪 HTTP 协议，不依赖调色后端；**
  - 调色后端只负责“根据点击位置生成新的处理图 + 调色方案”，不需要参与原图首次显示的流程。