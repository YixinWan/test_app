# 油画辅助后端接入说明（给前端工程师）

本文说明前端 App 如何调用油画辅助后端，获取“点击取色 + 调色建议 + 投影局部高亮图”。
后端已经实现好 HTTP 接口，你只需要按文档调用即可。

后端基于 **FastAPI**，统一使用 JSON：

- 基地址示例：`http://<backend-host>:8000`
- 所有响应结构统一为：

```json
{
  "code": 0,
  "message": "ok",
  "data": { ... }
}
```

`code = 0` 表示成功；`code != 0` 表示失败，`message` 为错误提示。

---

## 1. 后端当前支持的功能概览

1. **原图登记**（可选）：记录一张原始油画照片，返回 `imageId`。
2. **原图投影**：在进入绘画界面时，直接把整张原图投到画布上（返回整图的 `projectorImageUrl`）。
3. **点击取色 + 调色建议 + 局部高亮图**：用户点击原图的任意一点：
  - 返回该点颜色的调色建议（颜料组合 + 比例 + 分步调色步骤）；
  - 生成一张只有点击区域保留颜色、其他区域为黑色的掩膜图（返回 `projectorImageUrl`）。
4. **无需 imageId 的点击取色**：调试或简单集成时，可以直接用 `imageUrl + pixel` 完成第 3 点功能。

后端所有接口都通过 HTTP+JSON 提供，前端只需要根据下文的接口说明拼请求即可。

---

## 2. 整体调用链路

分为两类场景：

### 2.1 进入绘画界面：原图投影

1. App 选择好一张原图，准备进入绘画界面；
2. 前端调用 `POST /api/painting/show-original`，传入原图的 `imageUrl + width + height`；
3. 后端：
  - 记录原图信息，生成 `imageId`；
  - 返回一个可供投影仪直接访问的整图地址 `projectorImageUrl`；
4. 前端：
  - 将 `imageId` 保存到当前绘画会话的状态中；
  - 把 `projectorImageUrl` 原样传给投影仪的 `/show-image` 接口，让投影仪显示整张原图。

> 备注：老的 `/api/painting/register-original` 仍然可用，只做“登记不投影”；新接口 `/api/painting/show-original` 在登记的同时直接给出整图 `projectorImageUrl`，更适合当前流程。

### 2.2 在绘画界面内：点击取色 + 局部高亮

1. 用户在画布上点击某点；
2. 前端将该点击点从“屏幕/画布坐标”换算为“原图像素坐标 (pixel.x, pixel.y)`；
3. 前端调用 `POST /api/color-mix/from-click`（推荐）或 `POST /api/color-mix/from-click-by-url`：
  - 推荐：传入 `imageId + pixel`；
  - 简化/调试：传入 `imageUrl + pixel`；
4. 后端返回：
  - 该点颜色信息 `targetColor`；
  - 调色方案列表 `mixPlan`；
  - 分步调色步骤 `steps`；
  - 一张只高亮点击区域的掩膜图 `projectorImageUrl`；
5. 前端：
  - 根据 `targetColor + mixPlan + steps` 更新 App 内 UI；
  - 将 `projectorImageUrl` 原样传给投影仪 `/show-image`，让投影仪只亮这一块区域。

---

## 3. 接口一：原图登记（可选）

> 推荐使用，有利于后端做缓存和后续扩展；如果不想用，也可以直接看第 3 节 `from-click-by-url` 接口。

### 2.1 URL

`POST /api/painting/register-original`

### 2.2 Request Body

```json
{
  "imageUrl": "static/originals/origin_001.png",
  "width": 1920,
  "height": 1080
}
```

字段：

- `imageUrl` `string`
  - 当前实现中，后端把它当作“后端机器可直接读取到的路径”：
    - 相对路径，如：`static/originals/origin_001.png`（相对于后端工程根目录）；
    - 或绝对路径（本地测试环境）。
  - 正式环境可以切到真正的 HTTP URL（届时后端会做适配）。
- `width` / `height` `number`
  - 原图的像素尺寸；
  - 前端在图片加载完成后即可获取。

### 2.3 Response Body

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

字段：

- `imageId` `string`
  - 后端生成的原图唯一 ID；
  - 后续点击取色接口的主键。
- `normalizedUrl` `string`
  - 标准化后的原图 URL（当前实现与入参相同，用于后端自查，前端一般用不到）。

### 2.4 前端伪代码示例

```js
const BACKEND_BASE_URL = 'http://<backend-host>:8000';

async function registerOriginal(imageUrl, width, height) {
  const res = await fetch(`${BACKEND_BASE_URL}/api/painting/register-original`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageUrl, width, height }),
  });

  const json = await res.json();
  if (json.code !== 0) {
    throw new Error(json.message || 'register-original failed');
  }

  const { imageId } = json.data;
  // 请把 imageId 持久保存到当前绘画会话的状态中
  return imageId;
}
```

---

## 4. 接口二：点击取色 + 调色建议 + 掩膜图（核心）

### 3.1 URL

`POST /api/color-mix/from-click`

### 3.2 Request Body

```json
{
  "imageId": "img_xxxxxx",
  "pixel": {
    "x": 530,
    "y": 240
  }
}
```

字段：

- `imageId` `string`
  - 上面登记接口返回的 ID。
- `pixel` `object`
  - 点击点在**原图像素坐标系**中的位置：
  - `pixel.x`：0 ≤ x < 原图宽度；
  - `pixel.y`：0 ≤ y < 原图高度。

> 重点：如果前端对原图做了缩放/裁剪，需要根据缩放比例和偏移，把“屏幕坐标 / 画布坐标”换算为“原图像素坐标”，并保证不越界。

### 3.3 Response Body

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

字段说明：

- `targetColor`:
  - `hex` `string`：点击点的颜色，比如 `#ff6b6b`；
  - `name` `string`：目前为空字符串，后续可能扩展为中文色名（“珊瑚红”等）。
- `mixPlan` `array<object>`：调色方案，每一项代表一种颜料：
  - `name` `string`：颜料名称，例如“钛白”、“镉红”；
  - `color` `string`：该颜料代表色（用于渲染色条），形如 `#rrggbb`；
  - `ratio` `number`：该颜料在配方中的比例，百分比（0–100），带小数。
- `steps` `array<object>`：分步调色数据（可选使用）：
  - `step_num`：步骤编号，从 1 开始；
  - `parts`：每种颜料的“份数”；
  - `names`：对应颜料名称；
  - `rgbs`：对应颜料的 RGB 数组；
  - `mixed_hex`：本步骤混合后的颜色。
- `projectorImageUrl` `string`：
  - 相对 URL，例如 `/static/processed/img_xxxxxx_530_240.png`；
  - 指向一张**只有点击附近小区域保留颜色，其余区域为黑色**的新图片；
  - 这张图就是要投给投影仪显示的图片。

### 3.4 前端调用与 UI 更新（伪代码）

```js
const BACKEND_BASE_URL = 'http://<backend-host>:8000';

async function getColorMixFromClick(imageId, pixelX, pixelY) {
  const res = await fetch(`${BACKEND_BASE_URL}/api/color-mix/from-click`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      imageId,
      pixel: { x: pixelX, y: pixelY },
    }),
  });

  const json = await res.json();
  if (json.code !== 0) {
    throw new Error(json.message || 'color-mix failed');
  }

  const { targetColor, mixPlan, steps, projectorImageUrl } = json.data;
  return { targetColor, mixPlan, steps, projectorImageUrl };
}

// 示例：在画布点击事件里
async function handleCanvasClick(pixelX, pixelY, imageId, deviceBaseUrl) {
  const { targetColor, mixPlan, steps, projectorImageUrl } =
    await getColorMixFromClick(imageId, pixelX, pixelY);

  // 1）更新 App 内部的调色 UI
  updateColorUI({ targetColor, mixPlan, steps });

  // 2）拼出完整的后端图片 URL，给投影仪用
  const projectorImageFullUrl = `${BACKEND_BASE_URL}${projectorImageUrl}`;

  // 3）通知投影仪展示该图片（下一节有 show-image 协议）
  await sendToProjector(deviceBaseUrl, projectorImageFullUrl);
}
```

---

## 5. 接口三：不用 imageId，直接用 imageUrl（调试/简化）

如果暂时不想维护 `imageId`，可以直接使用这个接口：

### 4.1 URL

`POST /api/color-mix/from-click-by-url`

### 4.2 Request Body

```json
{
  "imageUrl": "static/originals/origin_001.png",
  "pixel": { "x": 530, "y": 240 }
}
```

语义与 `/api/color-mix/from-click` 类似，只是不用提前调用登记接口。返回结构完全一致：同样有 `targetColor/mixPlan/steps/projectorImageUrl`。

---

## 6. 与投影仪的配合调用

投影仪的 HTTP 协议在单独文档《PROJECTOR_API_README.md》中，这里只写与后端相关的关键点。

### 5.1 投影仪 Base URL

- 前端通过扫二维码或手动输入，拿到投影仪地址：
  - 形如：`http://192.168.1.50:9000`
- 在前端统一保存为：`deviceBaseUrl`。

### 5.2 显示图片接口

- URL：`POST /show-image`
- Request Body 示例：

```json
{
  "imageUrl": "http://<backend-host>:8000/static/processed/img_xxxxxx_530_240.png",
  "displayMode": "fit"
}
```

前端封装函数示例：

```js
async function sendToProjector(deviceBaseUrl, imageUrl) {
  const res = await fetch(`${deviceBaseUrl}/show-image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      imageUrl,         // 直接使用 BACKEND_BASE_URL + projectorImageUrl
      displayMode: 'fit'
    }),
  });

  const json = await res.json();
  if (json.code !== 0) {
    throw new Error(json.message || 'projector show-image failed');
  }
}
```

调用顺序简写为：

1. （可选）`register-original` → 拿到 `imageId`；
2. 每次点击 → `color-mix/from-click` 或 `color-mix/from-click-by-url`；
3. 用 `BACKEND_BASE_URL + projectorImageUrl` → 调 `deviceBaseUrl + /show-image`。

---

## 7. 错误码与前端处理建议

后端所有接口统一返回：

```json
{
  "code": <number>,
  "message": "<string>",
  "data": <object | null>
}
```

典型错误码：

- `1001`：`imageId` 或 `imageUrl` 不存在 / 图片读取失败；
- `1002`：`pixel` 越界（点击点超出原图范围）；
- `2001`：调色算法失败（例如调色盘文件缺失）；
- `2002`：生成处理图片失败（磁盘写入等问题）。

前端建议：

- 若 `code != 0`：
  - 用 `message` 做 Toast / 弹窗；
  - 本次点击不更新 UI，不调用投影仪；
  - 保持上一次结果。

---

## 8. 前端集成要点 Checklist

- [ ] 有一个后端基地址常量 `BACKEND_BASE_URL`；
- [ ] 在进入绘画页时，调用 `register-original` 或至少拿到与后端一致的 `imageUrl`；
- [ ] 在画布点击事件中，将坐标正确换算为**原图像素坐标**；
- [ ] 每次点击按顺序调用：
  - `/api/color-mix/from-click`（或 `/from-click-by-url`）；
  - 更新调色 UI；
  - 拼接 `BACKEND_BASE_URL + projectorImageUrl`，再调用投影仪的 `/show-image`；
- [ ] 对所有 `code != 0` 的情况做统一错误提示处理。

如需要针对具体框架（UniApp / 小程序 / Vue / React 等）的示例代码，可以在本文件基础上再补充对应实现，调用协议保持一致即可。
