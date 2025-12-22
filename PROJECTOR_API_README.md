# 投影仪 HTTP 接口规范

本文件说明 **App ⇄ 投影仪** 的 HTTP 协议约定，用于在用户点击画面后，自动让投影仪显示后端处理好的图片。

## 总体设计

- App 与投影仪通过扫描二维码或手动输入 IP 的方式建立连接。
- 建立连接后，App 持有一个形如 `http://<projector-ip>:<port>` 的 Base URL。
- 当调色后端返回 `projectorImageUrl`（处理好的图片地址）后：
  - App 不下载图片本身；
  - App 仅将该 URL 通过 HTTP 发送给投影仪；
  - 投影仪主动用 HTTP 拉取并显示该图片。

---

## 1. Base URL

- 示例：`http://192.168.1.50:9000`
- 所有接口均在此基础上拼接路径，例如：`POST http://192.168.1.50:9000/show-image`。

> 实际 IP、端口由投影仪或网关设备提供，前端通过**扫描投影仪二维码**的方式获取，并只在「创作/设备选择」界面使用该扫码结果作为投影仪地址。

### 前端如何获取 Base URL（实现说明，便于对齐）

- 扫码入口页面：`src/pages/selector/index.vue`（创作/设备选择页）。
- 扫码逻辑：方法 `handleBindDevice()` 内部调用 `uni.scanCode`。
- 扫码结果处理：
  - 将扫码结果归一化为带协议的 `baseUrl`，例如：`172.16.24.227:8888 → http://172.16.24.227:8888`；
  - 保存到本地：`uni.setStorageSync('deviceBaseUrl', baseUrl)`；
  - 之后所有给投影仪的 HTTP 调用都会从 `deviceBaseUrl` 读取这个地址。
- 说明：
  - 这是**专门用于连接投影仪/画架的扫码入口**；
  - 另一个扫码入口在 `src/pages/create/index.vue` 右上角，用于扫描**成品画作二维码**，与投影仪连接无关。

---

## 2. 显示图片接口（核心）

> 目的：接收一张图片的 URL，并在投影仪上显示这张图。

### URL

- `POST /show-image`

（如果你现有实现使用了不同路径，如 `/show`、`/display`，可保持现状，只需与 App 约定统一。）

### Request Body

```json
{
  "imageUrl": "https://your-backend/processed/img_001_530_240.png",
  "displayMode": "fit",
  "expireAt": "2025-12-31T23:59:59Z"
}
```

字段说明：

- `imageUrl` `string`（必填）：
  - 由调色后端返回的处理后图片 URL（`projectorImageUrl`）；
  - 投影仪需直接使用该 URL 通过 HTTP(S) 拉取图片并显示；
  - 建议支持 `http` 与 `https`。
- `displayMode` `string`（可选）：
  - 图像显示模式，示例：`"fit"`（等比缩放完整显示）、`"fill"`（铺满裁剪）、`"stretch"`（拉伸）；
  - 若不需要可忽略，统一默认值即可。
- `expireAt` `string`（可选）：
  - 该图像的有效期 ISO8601 时间；
  - 投影仪可选用来做缓存清理或刷新策略。

### Response Body

成功示例：

```json
{
  "code": 0,
  "message": "ok"
}
```

失败示例：

```json
{
  "code": 1001,
  "message": "invalid image url"
}
```

字段说明：

- `code` `number`：0 表示成功，非 0 表示失败。
- `message` `string`：错误原因或提示信息。

### 行为要求

- 收到合法请求时，投影仪应尽快开始拉取 `imageUrl` 对应的图片，并在当前画面显示：
  - 若拉取成功，更新显示为新图像；
  - 若拉取失败，可保持原画面不变，并通过 `code` / `message` 告知 App。
- 对相同 `imageUrl` 的重复调用应是幂等的：
  - 多次请求不会导致错误，只是重复显示同一张图。

---

## 3. 可选：设备状态接口

> 用于 App 在连接或调试时查询投影仪状态（可选实现）。

### URL

- `GET /status`

### Response Body 示例

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "deviceId": "projector-001",
    "online": true,
    "firmwareVersion": "1.0.0",
    "lastImageUrl": "https://your-backend/processed/img_001_530_240.png"
  }
}
```

字段说明：

- `deviceId` `string`：投影仪唯一标识。
- `online` `boolean`：运行是否正常。
- `firmwareVersion` `string`：固件版本号。
- `lastImageUrl` `string`：最近一次成功显示的图片 URL（如果有）。

> 若你已有不同的状态接口实现，可保持现状，本节仅为推荐。

---

## 4. 网络与安全要求

- `imageUrl` 必须是投影仪可以访问到的 HTTP/HTTPS 资源：
  - 若在局域网环境，后端应返回局域网可达的地址（如 `http://192.168.x.x/...`）。
  - 若在公网环境，需确保投影仪具有访问公网的能力。
- 若图片访问需要鉴权，推荐使用 **带签名的短时效 URL**：
  - 例如在 URL 中附带签名参数和过期时间：`?token=...&expires=...`；
  - 避免投影仪侧还要实现复杂的登录/刷新 Token 逻辑。
- 建议限制图片大小与分辨率，避免：
  - 拉取时间过长；
  - 投影仪内存不足或渲染性能问题。

---

## 5. 调用时序（整体配合说明）

1. 用户在 App 内选择原图并进入绘画页；
2. App 将原图信息发给调色后端（可选），获得 `imageId`；
3. 用户在 App 画面上点击某个点；
4. App 将 `imageId + pixel` 发送给调色后端，获得：
   - 该点色彩信息；
   - 调色建议；
   - `projectorImageUrl`（处理后的图片 URL）；
5. App 立刻调用投影仪的 `POST /show-image` 接口，将 `projectorImageUrl` 传给投影仪；
6. 投影仪从 `projectorImageUrl` 拉取图片并显示在画面上。

---

通过以上约定，前端 App 不需要持有或二次传输图片二进制数据，只需在“调色后端”和“投影仪”之间转发一个 URL，实现点击即自动更新投影画面的效果。