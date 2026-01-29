# 依据4.1清理后的色块结果，按分位数提取“暗部”（显示每个块，暗部用纯白表示）
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
from typing import List, Dict

def percentile_dark_from_step41(step41_outputs: List[Dict],
                                src_bgr: np.ndarray,
                                percentile: float = 0.30,
                                morph_kernel_size: int = 3,
                                min_region_area: int = 50):
    """
    对4.1得到的每个块（cleaned_mask/result_img_clean），
    基于该块内每个连通子块的V通道分布，按分位数提取最暗部分。

    返回：
      - per_block: 列表，每项包含：
          {
            'block_index': int,
            'white_mask': np.ndarray (uint8, 0/255),  # 暗部=白
            'thresholds': List[(label_id, thr_v)],    # 每个连通域的V阈值
            'preview_rgb': np.ndarray                 # 仅用于快速预览（可选）
          }
    """
    per_block = []

    hsv_full = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV)
    v_full = hsv_full[:, :, 2].astype(np.float32)

    for item in step41_outputs:
        cleaned_mask = item['cleaned_mask']
        # 连通域：在cleaned_mask里找每个子块
        num, labels = cv2.connectedComponents((cleaned_mask > 0).astype(np.uint8), connectivity=8)

        white_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)  # 暗部=255
        thresholds = []

        for lab in range(1, int(num)):
            comp_mask = (labels == lab)
            v_vals = v_full[comp_mask]
            if v_vals.size == 0:
                continue
            thr_v = float(np.quantile(v_vals, percentile))
            thresholds.append((lab, thr_v))

            comp_dark = comp_mask & (v_full <= thr_v)

            # 形态学与面积过滤
            if morph_kernel_size and morph_kernel_size > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
                comp_dark = cv2.morphologyEx(comp_dark.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
            if int(np.count_nonzero(comp_dark)) >= int(min_region_area):
                white_mask[comp_dark] = 255

        # 预览图（可选）：原块图 + 不改变原意，仅做占位
        preview_rgb = cv2.cvtColor(item['result_img_clean'], cv2.COLOR_BGR2RGB)

        per_block.append({
            'block_index': int(item['block_index']),
            'white_mask': white_mask,
            'thresholds': thresholds,
            'preview_rgb': preview_rgb,
        })

    return per_block

# 参数
percentile = 0.02   # 最暗30%
morph_kernel_size = 1
min_region_area = 50

# 运行（基于4.1的输出）
if 'step41_outputs' in locals() and isinstance(step41_outputs, list) and len(step41_outputs) > 0:
    per_block = percentile_dark_from_step41(
        step41_outputs,
        my_img,
        percentile=percentile,
        morph_kernel_size=morph_kernel_size,
        min_region_area=min_region_area,
    )

    # 展示：每个块两列 -> 左：4.1清理后色块 右：暗部纯白图
    n = len(per_block)
    cols = 2
    rows = n
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 3*rows))
    if rows == 1:
        axes = np.array([axes])

    for i, blk in enumerate(per_block):
        axes[i, 0].imshow(step41_outputs[i]['result_img_clean'][:, :, ::-1])  # BGR->RGB
        cdeg = step41_outputs[i]['center_deg']
        axes[i, 0].set_title(f'块#{step41_outputs[i]["block_index"]+1} (H≈{cdeg:.0f}°)\n4.1清理后结果')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(blk['white_mask'], cmap='gray', vmin=0, vmax=255)
        axes[i, 1].set_title(f'暗部=纯白 (≤{int(percentile*100)}%分位)')
        axes[i, 1].axis('off')

    plt.tight_layout()

    # 保存整图
    base_dir = workspace_root if 'workspace_root' in globals() else os.getcwd()
    combined_path = os.path.join(base_dir, '4_2_blocks_dark_combined.png')
    try:
        fig.savefig(combined_path, dpi=200, bbox_inches='tight')
        print(f'整图已保存: {combined_path}')
    except Exception as e:
        print(f'整图保存失败: {e}')

    plt.show()

    # 分别保存每个子图（左：清理后色块；右：暗部纯白）——不带标题、无白边，与原图尺寸一致
    try:
        for i, blk in enumerate(per_block):
            block_no = int(step41_outputs[i]['block_index']) + 1
            # 左侧：直接以BGR保存，不走Matplotlib，避免边框与尺寸变化
            clean_bgr = step41_outputs[i]['result_img_clean']  # HxWx3 (BGR)
            out_path_left = os.path.join(base_dir, f'block_{block_no:02d}_clean.png')
            ok1 = cv2.imwrite(out_path_left, clean_bgr)

            # 右侧：白掩膜，0/255 单通道，尺寸与原图一致
            white_mask = blk['white_mask']  # HxW (uint8, 0/255)
            out_path_right = os.path.join(base_dir, f'block_{block_no:02d}_dark_white.png')
            ok2 = cv2.imwrite(out_path_right, white_mask)

            if not ok1 or not ok2:
                print(f'块#{block_no} 子图保存存在失败: clean({ok1}), dark({ok2})')

        print(f'共保存 {n*2} 个子图到: {base_dir} （无标题、无白边、原尺寸）')
    except Exception as e:
        print(f'子图保存失败: {e}')

    # 简要打印每块的阈值信息
    for i, blk in enumerate(per_block):
        thr_preview = ', '.join([f'{lab}:{thr:.1f}' for lab, thr in blk['thresholds'][:5]])
        if len(blk['thresholds']) > 5:
            thr_preview += ', ...'
        print(f"块#{step41_outputs[i]['block_index']+1} 阈值(前5): {thr_preview}")
else:
    print('未找到4.1的输出(step41_outputs)。请先运行4.1单元格。')