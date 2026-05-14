# -*- coding: utf-8 -*-

# 导入需要的工具包
import os
import fitz
from typing import Tuple
from pymongo.collection import Collection
from typing_extensions import List

# 导入项目内部的自定义工具
from src import constant
from src.fields.manual_images import ManualImages
from src.client.mongodb_config import MongoConfig

# 全局配置
manual_images_collection: Collection = MongoConfig.get_collection("manual_images")  # 连接MongoDB数据库的 manual_images 表
image_save_dir = constant.image_save_dir  # 图片保存的文件夹（从配置里拿）
pdf_path = constant.pdf_path  # PDF文件路径（从配置里拿）


# 标题判断配置
TITLE_PROPERTIES = {
    "min_size": 10,          # 标题字体最小 >=10号
    "max_lines": 3,          # 标题最多3行
    "max_length": 30,        # 标题最多30个字
    "bold_weight": 0.7,      # 粗体权重（评分用）
    "page_clip": 50,         # 底部裁掉50pt（不搜页脚）
    "bottom_size": -200      # 向上扩展200pt找标题
}


def handle_image(img: Tuple, img_index: int, page: fitz.Page) -> ManualImages | None:
    """处理单个图片，提取信息 + 找标题 + 保存"""

    # 拿到图片唯一ID
    xref = img[0]

    # 从PDF里把图片提取出来
    base_image = page.parent.extract_image(xref)

    # ===================== 过滤小图标 =====================
    # 如果是png 或 宽度<=34，判定为小图标/水印，直接跳过
    if base_image["ext"] == "png" or base_image["width"] <= 34:
        return None

    # ===================== 保存图片到本地 =====================
    image_path = save_image(base_image, img_index, page.number)

    # ===================== 拿到图片在PDF里的坐标区域 =====================
    img_rect = page.get_image_bbox(img)

    # ===================== 扩大搜索区域，向上找标题 =====================
    expanded_rect = get_expanded_rect(img_rect, page.rect)

    # ===================== 在扩大区域里找和图片相关的文字块 =====================
    related_blocks = get_related_text_blocks(page, expanded_rect, img_rect.y0)

    # 把判断为【标题】的文本挑出来
    title_blocks = [text for is_title, text in related_blocks if is_title]

    # ===================== 返回图片信息对象 =====================
    return ManualImages(
        image_path=image_path,   # 图片保存路径
        page=page.number + 1,    # 页码（从1开始）
        title="\n".join(title_blocks)  # 自动找到的标题
    )


def save_image(base_image: dict, img_index: int, page_number: int) -> str:
    """保存图片并返回路径"""

    # 图片命名规则：page第几页_img第几张.后缀
    image_name = f"page{page_number + 1}_img{img_index + 1}.{base_image['ext']}"

    # 拼接完整路径
    image_path = os.path.join(image_save_dir, image_name)

    # 把图片二进制数据写入文件
    with open(image_path, "wb") as f:
        f.write(base_image["image"])

    # 返回路径
    return image_path


def get_expanded_rect(img_rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    """把图片区域向上扩大，用来找标题"""

    # 向上扩大 200pt，向下扩大图片高度的3倍
    expanded = img_rect + (0, TITLE_PROPERTIES["bottom_size"], 0, img_rect.height * 3)

    # 底部不超过页面-50pt（避免搜到页脚、页码）
    expanded[3] = min(expanded[3], page_rect[3] - TITLE_PROPERTIES["page_clip"])

    # 返回合法区域
    return expanded.intersect(page_rect)


def get_related_text_blocks(page: fitz.Page, rect: fitz.Rect, img_y: float) -> List[Tuple[bool, str]]:
    """在扩大的区域里找文字块，并判断是不是标题"""

    related_blocks = []

    # 遍历页面所有文字块
    for block in page.get_text("blocks"):
        block_rect = fitz.Rect(block[:4])

        # 如果文字块不在扩大区域里 → 跳过
        if not block_rect.intersects(rect):
            continue

        # 拿到文字内容
        block_text = block[4].strip()

        # 判断文字是否在图片上方
        above = block_rect.y1 < img_y

        # 判断这个块是不是标题
        is_title_block = is_title_block_candidate(page, block, above)

        # 加入结果列表
        related_blocks.append((is_title_block, block_text))

    return related_blocks


def is_title_block_candidate(page: fitz.Page, block: tuple, above: bool) -> bool:
    """智能判断：这个文本块是不是图片标题"""

    # 如果不是文本块 或 内容为空 → 不是标题
    if block[6] != 0 or not block[4].strip():
        return False

    try:
        # 获取字体信息（大小、是否粗体）
        span = page.get_text("dict")["blocks"][block[5]]["lines"][0]["spans"][0]
    except (IndexError, KeyError):
        return False

    # 文本内容
    text = block[4].strip()

    # 字体大小
    font_size = span["size"]

    # 是否粗体
    is_bold = "bold" in span["font"].lower()

    # 如果以句号结尾 → 不是标题
    if text.endswith(('.', '。', '!', '！')):
        return False

    # ===================== 开始打分 =====================
    score = 0

    # 字体≥10 → +2分
    score += 2 if font_size >= TITLE_PROPERTIES["min_size"] else 0

    # 是粗体 → +1分
    score += 1 if is_bold else 0

    # 行数≤3 → +0.5分
    score += 0.5 if (text.count('\n') + 1) <= TITLE_PROPERTIES["max_lines"] else 0

    # 字数≤30 → +0.5分
    score += 0.5 if len(text) <= TITLE_PROPERTIES["max_length"] else 0

    # 在图片上方 → +2分；在下方 → -1分
    score += 2 if above else -1

    # 最终 ≥3分 判定为标题
    return score >= 3
