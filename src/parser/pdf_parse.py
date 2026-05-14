# -*- coding: utf-8 -*-

import re
import fitz
import pdfplumber
import json
import copy
import hashlib
import tiktoken
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.collection import Collection
from typing_extensions import List

from src import constant
from src.fields.manual_images import ManualImages
from src.fields.manual_info_mongo import ManualInfo
from src.client.mongodb_config import MongoConfig
import src.parser.image_handler as image_handler
from src.client.semantic_chunk_client import request_semantic_chunk


# 全局配置
_chunk_size = 256  # 每次切分的文本块最大长度（token 计数）
_chunk_overlap = 50  # 块与块之间重叠的长度（防止句子被切断）
_min_filter_pages = 4  # 从第 5 页开始保留（idx=4，因为页码从0开始），作用：跳过封面、扉页、目录前几页
_max_filter_pages = 247  # 保留到第 248 页结束（idx=247），作用：跳过最后空白页、附录冗余页
_semantic_group_size = 10  # 语义分组大小（高级分块用）
_max_parent_size = 512  # 父块最大长度（分层分块用）
_page_clip = 50  # 页脚裁剪高度：50 pt（约1.76厘米）
encoding = tiktoken.get_encoding("cl100k_base")  # 加载 OpenAI 的 token 编码器（用来精确计算文本长度）
manual_text_collection: Collection = MongoConfig.get_collection("manual_text")  # 连接 MongoDB 的 manual_text 集合（存解析好的文档）
file_path = constant.pdf_path  # PDF 文件路径（从常量里取）


# ===== TextSplitter 设置 =====
# 初始化一个【递归文本切割器】，把长文本切成小段
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_chunk_size,        # 每块最大 256 token
    chunk_overlap=_chunk_overlap,  # 块之间重叠 50 token
    separators=["\n\n", "\n"],     # 切割优先级：先按段落，再按换行
    length_function=lambda text: len(encoding.encode(text))  # 用 token 算长度，不是字符
)


# ===== 文本预处理部分 =====
def sentence_split(text: str) -> list[str]:
    """按中文/英文标点切句"""
    # 按【中文句号、换行、制表符】切割文本
    sentences = re.split(r'(?<=[。\n\t])+', text.strip())
    
    # 去掉空字符串，把每句话清理干净后返回
    return [s.strip() for s in sentences if s.strip()]


# 定义一个函数，名字叫 load_pdf，返回值是：Document 组成的列表
def load_pdf() -> list[Document]:
    # 创建一个空列表，用来存放最终解析好的每一页文档
    raw_docs = []

    # 用 pdfplumber 打开 PDF 文件（自动管理文件，不用手动关）
    with pdfplumber.open(file_path) as pdf:
        # 同时用 fitz(PyMuPDF) 打开同一个PDF，专门用来提取图片
        fitz_pdf = fitz.open(file_path)

        # 遍历 PDF 的每一页，idx=页码（从0开始），page=当前页对象
        # tqdm 只是显示加载进度条，不影响功能
        for idx, page in enumerate(tqdm(pdf.pages)):

            # ====================== 过滤不需要的页 ======================
            # 如果当前页 小于 最小保留页 或 大于 最大保留页 → 跳过
            # 作用：跳过封面、目录、最后空白页
            if idx < _min_filter_pages or idx > _max_filter_pages:
                continue

            # ====================== 裁剪页面（裁掉页脚） ======================
            # 裁剪区域：左上角(0,0)，右下角(宽度, 高度-页脚高度)
            # 意思：裁掉底部 _page_clip 高度的页脚
            crop_box = (0, 0, page.width, page.height - _page_clip)
            # 执行裁剪，得到裁好的页面
            cropped_page = page.crop(crop_box)

            # ====================== 提取文本 ======================
            # 从裁剪后的页面提取文字，没文字就为空字符串
            text = cropped_page.extract_text() or ""

            # ====================== 提取图片 ======================
            # 建一个空列表，存当前页的图片信息
            manual_images_list: List[ManualImages] = []
            # 用 fitz 打开当前页（专门处理图片）
            fitz_page = fitz_pdf.load_page(idx)
            # 获取当前页所有图片
            images = fitz_page.get_images(full=True)

            # 遍历当前页的每一张图片
            for img_index, img in enumerate(images):
                # 调用图片处理器，处理这张图片，返回结构化信息
                manual_image: ManualImages = image_handler.handle_image(img, img_index, fitz_page)
                # 如果处理出有效图片，就转成JSON格式，加入图片列表
                if manual_image:
                    manual_images_list.append(json.loads(manual_image.json()))

            # ====================== 只保留有文字的页 ======================
            # 如果去掉空格后还有文字，才继续处理
            if text.strip():
                # 给文本生成一个唯一ID（MD5加密），用来去重/标识
                unique_id = hashlib.md5(text.encode('utf-8')).hexdigest()

                # ====================== 构造文档元数据 ======================
                metadata = {
                    "unique_id": unique_id,    # 文本唯一ID
                    "source": file_path,        # PDF文件路径
                    "page": idx + 1,            # 真实页码（从1开始）
                    "images_info": manual_images_list  # 当前页所有图片信息
                }

                # ====================== 存入最终文档列表 ======================
                # 把 文本内容 + 元数据 封装成 Document 对象，加入列表
                raw_docs.append(Document(page_content=text, metadata=metadata))

        # 关闭 fitz 打开的PDF（释放资源）
        fitz_pdf.close()

    # 返回所有解析好的页面文档列表
    return raw_docs


def texts_split(raw_docs: list[Document]) -> list[Document]:
    """句子级 + 语义感知切分"""
    all_split_docs = []

    for doc in tqdm(raw_docs):

        # 语义切分
        grouped_chunks = request_semantic_chunk(doc.page_content, group_size=_semantic_group_size)

        # 父doc
        parent_docs = []
        for group in grouped_chunks:
            parent_id = hashlib.md5(group.encode('utf-8')).hexdigest()
            parent_metadata = copy.deepcopy(doc.metadata)
            parent_metadata["unique_id"] = parent_id 
            parent_doc = Document(page_content=group, metadata=parent_metadata)
            parent_docs.append(parent_doc)
            if len(group) < _max_parent_size:
                all_split_docs.append(parent_doc)
        save_2_mongo(parent_docs)

        # 子doc
        for chunk in parent_docs:
            # 带overlap继续句子级切分
            split_docs = text_splitter.create_documents([chunk.page_content], metadatas=[chunk.metadata])
            reid_split_docs = []
            for child_doc in split_docs:
                child_id = hashlib.md5(child_doc.page_content.encode('utf-8')).hexdigest()
                if child_doc.page_content == chunk.page_content:
                    continue
                child_metadata = copy.deepcopy(chunk.metadata)
                child_metadata["unique_id"] = child_id
                child_metadata["parent_id"] = chunk.metadata["unique_id"]
                reid_child_doc = Document(page_content=child_doc.page_content, metadata=child_metadata)
                reid_split_docs.append(reid_child_doc)

            save_2_mongo(reid_split_docs)
            all_split_docs.extend(reid_split_docs)

    return all_split_docs


def save_2_mongo(split_docs):
    for doc in split_docs:
        # 从 metadata 中提取关键参数
        metadata = doc.metadata

        # 构造唯一性 unique_id
        unique_id = metadata.get("unique_id")
        if not unique_id:
            continue

        # 创建文档记录对象
        doc_record = ManualInfo(
            unique_id=unique_id,
            page_content=doc.page_content,
            metadata=metadata
        )

        # 更新数据库操作
        manual_text_collection.update_one(
            {"unique_id": doc_record.unique_id},
            {"$set": doc_record.model_dump()},
            upsert=True
        )


