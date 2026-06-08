# -*- coding: utf-8 -*-
"""
网页内容抓取器 — 垂直搜索基础设施

功能：
  从 URL 抓取网页内容并转换为纯文本，供 RL 推理的 <read_page> 工具使用。

使用：
  from src.rl.web_reader import WebPageReader
  reader = WebPageReader()
  content = reader.fetch("https://www.xiaomi.com/...", max_chars=2000)

依赖：
  - requests（HTTP 请求）
  - inscriptis（HTML → 纯文本，对中文支持好）
"""

import re
import os
import logging
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ── 非HTML后缀黑名单 ────────────────────────────────────────
_BINARY_EXTENSIONS = frozenset({
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv",
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dmg", ".iso", ".apk",
})

# ── 默认配置 ────────────────────────────────────────────────
DEFAULT_TIMEOUT      = 10    # 秒
DEFAULT_MAX_CHARS    = 2000  # 单页最大字符数
DEFAULT_MAX_SIZE     = 2 * 1024 * 1024  # 2MB
DEFAULT_USER_AGENT   = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class WebPageReader:
    """
    网页内容抓取器，线程安全。

    职责：
      1. 从 URL 抓取 HTML
      2. 过滤非 HTML 资源
      3. HTML → 纯文本（去除标签、脚本、样式）
      4. 截断到指定长度
    """

    def __init__(
        self,
        timeout:   int = DEFAULT_TIMEOUT,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_size:  int = DEFAULT_MAX_SIZE,
    ):
        self.timeout   = timeout
        self.max_chars = max_chars
        self.max_size  = max_size
        self._session  = requests.Session()
        self._session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    def fetch(self, url: str, max_chars: Optional[int] = None) -> str:
        """
        抓取并解析网页内容。

        Args:
            url:       目标 URL
            max_chars: 最大返回字符数（None 使用默认值）

        Returns:
            纯文本内容，失败时返回错误提示字符串
        """
        max_chars = max_chars or self.max_chars

        # URL 格式校验
        if not self.validate_url(url):
            return f"无法读取页面：URL地址无效或格式不支持（{url[:100]}）"

        try:
            return self._do_fetch(url, max_chars)
        except requests.Timeout:
            return f"页面读取超时：{self._short_url(url)}"
        except requests.ConnectionError:
            return f"无法连接到页面：{self._short_url(url)}"
        except requests.HTTPError as e:
            code = e.response.status_code if e.response else "?"
            return f"页面返回错误（HTTP {code}）：{self._short_url(url)}"
        except Exception as e:
            return f"页面读取失败：{str(e)[:100]}"

    def _do_fetch(self, url: str, max_chars: int) -> str:
        """实际抓取逻辑"""
        resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()

        # 检查 Content-Type
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"不支持的内容类型：{content_type[:50]}"

        # 检查大小
        content_length = len(resp.content)
        if content_length > self.max_size:
            return f"页面内容过大（{content_length // 1024}KB），已跳过"

        # 检测编码
        encoding = resp.encoding or "utf-8"
        try:
            html = resp.content.decode(encoding, errors="replace")
        except Exception:
            html = resp.content.decode("utf-8", errors="replace")

        # HTML → 纯文本
        text = self._html_to_text(html)

        # 提取域名作为来源标注
        domain = urlparse(url).netloc

        # 截断
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return f"[页面来源：{domain}]\n{text}"

    def _html_to_text(self, html: str) -> str:
        """HTML 转纯文本，优先使用 inscriptis，备选正则"""
        try:
            import inscriptis
            text = inscriptis.get_text(html)
        except ImportError:
            # 备选方案：简单正则去除标签
            text = self._simple_html_strip(html)

        # 清理多余空白
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _simple_html_strip(html: str) -> str:
        """简单 HTML 去标签（不依赖第三方库的备选方案）"""
        # 移除 script/style 块
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>",  "", text, flags=re.DOTALL | re.IGNORECASE)
        # 移除所有 HTML 标签
        text = re.sub(r"<[^>]+>", " ", text)
        # 解码常见 HTML 实体
        for entity, char in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')]:
            text = text.replace(entity, char)
        return text

    @staticmethod
    def validate_url(url: str) -> bool:
        """校验 URL 格式是否可抓取"""
        if not url or not isinstance(url, str):
            return False

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return False

        try:
            parsed = urlparse(url)
        except Exception:
            return False

        if not parsed.netloc:
            return False

        # 过滤非 HTML 后缀
        path_lower = parsed.path.lower()
        for ext in _BINARY_EXTENSIONS:
            if path_lower.endswith(ext):
                return False

        return True

    @staticmethod
    def _short_url(url: str) -> str:
        """缩短 URL 用于错误提示"""
        if len(url) > 80:
            return url[:77] + "..."
        return url


if __name__ == "__main__":
    # 快速测试
    logging.basicConfig(level=logging.INFO)

    reader = WebPageReader()

    test_urls = [
        "https://www.mi.com",                    # 小米官网
        "https://www.xiaomiev.com",              # 小米汽车
        "https://invalid-test-12345.com",         # 不存在的域名
        "https://example.com/test.pdf",           # 非 HTML
    ]

    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        result = reader.fetch(url)
        print(result[:500])
        print(f"  ... ({len(result)} 字符)")
