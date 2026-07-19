"""常量定義模組。

集中管理專案中使用的常量，避免魔法字串分散在程式碼中。
"""

from __future__ import annotations

# ========================== 日誌常量 ==========================

LOG_PREFIX = "[ImageGen]"
"""統一的日誌字首。"""


# ========================== 安全設定 ==========================

GEMINI_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)
"""Gemini API 支援的安全類別列表。"""


# ========================== 預設配置值 ==========================

DEFAULT_TIMEOUT = 180
"""預設請求超時時間（秒）。"""

DEFAULT_DOWNLOAD_TIMEOUT = 30
"""預設圖像下載超時時間（秒）。"""

DEFAULT_MAX_RETRY_ATTEMPTS = 5
"""預設最大重試次數。"""

DEFAULT_ASPECT_RATIO = "自動"
"""預設寬高比。"""

DEFAULT_RESOLUTION = "1K"
"""預設解析度。"""

DEFAULT_MAX_CONCURRENT_TASKS = 3
"""預設最大併發任務數。"""

DEFAULT_MAX_BATCH_COUNT = 4

DEFAULT_MAX_IMAGE_SIZE_MB = 10
"""預設最大圖片大小（MB）。"""

DEFAULT_MAX_CACHE_COUNT = 100
"""預設最大快取檔案數量。"""

DEFAULT_CLEANUP_INTERVAL_HOURS = 24
"""預設快取清理間隔（小時）。"""

DEFAULT_DAILY_LIMIT_COUNT = 10
"""預設每日生成限制次數。"""

DEFAULT_RATE_LIMIT_SECONDS = 0
"""預設使用者請求頻率限制（秒），0 表示不限制。"""

# ========================== 脫敏常量 ==========================

MASK_VISIBLE_CHARS = 4
"""敏感資訊脫敏時兩端顯示的字元數。"""

MASK_MIN_LENGTH = 8
"""需要脫敏的最小字串長度。"""

MASK_PLACEHOLDER = "****"
"""脫敏佔位符。"""

# ========================== 資料保留策略 ==========================

USAGE_DATA_RETENTION_DAYS = 7
"""使用資料保留天數。"""


# ========================== 解析度對映 ==========================

# 1K 解析度對映（適用於多種適配器）
RESOLUTION_1K_MAP = {
    "1:1": "1024x1024",
    "4:3": "1024x768",
    "3:4": "768x1024",
    "16:9": "1024x576",
    "9:16": "576x1024",
    "3:2": "1024x640",
    "2:3": "640x1024",
}

# 2K 解析度對映
RESOLUTION_2K_MAP = {
    "1:1": "2048x2048",
    "4:3": "2048x1536",
    "3:4": "1536x2048",
    "3:2": "2048x1360",
    "2:3": "1360x2048",
    "16:9": "2048x1152",
    "9:16": "1152x2048",
}


# ========================== 支援的寬高比 ==========================

SUPPORTED_ASPECT_RATIOS = (
    "自動",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
)
"""工具引數中支援的寬高比列表。"""


# ========================== 支援的解析度 ==========================

SUPPORTED_RESOLUTIONS = ("1K", "2K", "4K")
"""工具引數中支援的解析度列表。"""


# ========================== API 端點 ==========================

GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
"""Gemini API 預設 Base URL。"""

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com"
"""OpenAI API 預設 Base URL。"""

GITEE_AI_DEFAULT_BASE_URL = "https://ai.gitee.com"
"""Gitee AI 預設 Base URL。"""

JIMENG_DEFAULT_BASE_URL = "http://localhost:5100"
"""Jimeng2API 預設 Base URL。"""
