import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", str(DATA_DIR / "knowledge"))).resolve()
MEMORY_DIR = DATA_DIR / "memory"

for path in [DATA_DIR, KNOWLEDGE_DIR, MEMORY_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 DEEPSEEK_API_KEY\n"
        "示例：DEEPSEEK_API_KEY=sk-xxxxxx"
    )

DEFAULT_USER_ID = os.getenv("USER_ID", "demo_user")
RESEARCH_TOPIC = os.getenv("RESEARCH_TOPIC", "你的研究主题")
MAX_HISTORY_TURNS = 6
MAX_REWRITE_TIMES = 1
MAX_REACT_STEPS = 2
LOCAL_RETRIEVAL_TOP_K = 4
MEMORY_TOP_K = 3
ARXIV_TOP_K = 3
