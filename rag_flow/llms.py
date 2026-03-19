import os
import logging

from llama_index.core.llms import ChatMessage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 设置日志模版
logging.basicConfig(level=logging.WARNING,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "openai": {
        "base_url": "https://api.laozhang.ai/v1",
        "api_key": os.getenv("LAOZHANG_API_KEY"),
        "chat_model": "gpt-5.1-2025-11-13"
    },
    "qwen": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": DashScopeGenerationModels.QWEN_MAX,
    },
    "vllm": {
        "base_url": "http://ai.bygpu.com:58111/v1",
        "api_key": "vllm",
        "chat_model": "Qwen/Qwen3-4B"
    }
}

EMBED_MODEL_API_CONFIGS = {
    "qwen": {
        "embedding_model": DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,
        "text_type": DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
        "embed_api_key": os.getenv("DASHSCOPE_API_KEY"),
        "dimensions": 1536
    },
    "bge": {
        "embedding_model": r"C:\Users\qian gao\models\BAAI\bge-base-zh-v1___5",
    }
}

# 默认配置
DEFAULT_LLM_TYPE = "qwen"
DEFAULT_TEMPERATURE = 0.0


class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE):
    """
    初始化LLM实例

    Args:
        llm_type (str): LLM类型，可选值为 'openai', 'oneapi', 'qwen', 'vllm'
    Raises:
        LLMInitializationError: 当LLM初始化失败时抛出
    """
    try:
        # 检查llm_type是否有效
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # 特殊处理 ollama 类型
        if llm_type == "vllm":
            os.environ["OPENAI_API_KEY"] = "NA"
        if llm_type == "qwen":
            llm_chat = DashScope(
                api_key=config["api_key"],
                model=config["chat_model"],
                temperature=DEFAULT_TEMPERATURE,
                timeout=30,  # 添加超时配置（秒）
                max_retries=2  # 添加重试次数
            )
            logger.info(f"成功初始化 {llm_type} LLM")
            return llm_chat
        else:
            llm_chat = OpenAI(
                api_base=config["base_url"],
                api_key=config["api_key"],
                model=config["chat_model"],
                temperature=DEFAULT_TEMPERATURE,
                timeout=30,  # 添加超时配置（秒）
                max_retries=2  # 添加重试次数
            )

        logger.info(f"成功初始化 {llm_type} LLM")
        # return llm_chat
        return llm_chat

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


def initialize_embedding(llm_type: str = DEFAULT_LLM_TYPE):
    """
    初始化Embedding实例

    Args:
        llm_type (str): LLM类型，可选值为 'openai', 'oneapi', 'qwen', 'vllm'
    Raises:
        LLMInitializationError: 当Embedding初始化失败时抛出
    """
    try:
        # 检查llm_type是否有效
        if llm_type not in EMBED_MODEL_API_CONFIGS:
            raise ValueError(f"不支持的Embedding类型: {llm_type}. 可用的类型: {list(EMBED_MODEL_API_CONFIGS.keys())}")

        config = EMBED_MODEL_API_CONFIGS[llm_type]

        # 创建Embedding实例
        if llm_type == "qwen":
            llm_embedding = DashScopeEmbedding(
                api_key=config["embed_api_key"],
                model_name=config["embedding_model"],
                text_type=config["text_type"],
                # dimensions=config.get("dimensions", 1536)  # 默认维度为1536
            )
        elif llm_type == "bge":
            llm_embedding = HuggingFaceEmbedding(
                model_name=config["embedding_model"],
            )

        logger.info(f"成功初始化 {llm_type} Embedding")
        return llm_embedding

    except ValueError as ve:
        logger.error(f"Embedding配置错误: {str(ve)}")
        raise LLMInitializationError(f"Embedding配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化Embedding失败: {str(e)}")
        raise LLMInitializationError(f"初始化Embedding失败: {str(e)}")

# 示例使用
if __name__ == "__main__":
    try:
        # 测试不同类型的LLM初始化
        llm_chat = initialize_llm("qwen")
        messages = [
            ChatMessage(
                role="system", content="You are a pirate with a colorful personality"
            ),
            ChatMessage(role="user", content="What is your name"),
        ]
        resp = llm_chat.stream_chat(messages)
        for r in resp:
            print(r.delta, end="")
        
        # 测试不同类型的Embedding初始化
        llm_embedding = initialize_embedding("bge")
        embeddings = llm_embedding.get_text_embedding("Hello, world!")
        print(embeddings[:5])
        print(len(embeddings))
    except LLMInitializationError as e:
        logger.error(f"程序终止: {str(e)}")