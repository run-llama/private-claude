import os
from typing import Dict
from llama_index.core.settings import Settings
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def llm_config_from_env() -> Dict:
    from llama_index.core.constants import DEFAULT_TEMPERATURE

    model = os.getenv("MODEL")
    temperature = os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    max_tokens = os.getenv("LLM_MAX_TOKENS")
    profile_name = os.getenv("AWS_PROFILE_NAME")

    config = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
        "profile_name": profile_name,
    }
    return config


def embedding_config_from_env() -> Dict:
    model = os.getenv("EMBEDDING_MODEL")

    config = {"model_name": model}
    return config


def init_settings():
    llm_configs = llm_config_from_env()
    embedding_configs = embedding_config_from_env()

    Settings.llm = Bedrock(**llm_configs)
    Settings.embed_model = HuggingFaceEmbedding(**embedding_configs)
    Settings.chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
    Settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "20"))
