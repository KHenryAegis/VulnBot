import os

import dataclasses
import sys
from enum import StrEnum
from typing import Dict, Any
from pathlib import Path

from config.pydantic_settings_file import *
from config.pydantic_settings_file import BaseFileSettings

PENTEST_ROOT = Path(os.environ.get("PENTEST_ROOT", ".")).resolve()


class Mode(StrEnum):
    Auto = "auto"
    Manual = "manual"
    SemiAuto = "semi"

    def __missing__(self, key):
        return self.Auto


class BasicConfig(BaseFileSettings):
    model_config = SettingsConfigDict(yaml_file=PENTEST_ROOT / "basic_config.yaml")

    log_verbose: bool = True

    enable_rag: bool = False

    mode: str = Mode.Auto

    @cached_property
    def LOG_PATH(self) -> Path:

        p = PENTEST_ROOT / "logs"
        return p

    KB_ROOT_PATH: str = str(PENTEST_ROOT / "data/knowledge_base")

    http_default_timeout: int = 300

    kali: dict = {
        "hostname": "10.10.0.5",
        "port": 22,
        "username": "root",
        "password": "root",
    }

    default_bind_host: str = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"

    api_server: dict = {"host": default_bind_host, "port": 7861, "public_host": "127.0.0.1", "public_port": 7861}

    webui_server: dict = {"host": default_bind_host, "port": 8501}

    def make_dirs(self):
        for p in [
            self.LOG_PATH,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        Path(self.KB_ROOT_PATH).mkdir(parents=True, exist_ok=True)


class DBConfig(BaseFileSettings):
    model_config = SettingsConfigDict(yaml_file=PENTEST_ROOT / "db_config.yaml")
    mysql: dict = {
        "host": "",
        "port": 3306,
        "user": "",
        "password": "",
        "database": ""
    }


class KBConfig(BaseFileSettings):
    model_config = SettingsConfigDict(yaml_file=PENTEST_ROOT / "kb_config.yaml")

    default_vs_type: str = "milvus"

    milvus: dict = {
        "uri": "",
        "user": "",
        "password": "",
    }

    kb_name: str = ""

    chunk_size: int = 750
    overlap_size: int = 150
    top_n: int = 1
    top_k: int = 3
    score_threshold: float = 0.5
    search_params: dict = {
        "metric_type": "L2",
        "params": {
            "nprobe": 10
        }
    }
    index_params: dict = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }

    text_splitter_dict: dict[str, dict[str, Any]] = {
        "SpacyTextSplitter": {
            "source": "huggingface",
            "tokenizer_name_or_path": "gpt2",
        },
        "RecursiveCharacterTextSplitter": {
            "source": "tiktoken",
            "tokenizer_name_or_path": "cl100k_base",
        }
    }

    text_splitter_name: str = "RecursiveCharacterTextSplitter"


class LLMConfig(BaseFileSettings):
    model_config = SettingsConfigDict(yaml_file=PENTEST_ROOT / "model_config.yaml")

    api_key: str = ""
    llm_model: str = "openai"
    base_url: str = ""
    llm_model_name: str = ""
    embedding_models: str = "maidalun1020/bce-embedding-base_v1"
    embedding_type: str = "local"
    context_length: int = 120000
    embedding_url: str = ""
    rerank_model: str = "maidalun1020/bce-reranker-base_v1"
    temperature: float = 0.5
    history_len: int = 5
    timeout: int = 600
    # Upper bound on the tokens of one reply (0 sends no limit). A reply that reaches it raises LLMTruncated.
    max_tokens: int = 8192
    # Attempts per request for transient endpoint failures (connection errors, timeouts, 408/409/429/5xx).
    max_retries: int = 3
    # Attempts to obtain a usable plan from the model before the run stops.
    max_plan_attempts: int = 3
    # Largest task list accepted from the model. The plan prompt asks for 1 to 5 tasks.
    max_plan_tasks: int = 30
    proxies: Dict[str, str] = dataclasses.field(default_factory=dict)


class ConfigsContainer:
    PENTEST_ROOT = PENTEST_ROOT

    basic_config: BasicConfig = settings_property(BasicConfig())
    kb_config: KBConfig = settings_property(KBConfig())
    llm_config: LLMConfig = settings_property(LLMConfig())
    db_config: DBConfig = settings_property(DBConfig())

    def create_all_templates(self):
        self.basic_config.create_template_file(write_file=True, file_format="yaml")
        self.kb_config.create_template_file(write_file=True, file_format="yaml")
        self.llm_config.create_template_file(write_file=True, file_format="yaml")
        self.db_config.create_template_file(write_file=True, file_format="yaml")

    def set_auto_reload(self, flag: bool = True):
        self.basic_config.auto_reload = flag
        self.kb_config.auto_reload = flag
        self.llm_config.auto_reload = flag
        self.db_config.auto_reload = flag


Configs = ConfigsContainer()
