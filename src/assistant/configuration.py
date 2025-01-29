import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from dataclasses import dataclass

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""
    max_web_research_loops: int = 3
    llm_provider: str = "gemini"#"gemini"  # 可选值: "ollama", "openai", "anthropic", "gemini"
    llm_model: str = "gemini-exp-1206"#"gemini-exp-1206"  # 模型名称
    output_language: str = "English"  # 输出语言，可选值: "Chinese", "English"
    summary_max_length: int = 10000  # 摘要最大长度（字符数）
    summary_min_length: int = 5000  # 摘要最小长度（字符数）
    openai_api_key: str = ""  # OpenAI API密钥
    anthropic_api_key: str = ""  # Anthropic API密钥
    google_api_key: str = "AIzaSyAH_pq8prelABgkDiDgJv6YLaSieI7xCCM"  # Google API密钥

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})