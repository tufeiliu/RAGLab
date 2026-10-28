"""YAML model registry with environment-only secrets and explicit provider adapters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

@dataclass(frozen=True)
class ModelSpec:
    id: str
    kind: str
    label: str
    provider: str
    model: str
    api_key_env: str


class ModelCatalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.models = {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping in {self.path}")
        groups = data.get("models") or {}
        if not isinstance(groups, dict):
            raise ValueError(f"'models' must be a mapping in {self.path}")
        for kind in ("chat", "embedding"):
            entries = groups.get(kind) or {}
            if not isinstance(entries, dict):
                raise ValueError(f"models.{kind} must be a mapping in {self.path}, not {type(entries).__name__}")
            for alias, raw in entries.items():
                if not isinstance(raw, dict):
                    raise ValueError(f"models.{kind}.{alias} must be a mapping; check YAML indentation")
                try:
                    spec = ModelSpec(id=alias, kind=kind, **raw)
                except TypeError as exc:
                    raise ValueError(f"Invalid models.{kind}.{alias} in {self.path}: {exc}") from exc
                if spec.provider not in {"openai", "anthropic"}:
                    raise ValueError(f"Unsupported provider: {spec.provider}")
                if spec.provider == "anthropic" and kind == "embedding":
                    raise ValueError("Anthropic does not offer this embedding adapter")
                self.models[(kind, alias)] = spec
        if not self.models:
            raise ValueError(f"No models configured in {self.path}. Define models.chat and models.embedding.")

    def choices(self, kind: str, available_only: bool = True) -> dict[str, ModelSpec]:
        return {alias: spec for (category, alias), spec in self.models.items()
                if category == kind and (not available_only or bool(os.getenv(spec.api_key_env)))}

    def get(self, kind: str, alias: str, require_key: bool = True) -> ModelSpec:
        try:
            spec = self.models[(kind, alias)]
        except KeyError as exc:
            raise ValueError(f"Unknown {kind} model alias {alias!r} in {self.path}") from exc
        if require_key and not os.getenv(spec.api_key_env):
            raise EnvironmentError(f"Set {spec.api_key_env} in .env or your Space Secrets to use {spec.label}")
        return spec


def chat_client(spec: ModelSpec, cache=None):
    """cache: optional LangChain BaseCache (e.g. raglab.cache.SQLiteResponseCache)."""
    key = os.environ[spec.api_key_env]
    if spec.provider == "openai":
        from langchain_openai import ChatOpenAI
        # Some reasoning models reject temperature; leave unspecified for portability.
        return ChatOpenAI(model=spec.model, api_key=key, cache=cache)
    if spec.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError("Install raglab-experiments[anthropic] to use Claude") from exc
        return ChatAnthropic(model=spec.model, api_key=key, cache=cache)
    raise ValueError(spec.provider)


def embedding_client(spec: ModelSpec):
    if spec.provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=spec.model, api_key=os.environ[spec.api_key_env])
    raise ValueError(f"No embedding adapter for {spec.provider}")
