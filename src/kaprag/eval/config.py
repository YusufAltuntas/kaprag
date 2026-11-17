"""Eval run configs: `configs/<name>.yaml`.

description: One line saying what this run tests (becomes the run's README.md).
topic: baseline-dense            # used in the run folder name; [a-z0-9-]
pipeline:
  factory: fake_oracle           # registry key, or "package.module:function"
  params: {error_rate: 0.25}     # passed to the factory as PipelineContext.params
judge:
  kind: llm                      # llm (claude-sonnet-5-5) | fake (tests only)
  model: claude-sonnet-5-5
concurrency: 4                   # questions in flight at once
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kaprag.eval.judge import JUDGE_MODEL

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineConfig(_Strict):
    factory: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class JudgeConfig(_Strict):
    kind: Literal["llm", "fake"] = "llm"
    model: str = JUDGE_MODEL


class EvalConfig(_Strict):
    description: str = Field(min_length=1, pattern=r"^[^\r\n]+$")
    topic: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    pipeline: PipelineConfig
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    concurrency: int = Field(default=4, ge=1)


class ConfigError(ValueError):
    pass


def config_path(name: str, config_dir: Path = CONFIG_DIR) -> Path:
    return Path(config_dir) / f"{name}.yaml"


def load_config(name: str, config_dir: Path = CONFIG_DIR) -> EvalConfig:
    """Load and validate `configs/<name>.yaml`."""
    path = config_path(name, config_dir)
    if not path.is_file():
        raise ConfigError(f"config {name!r} not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return EvalConfig.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"invalid config {path}: {exc}") from None
