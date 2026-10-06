import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_MODELS_CONFIG = _REPO_ROOT / "config/models.yaml"
_DEFAULT_MODEL_PRICING_CONFIG = _REPO_ROOT / "config/model_pricing.yaml"
_DEFAULT_SCENARIO_GENERATION_CONFIG = _REPO_ROOT / "config/scenario_generation.yaml"
_DEFAULT_TRANSCRIPT_GENERATION_CONFIG = _REPO_ROOT / "config/transcript_generation.yaml"


class EmptyCompletionError(RuntimeError):
    """A model returned reasoning (or nothing) but no answer text.

    Raised so an empty generation surfaces as a *failed generation* rather than being
    scored as substantive model behavior. Both call layers retry on it: inspect_ai's
    ``retry_on_error`` (exception-driven, see ``chat_eval.generate_requiring_answer``)
    and ``api.py``'s tenacity envelope (via ``RETRYABLE_EXCEPTIONS``).
    """


class JudgeParseError(RuntimeError):
    """A judge response carried no valid verdict tag for one or more axes.

    Raised from the scorer so the sample fails rather than recording the axis defaults
    ("no"/"none"), which are indistinguishable from genuine negatives. inspect_ai's
    ``retry_on_error`` covers scorer exceptions, so the sample (rollout included) is
    re-run; if the retries are exhausted the sample is logged as an error and excluded
    from the rates rather than quietly biasing them.
    """


def load_models(config_path: Path = _DEFAULT_MODELS_CONFIG) -> dict:
    raw = yaml.safe_load(config_path.read_text())
    return {
        name: {**{k: v for k, v in cfg.items() if k != "api_key_env"}, "api_key": os.environ[cfg["api_key_env"]]}
        for name, cfg in raw.items()
    }


def to_inspect_model_id(cfg: dict) -> str:
    """Map a resolved model config (a `load_models()` entry) to an inspect-ai model id.

    This id format (`provider/model`) is the join key into `config/model_pricing.yaml`,
    so it's the single place both `CostTracker` and `chat_eval.to_inspect_model_id`
    derive the provider prefix from.
    """
    base_url = cfg.get("base_url", "")
    if "anthropic.com" in base_url:
        return f"anthropic/{cfg['model']}"
    if "openrouter.ai" in base_url:
        return f"openrouter/{cfg['model']}"
    if "generativelanguage.googleapis.com" in base_url:
        return f"google/{cfg['model']}"
    return f"openai/{cfg['model']}"


def load_model_pricing(config_path: Path = _DEFAULT_MODEL_PRICING_CONFIG) -> dict:
    """Load $/1M-token pricing keyed by inspect-ai model id (see `to_inspect_model_id`).

    Same file format as `inspect eval --model-cost-config` (`inspect_ai.model.ModelCost`).
    """
    return yaml.safe_load(config_path.read_text())


def load_scenario_config(config_path: Path = _DEFAULT_SCENARIO_GENERATION_CONFIG) -> dict:
    """Load categories, mistake_types, and severity_levels for scenario generation."""
    return yaml.safe_load(config_path.read_text())


def load_transcript_config(config_path: Path = _DEFAULT_TRANSCRIPT_GENERATION_CONFIG) -> dict:
    """Load user_tone and user_message_format options for transcript generation."""
    return yaml.safe_load(config_path.read_text())


@dataclass
class CostTracker:
    """Accumulates token usage and $ cost per model, keyed by `params["model"]`.

    Pricing comes from `config/model_pricing.yaml` (`load_model_pricing`), looked up by
    inspect-ai model id (`to_inspect_model_id(params)`); models missing an entry there
    are tracked (tokens) but cost 0.
    """

    pricing: dict[str, dict[str, float]] = field(default_factory=load_model_pricing, repr=False)
    usage: dict[str, dict[str, float]] = field(default_factory=dict)
    # Guards the read-modify-write increments below so a shared tracker can be
    # threaded through parallel API calls without losing updates.
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)

    def _entry(self, model_key: str) -> dict[str, float]:
        return self.usage.setdefault(model_key, {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})

    def _rates(self, params: dict) -> dict[str, float]:
        return self.pricing.get(to_inspect_model_id(params), {})

    def add_chat_usage(self, params: dict, usage) -> None:
        if usage is None:
            return
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        rates = self._rates(params)
        with self._lock:
            entry = self._entry(params["model"])
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
            entry["cost"] += prompt_tokens / 1e6 * rates.get("input", 0.0)
            entry["cost"] += completion_tokens / 1e6 * rates.get("output", 0.0)

    def add_embedding_usage(self, params: dict, usage) -> None:
        if usage is None:
            return
        prompt_tokens = usage.prompt_tokens or 0
        rates = self._rates(params)
        with self._lock:
            entry = self._entry(params["model"])
            entry["prompt_tokens"] += prompt_tokens
            entry["cost"] += prompt_tokens / 1e6 * rates.get("input", 0.0)

    def print_summary(self, label: str = "") -> None:
        suffix = f" ({label})" if label else ""
        print(f"\nLLM cost so far{suffix}:")
        total = 0.0
        for model, stats in self.usage.items():
            print(
                f"  {model:30s} prompt={stats['prompt_tokens']:>9.0f}  "
                f"completion={stats['completion_tokens']:>9.0f}  cost=${stats['cost']:.4f}"
            )
            total += stats["cost"]
        print(f"  {'TOTAL':30s} cost=${total:.4f}")
