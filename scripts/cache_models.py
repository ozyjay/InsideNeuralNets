#!/usr/bin/env python3
"""Download and validate the pretrained models used by the local demo."""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MODEL_OPTIONS, ModelUnavailableError, load_model

SUPPORTED_MODEL_KEYS = tuple(option.key for option in MODEL_OPTIONS)


def cache_models(model_keys: Sequence[str]) -> list[str]:
    """Cache each requested model and return the keys that failed."""
    failures: list[str] = []
    for model_key in model_keys:
        print(f"Caching {model_key}...")
        bundle = None
        try:
            bundle = load_model(model_key)
        except ModelUnavailableError as exc:
            print(f"Could not cache {model_key}: {exc}")
            failures.append(model_key)
        else:
            print(f"Cached and validated {bundle.label}.")
        finally:
            load_model.cache_clear()
            del bundle
            gc.collect()
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and validate pretrained model weights for offline demo use."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=SUPPORTED_MODEL_KEYS,
        dest="models",
        help="Cache one supported model. Repeat to cache several. Defaults to all models.",
    )
    args = parser.parse_args(argv)

    model_keys = tuple(args.models or SUPPORTED_MODEL_KEYS)
    failures = cache_models(model_keys)
    if failures:
        print("Model caching failed for: " + ", ".join(failures))
        return 1

    print("All requested model weights are ready for offline use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
