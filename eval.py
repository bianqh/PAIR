from __future__ import annotations

import sys
from pathlib import Path


def _resolve_config_path(argv: list[str]) -> str:
    for index, item in enumerate(argv):
        if item == "--config" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--config="):
            return item.split("=", 1)[1]
    return ""


def _select_main():
    config_path = _resolve_config_path(sys.argv[1:])
    parts = set(Path(config_path).parts)
    if "clip_qwen" in parts:
        from clip_qwen.eval import main

        return main
    from clip_rsitmd_baseline.eval import main

    return main


if __name__ == "__main__":
    _select_main()()
