"""Shared command-line launcher for one-file baseline entry points."""
from __future__ import annotations

import sys
from pathlib import Path


def run(model_name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from baseline.train import main, parse_args

    args = parse_args()
    args.model_list = model_name
    main(args)
