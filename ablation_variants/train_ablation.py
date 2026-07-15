from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train as benchmark_train
from variants import (
    PAPER_VARIANT_NAMES,
    apply_variant_training_controls,
    create_ablation_model,
    get_ablation_spec,
    transform_rule_index,
)


def parse_args():
    parser = benchmark_train.build_arg_parser()
    parser.description = "Train one paper-aligned DiNSR ablation variant."
    parser.add_argument("--variant", required=True, choices=PAPER_VARIANT_NAMES)
    parser.set_defaults(output_dir="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_ablation_spec(args.variant)
    apply_variant_training_controls(args, spec)
    if not args.output_dir:
        args.output_dir = str(PROJECT_ROOT / "outputs" / "ablation" / spec.slug)

    benchmark_train.main(
        args,
        model_name=spec.paper_name,
        model_factory=create_ablation_model,
        rule_index_transform=transform_rule_index,
        variant_metadata=spec.as_dict(),
    )


if __name__ == "__main__":
    main()
