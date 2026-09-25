#!/usr/bin/env python3
"""Generate prompt files for each operator in a dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kernelgenbench.dataset import get_kernelgenbench_operators
from kernelgenbench.dataset.kernel_list import DynamicImplInfo
from kernelgenbench.dataset.dataloader import TorchOpsLoader


# Global loaders (reuse to avoid repeated initialization)
_torch_ops_loader = None
_impl_info_loader = None


def get_loaders():
    global _torch_ops_loader, _impl_info_loader
    if _torch_ops_loader is None:
        _torch_ops_loader = TorchOpsLoader(to_str=True)
    if _impl_info_loader is None:
        _impl_info_loader = DynamicImplInfo()
    return _torch_ops_loader, _impl_info_loader


def load_template(template_path: Path) -> str:
    with open(template_path, "r") as f:
        return f.read()


def get_operator_info(op_name: str, namespace: str) -> dict:
    """Get operator signatures and implementation info."""
    info = {"signatures": "", "impl_info": "", "input_args": ""}

    if namespace in ("vllm13", "cublas"):
        info["signatures"] = f"- `{op_name}`: (See {namespace} documentation)"
        info["impl_info"] = f"- `{op_name}`"
        info["input_args"] = f"(See {namespace} documentation for parameter details)"
        return info

    loader, impl_info_loader = get_loaders()
    try:
        api_info = loader.get_operator(namespace, op_name)
        sig_lines = []
        for overload_name, schema in api_info.schemas.items():
            full_name = f"{op_name}.{overload_name}" if overload_name else op_name
            sig_lines.append(f"- `{full_name}`: {schema}")
        info["signatures"] = "\n".join(sig_lines)

        impl = impl_info_loader.get(op_name, namespace=namespace)
        if impl:
            info["impl_info"] = "\n".join(f"- `{name}` (autograd: {auto.name})" for name, auto in impl)
        else:
            impl_lines = [f"- `{name}`" for name in api_info.schemas.keys()]
            info["impl_info"] = "\n".join(impl_lines) if impl_lines else "- `default`"

        if api_info.schemas:
            first_schema = next(iter(api_info.schemas.values()))
            info["input_args"] = f"```\n{first_schema}\n```"
    except Exception as e:
        print(f"Warning: Could not get info for {namespace}::{op_name}: {e}")
        info["signatures"] = f"(Could not load schema for {op_name})"
        info["impl_info"] = f"- `{op_name}`"
        info["input_args"] = "(See documentation)"

    return info


def render_prompt(template: str, operator: str, full_name: str, op_info: dict,
                  gpu_id: int = 0, python_path: str = "python") -> str:
    prompt = template.replace("{{OPERATOR}}", operator)
    prompt = prompt.replace("{{FULL_NAME}}", full_name)
    prompt = prompt.replace("{{GPU_ID}}", str(gpu_id))
    prompt = prompt.replace("{{PYTHON_PATH}}", python_path)
    prompt = prompt.replace("{{OP_SIGNATURES}}", op_info["signatures"])
    prompt = prompt.replace("{{IMPL_INFO}}", op_info["impl_info"])
    prompt = prompt.replace("{{INPUT_ARGS}}", op_info["input_args"])
    prompt = prompt.replace("{{REFERENCE_CODE}}", "")
    return prompt


def generate_prompts_for_dataset(
    dataset: str,
    output_dir: Path,
    template_path: Path,
    operators: list[str] | None = None,
    force: bool = False,
    python_path: str = "python",
) -> None:
    if dataset != "KernelGenBench":
        raise ValueError(f"Unknown dataset: {dataset}. Available: ['KernelGenBench']")

    template = load_template(template_path)
    flat_ops = get_kernelgenbench_operators()

    dataset_output_dir = output_dir / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    if operators:
        flat_ops = {k: v for k, v in flat_ops.items() if k.split("::")[-1] in operators}

    ops_list_path = dataset_output_dir / "ops_list.txt"
    with open(ops_list_path, "w") as f:
        for full_name in sorted(flat_ops.keys()):
            f.write(f"{full_name}\n")
    print(f"Generated {ops_list_path} ({len(flat_ops)} operators)")

    generated = 0
    skipped = 0
    for full_name in sorted(flat_ops.keys()):
        namespace, op_name = full_name.split("::", 1)
        prompt_path = dataset_output_dir / f"{full_name.replace('::', '__')}.md"

        if prompt_path.exists() and not force:
            skipped += 1
            continue

        op_info = get_operator_info(op_name, namespace)
        prompt = render_prompt(
            template=template,
            operator=op_name,
            full_name=full_name,
            op_info=op_info,
            python_path=python_path,
        )
        with open(prompt_path, "w") as f:
            f.write(prompt)
        generated += 1

    print(f"Generated {generated} prompts, skipped {skipped} existing")


def main():
    parser = argparse.ArgumentParser(description="Generate prompt files for operator datasets")
    parser.add_argument("--dataset", "-d", type=str, default="KernelGenBench",
                        choices=["KernelGenBench"], help="Dataset to generate prompts for")
    parser.add_argument("--op", "-o", type=str, default=None,
                        help="Specific operator(s) to generate, comma-separated (e.g., add,softmax)")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "prompts",
                        help="Output directory for prompts (default: prompts/)")
    parser.add_argument("--template", type=Path,
                        default=SCRIPT_DIR / "templates" / "triton_kernel.md",
                        help="Path to prompt template")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing prompt files")
    parser.add_argument("--python-path", type=str, default="python",
                        help="Python interpreter path for prompts (default: python)")

    args = parser.parse_args()
    operators = args.op.split(",") if args.op else None

    print(f"\n=== Generating prompts for {args.dataset} ===")
    generate_prompts_for_dataset(
        dataset=args.dataset,
        output_dir=args.output_dir,
        template_path=args.template,
        operators=operators,
        force=args.force,
        python_path=args.python_path,
    )


if __name__ == "__main__":
    main()
