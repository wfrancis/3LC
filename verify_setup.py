#!/usr/bin/env python3
"""
Pre-flight environment check. Run BEFORE register_tables.py.

Verifies Python version, packages, GPU, data files, and 3LC login.
Catches the most common setup problems before they waste your time.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parent

_PASS = "[OK]"
_FAIL = "[FAIL]"
_WARN = "[WARN]"
_SKIP = "[SKIP]"

_fail_count = 0
_warn_count = 0


def _check(label: str, ok: bool, detail: str, *, warn_only: bool = False) -> None:
    global _fail_count, _warn_count
    if ok:
        print(f"  {_PASS} {label}: {detail}")
    elif warn_only:
        _warn_count += 1
        print(f"  {_WARN} {label}: {detail}")
    else:
        _fail_count += 1
        print(f"  {_FAIL} {label}: {detail}")


def _try_import(module: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, "__version__", "installed")
        return True, str(ver)
    except ImportError:
        return False, "not installed"


def check_python() -> None:
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    ok = v.major == 3 and 9 <= v.minor < 14
    hint = ver_str if ok else f"{ver_str} — need 3.9 to 3.13"
    _check("Python version", ok, hint)

    if sys.platform == "win32" and "LocalCache" in sys.executable:
        _check(
            "Python source",
            False,
            "Microsoft Store Python detected — can break 3LC project discovery. "
            "Install from python.org instead.",
            warn_only=True,
        )


def check_packages() -> None:
    required = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("3lc (tlc)", "tlc"),
        ("3lc-ultralytics", "tlc_ultralytics"),
        ("ultralytics", "ultralytics"),
        ("umap-learn", "umap"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
    ]
    for display, module in required:
        ok, detail = _try_import(module)
        if not ok and module == "umap":
            detail += " — pip install umap-learn (training will crash without it)"
        _check(display, ok, detail)


def check_gpu() -> None:
    try:
        import torch

        has_cuda = torch.cuda.is_available()
        if has_cuda:
            name = torch.cuda.get_device_name(0)
            _check("CUDA GPU", True, name)
        else:
            _check(
                "CUDA GPU",
                False,
                "not available — training will run on CPU (much slower)",
                warn_only=True,
            )
    except Exception as e:
        _check("CUDA GPU", False, str(e), warn_only=True)


def check_data() -> None:
    required_files = [
        "config.yaml",
        "dataset.yaml",
        "sample_submission.csv",
        "register_tables.py",
        "train.py",
        "predict.py",
    ]
    required_dirs = [
        "data/train/images",
        "data/train/labels",
        "data/val/images",
        "data/val/labels",
        "data/test/images",
    ]

    for f in required_files:
        p = KIT_DIR / f
        _check(f"File: {f}", p.is_file(), "found" if p.is_file() else "MISSING")

    for d in required_dirs:
        p = KIT_DIR / d
        if p.is_dir():
            count = sum(1 for _ in p.iterdir())
            _check(f"Dir:  {d}", True, f"{count} files")
        else:
            _check(f"Dir:  {d}", False, "MISSING")


def check_cuda_pytorch_order() -> None:
    """Detect the common mistake: CPU-only torch when CUDA drivers exist."""
    try:
        import torch

        if not torch.cuda.is_available():
            cuda_version = getattr(torch.version, "cuda", None)
            if cuda_version is None:
                _check(
                    "PyTorch CUDA build",
                    False,
                    f"torch {torch.__version__} is CPU-only. "
                    "If you have a GPU: pip uninstall torch torchvision, "
                    "then reinstall with --index-url for your CUDA version.",
                    warn_only=True,
                )
            else:
                _check("PyTorch CUDA build", True, f"CUDA {cuda_version} (driver may be missing)")
        else:
            cuda_ver = getattr(torch.version, "cuda", "unknown")
            _check("PyTorch CUDA build", True, f"CUDA {cuda_ver}")
    except ImportError:
        _check("PyTorch CUDA build", False, "torch not installed", warn_only=True)


def main() -> int:
    print("=" * 65)
    print("  ENVIRONMENT CHECK — run before register_tables.py")
    print("=" * 65)

    print("\n-- Python --")
    check_python()

    print("\n-- Packages --")
    check_packages()

    print("\n-- GPU --")
    check_gpu()
    check_cuda_pytorch_order()

    print("\n-- Starter kit files --")
    check_data()

    print("\n" + "=" * 65)
    if _fail_count == 0 and _warn_count == 0:
        print("  ALL CHECKS PASSED — ready to run register_tables.py")
    elif _fail_count == 0:
        print(f"  PASSED with {_warn_count} warning(s) — review above, then proceed")
    else:
        print(f"  {_fail_count} FAILED, {_warn_count} warning(s) — fix errors before proceeding")
    print("=" * 65)
    return 1 if _fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
