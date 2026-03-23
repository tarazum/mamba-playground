"""
Experiment 01 — Environment Setup Check

Checks:
    1. Python + PyTorch version and CUDA availability
    2. GPU detection: compute capability, Blackwell (sm_120) notes if RTX 5070 detected
    3. mamba-ssm availability (GPU-only package)
    4. Speed test: MinimalSSM (CPU) throughput on typical sequence lengths
    5. Environment report saved to results/01_env_report.json

Run this first before any other experiment.
"""

import json
import platform
import sys
import time
from pathlib import Path

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)


def check_python():
    return {
        "version": sys.version,
        "platform": platform.platform(),
    }


def check_torch():
    try:
        import torch
        info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": [],
        }
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1e9, 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
                # Blackwell detection (sm_120 = compute 12.0)
                if props.major == 12:
                    gpu_info["architecture"] = "Blackwell"
                    gpu_info["blackwell_notes"] = (
                        "RTX 5000 series (Blackwell, sm_120) detected. "
                        "mamba-ssm pre-compiled wheels may NOT include sm_120 support. "
                        "If installation fails, build from source:\n"
                        "  export TORCH_CUDA_ARCH_LIST='12.0'\n"
                        "  pip install mamba-ssm --no-binary mamba-ssm\n"
                        "Requires: CUDA 12.6+, PyTorch 2.6+, gcc/nvcc in PATH."
                    )
                elif props.major == 9:
                    gpu_info["architecture"] = "Hopper"
                elif props.major == 8:
                    gpu_info["architecture"] = "Ampere"
                elif props.major == 7:
                    gpu_info["architecture"] = "Volta/Turing"
                info["gpus"].append(gpu_info)
        return info
    except ImportError:
        return {"error": "torch not installed — run: pip install -r requirements.txt"}


def check_mamba_ssm():
    try:
        import mamba_ssm
        return {
            "available": True,
            "version": getattr(mamba_ssm, "__version__", "unknown"),
            "note": "GPU mamba-ssm is available — full Mamba kernels enabled",
        }
    except ImportError:
        return {
            "available": False,
            "note": (
                "mamba-ssm not installed (GPU package, requires CUDA). "
                "Experiments will use the CPU MinimalSSM fallback from core/ssm.py. "
                "To install on GPU machine: pip install -r requirements-gpu.txt"
            ),
        }


def check_causal_conv1d():
    try:
        import causal_conv1d
        return {"available": True, "version": getattr(causal_conv1d, "__version__", "unknown")}
    except ImportError:
        return {"available": False, "note": "Required for mamba-ssm — not needed for CPU fallback"}


def speed_test_cpu_ssm():
    """Benchmark MinimalSSM on typical sequence lengths."""
    import torch
    from core.ssm import SSMClassifier

    configs = [
        {"seq_len": 32, "batch": 32, "d_model": 32, "n_layers": 2},
        {"seq_len": 64, "batch": 32, "d_model": 32, "n_layers": 2},
        {"seq_len": 128, "batch": 16, "d_model": 64, "n_layers": 2},
    ]

    results = []
    for cfg in configs:
        model = SSMClassifier(
            input_dim=6,
            d_model=cfg["d_model"],
            n_classes=3,
            n_layers=cfg["n_layers"],
        )
        model.eval()
        x = torch.randn(cfg["batch"], cfg["seq_len"], 6)

        # warmup
        with torch.no_grad():
            model(x)

        # time 10 forward passes
        start = time.perf_counter()
        n_runs = 10
        with torch.no_grad():
            for _ in range(n_runs):
                model(x)
        elapsed = time.perf_counter() - start
        ms_per_batch = (elapsed / n_runs) * 1000
        throughput = (cfg["batch"] * n_runs) / elapsed

        results.append({
            **cfg,
            "ms_per_batch": round(ms_per_batch, 1),
            "sequences_per_sec": round(throughput, 0),
            "verdict": "fast enough" if ms_per_batch < 500 else "slow — consider GPU",
        })

    return results


def speed_test_gpu_ssm():
    """Benchmark mamba-ssm on GPU if available."""
    try:
        import torch
        import mamba_ssm
        if not torch.cuda.is_available():
            return {"skipped": "CUDA not available"}

        from mamba_ssm import Mamba

        model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
        model.eval()
        x = torch.randn(32, 64, 64).cuda()

        # warmup
        with torch.no_grad():
            model(x)

        torch.cuda.synchronize()
        start = time.perf_counter()
        n_runs = 50
        with torch.no_grad():
            for _ in range(n_runs):
                model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        ms_per_batch = (elapsed / n_runs) * 1000
        throughput = (32 * n_runs) / elapsed

        return {
            "ms_per_batch": round(ms_per_batch, 2),
            "sequences_per_sec": round(throughput, 0),
            "speedup_note": "Compare to CPU result for same config",
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 60)
    print("Agent Pool / SimpleAO / TSE — Mamba Playground Setup Check")
    print("=" * 60)

    report = {}

    print("\n[1/5] Python environment...")
    report["python"] = check_python()
    print(f"  Python {report['python']['version'].split()[0]} on {platform.system()}")

    print("\n[2/5] PyTorch + CUDA...")
    report["torch"] = check_torch()
    if "error" in report["torch"]:
        print(f"  ERROR: {report['torch']['error']}")
    else:
        t = report["torch"]
        print(f"  PyTorch {t['version']} | CUDA {'available' if t['cuda_available'] else 'NOT available'}")
        for gpu in t["gpus"]:
            arch = gpu.get("architecture", "Unknown")
            print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_gb']}GB) "
                  f"— {arch} sm_{gpu['compute_capability'].replace('.', '')}")
            if "blackwell_notes" in gpu:
                print(f"\n  ⚠️  BLACKWELL DETECTED — IMPORTANT:")
                print(f"  {gpu['blackwell_notes']}\n")

    print("\n[3/5] mamba-ssm (GPU package)...")
    report["mamba_ssm"] = check_mamba_ssm()
    status = "✓ available" if report["mamba_ssm"]["available"] else "✗ not installed"
    print(f"  {status}: {report['mamba_ssm']['note']}")

    print("\n[4/5] causal-conv1d...")
    report["causal_conv1d"] = check_causal_conv1d()
    print(f"  {'✓' if report['causal_conv1d']['available'] else '✗'} "
          f"{report['causal_conv1d'].get('version', report['causal_conv1d'].get('note', ''))}")

    print("\n[5/5] CPU MinimalSSM speed test...")
    try:
        report["cpu_speed"] = speed_test_cpu_ssm()
        for r in report["cpu_speed"]:
            print(f"  seq={r['seq_len']:3d} batch={r['batch']:2d} d={r['d_model']:2d}: "
                  f"{r['ms_per_batch']:6.1f}ms/batch  "
                  f"{r['sequences_per_sec']:5.0f} seq/s  [{r['verdict']}]")
    except Exception as e:
        report["cpu_speed"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    if report["torch"].get("cuda_available"):
        print("\n[+] GPU mamba-ssm speed test...")
        report["gpu_speed"] = speed_test_gpu_ssm()
        if "error" in report["gpu_speed"]:
            print(f"  Skipped: {report['gpu_speed']['error']}")
        elif "skipped" in report["gpu_speed"]:
            print(f"  Skipped: {report['gpu_speed']['skipped']}")
        else:
            g = report["gpu_speed"]
            print(f"  {g['ms_per_batch']:.2f}ms/batch  {g['sequences_per_sec']:.0f} seq/s")

    # Save report
    out = results_dir / "01_env_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Report saved to {out}")
    mamba_ok = report["mamba_ssm"]["available"]
    cuda_ok = report["torch"].get("cuda_available", False)
    print(f"\nStatus:")
    print(f"  CPU MinimalSSM:  ✓ ready (always available)")
    print(f"  GPU mamba-ssm:   {'✓ ready' if mamba_ok else '✗ not available (see notes above)'}")
    print(f"\nNext: python experiments/02_event_classify.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
