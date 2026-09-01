import json
import subprocess
import tempfile
from pathlib import Path

import torch

import modal

app = modal.App("gns-ampere-symmetric-cutlass")
kernel_cache = modal.Volume.from_name(
    "gns-ampere-kernel-cache-v2",
    create_if_missing=True,
)
source_root = Path(__file__).parents[1]

benchmark_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.13",
    )
    .apt_install("git")
    .uv_pip_install(
        "torch==2.11.0",
        "beartype>=0.22.9",
        "einops>=0.8.2",
        "jaxtyping>=0.3.11",
        "flashinfer-python==0.6.13",
    )
    .uv_pip_install(
        "gram-newton-schulz @ git+https://github.com/Dao-AILab/gram-newton-schulz.git@e45d0aca7083cb275c9a303220c05c4abecd9187",
        extra_options="--no-build-isolation",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "FLASHINFER_JIT_CACHE_DIR": "/root/.cache/flashinfer",
            "PYTHONPATH": "/root",
        }
    )
    .add_local_dir(
        str(source_root / "gram_newton_schulz_ampere"),
        "/root/gram_newton_schulz_ampere",
        copy=True,
    )
    .add_local_file(
        str(Path(__file__).with_name("benchmark_symmetric_cutlass.py")),
        "/root/benchmark_symmetric_cutlass.py",
        copy=True,
    )
)


def run_benchmark_process(arguments: list[str]) -> dict[str, object]:
    completed_process = subprocess.run(
        ["python", "/root/benchmark_symmetric_cutlass.py", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": completed_process.returncode,
        "stdout": completed_process.stdout,
        "stderr": completed_process.stderr,
    }


def benchmark_device(batch_sizes: tuple[int, ...]) -> dict[str, object]:
    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "batches": {},
    }
    batch_results: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        for batch_size in batch_sizes:
            output_path = temporary_path / f"batch-{batch_size}.json"
            process_result = run_benchmark_process(
                [
                    "--batch-sizes",
                    str(batch_size),
                    "--warmups",
                    "5",
                    "--repeats",
                    "20",
                    "--output",
                    str(output_path),
                ]
            )
            if process_result["returncode"] == 0:
                process_result["payload"] = json.loads(output_path.read_text())
            batch_results[str(batch_size)] = process_result
    results["batches"] = batch_results
    return results


@app.function(
    image=benchmark_image,
    gpu="A10",
    cpu=8,
    memory=16384,
    timeout=1800,
    volumes={"/root/.cache": kernel_cache},
)
def benchmark_sm86(
    batch_sizes: tuple[int, ...] = (1, 8, 32),
) -> dict[str, object]:
    return benchmark_device(batch_sizes)


@app.function(
    image=benchmark_image,
    gpu="A100-80GB",
    cpu=8,
    memory=8192,
    timeout=1800,
    volumes={"/root/.cache": kernel_cache},
)
def benchmark_sm80(
    batch_sizes: tuple[int, ...] = (1, 8, 32),
) -> dict[str, object]:
    return benchmark_device(batch_sizes)
