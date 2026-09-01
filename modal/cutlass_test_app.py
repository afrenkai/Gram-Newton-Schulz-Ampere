import json
import subprocess
from pathlib import Path

import modal

app = modal.App("gns-ampere-cutlass-tests")
source_root = Path(__file__).parents[1]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.13",
    )
    .uv_pip_install(
        "torch==2.11.0",
        "beartype>=0.22.9",
        "einops>=0.8.2",
        "jaxtyping>=0.3.11",
        "flashinfer-python==0.6.13",
        "pytest==9.1.1",
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
        str(source_root / "tests" / "test_cutlass_ns.py"),
        "/root/tests/test_cutlass_ns.py",
        copy=True,
    )
)


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=16384,
    timeout=1800,
)
def test_sm86() -> dict[str, object]:
    completed = subprocess.run(
        ["python", "-m", "pytest", "-q", "/root/tests/test_cutlass_ns.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


@app.local_entrypoint()
def main() -> None:
    result = test_sm86.remote()
    print(json.dumps(result, indent=2))
    if result["returncode"] != 0:
        raise RuntimeError("SM86 CUTLASS tests failed")
