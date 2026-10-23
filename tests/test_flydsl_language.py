# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from pathlib import Path

import pytest

from triton_kernel_agent.opt_manager import OptimizationManager
from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.prompt_manager import PromptManager, check_kernel_language
from triton_kernel_agent.worker import VerificationWorker, flydsl_language_violation

_REPO = Path(__file__).resolve().parents[1]
_FLYDSL_RULES = ("fx.Int32", "A.shape", "wavefront")

_GPU = {
    "name": "AMD Instinct MI300X",
    "architecture": "gfx942",
    "peak_memory_bw_gbps": 5300,
    "peak_fp32_tflops": 1,
    "peak_fp16_tflops": 1,
    "peak_bf16_tflops": 1,
    "sm_count": 304,
    "max_threads_per_sm": 2048,
    "l1_cache_kb": 32,
    "l2_cache_mb": 256,
    "memory_gb": 192,
    "memory_type": "HBM3",
}
_ROOFLINE = {
    "bottleneck": "memory",
    "compute_sol_pct": 1.0,
    "memory_sol_pct": 2.0,
    "efficiency_pct": 3.0,
    "headroom_pct": 4.0,
    "at_roofline": False,
    "uses_tensor_cores": False,
    "warnings": [],
}


def _opt_prompt(language: str) -> str:
    manager = PromptManager(
        target_platform=get_platform("rocm"),
        kernel_language=language,
    )
    return manager.render_kernel_optimization_prompt(
        problem_description="matvec",
        kernel_code="def kernel_function(a, b):\n    return a\n",
        gpu_specs=_GPU,
        roofline=_ROOFLINE,
        category="memory",
        summary="bandwidth",
        reasoning="strided loads",
        root_cause={"cause": "stride", "evidence": []},
        recommended_fix={"fix": "vectorize", "rationale": "coalesce"},
        pytorch_baseline_ms=1.0,
    )


def test_default_prompt_stays_triton():
    text = _opt_prompt("triton")
    assert "@triton.jit" in text
    assert "FlyDSL" not in text


def test_flydsl_prompt_does_not_ask_for_triton():
    manager = PromptManager(
        target_platform=get_platform("rocm"),
        kernel_language="flydsl",
    )
    text = _opt_prompt("flydsl")
    assert "@flyc.kernel" in text
    assert "@triton.jit" not in text
    for rule in _FLYDSL_RULES:
        assert rule in text
    generated = manager.render_kernel_generation_prompt(
        "matvec", "def test_kernel():\n    pass\n"
    )
    assert "@flyc.kernel" in generated
    assert "@triton.jit" not in generated
    for rule in _FLYDSL_RULES:
        assert rule in generated
    refined = manager.render_kernel_refinement_prompt(
        problem_description="matvec",
        test_code="def test_kernel():\n    pass\n",
        kernel_code="def kernel_function(a, b):\n    return a\n",
        error_info={"stdout": "", "stderr": "boom"},
    )
    assert "@flyc.kernel" in refined
    assert "@triton.jit" not in refined
    for rule in _FLYDSL_RULES:
        assert rule in refined


def test_unknown_kernel_language_is_rejected():
    with pytest.raises(ValueError, match="kernel_language"):
        check_kernel_language("FlyDSL")
    with pytest.raises(ValueError, match="kernel_language"):
        PromptManager(target_platform=get_platform("rocm"), kernel_language="fly_dsl")


def test_guidelines_override_reaches_optimization_prompt(tmp_path):
    override = tmp_path / "custom_guidelines.j2"
    override.write_text("CUSTOM_FLYDSL_GUIDELINE\n")
    manager = PromptManager(
        target_platform=get_platform("rocm"),
        kernel_language="flydsl",
        template_overrides={"triton_guidelines": str(override)},
    )
    text = manager.render_kernel_optimization_prompt(
        problem_description="matvec",
        kernel_code="def kernel_function(a, b):\n    return a\n",
        gpu_specs=_GPU,
        roofline=_ROOFLINE,
        category="memory",
        summary="bandwidth",
        reasoning="strided loads",
        root_cause={"cause": "stride", "evidence": []},
        recommended_fix={"fix": "vectorize", "rationale": "coalesce"},
        pytorch_baseline_ms=1.0,
    )
    assert "CUSTOM_FLYDSL_GUIDELINE" in text
    assert "fx.Int32" not in text


def test_amd_flydsl_yaml_selects_flydsl(tmp_path):
    manager = OptimizationManager(
        config=str(_REPO / "examples/configs/amd_flydsl.yaml"),
        gpu_ids=[0],
        log_dir=tmp_path,
    )
    assert manager.worker_kwargs["kernel_language"] == "flydsl"
    guidelines = manager.worker_kwargs["templates"]["triton_guidelines"]
    assert guidelines.endswith("flydsl_guidelines.j2")


def test_manager_sets_flydsl_env_before_initial_verify(tmp_path, monkeypatch):
    order = []

    def ensure():
        order.append("ensure")

    monkeypatch.setattr(
        "triton_kernel_agent.flydsl_env.ensure_flydsl_rocm_env",
        ensure,
    )
    manager = OptimizationManager(
        strategy="beam_search",
        num_workers=1,
        strategy_config={"num_top_kernels": 1, "num_bottlenecks": 1},
        gpu_ids=[0],
        log_dir=tmp_path,
        kernel_language="flydsl",
    )
    manager._verify_initial_kernel = lambda *a, **k: order.append("verify") or True
    manager._benchmark_pytorch_baseline = lambda *a, **k: order.append("eager") or 1.0
    manager._benchmark_pytorch_compile = lambda *a, **k: order.append("compile") or 1.0

    def bench_initial(*a, **k):
        order.append("kernel")
        raise RuntimeError("stop")

    manager._benchmark_initial_kernel = bench_initial
    with pytest.raises(RuntimeError, match="stop"):
        manager.run_optimization(
            initial_kernel="def kernel_function():\n    pass\n",
            problem_file=tmp_path / "problem.py",
            test_code="def test_kernel():\n    pass\n",
        )
    assert order == ["ensure", "verify", "eager", "compile", "kernel"]


def test_flydsl_candidate_must_stay_flydsl():
    triton_kernel = "import triton\n\n@triton.jit\ndef kernel():\n    return\n"
    assert flydsl_language_violation(triton_kernel)

    commented = "# do not use triton here\n@flyc.kernel\ndef kernel():\n    return\n"
    assert flydsl_language_violation(commented) is None

    starter = (_REPO / "examples/optimize_01_matvec_flydsl/input.py").read_text()
    assert flydsl_language_violation(starter) is None


def test_flydsl_worker_rejects_triton_before_running_the_test(tmp_path):
    worker = VerificationWorker(
        worker_id=0,
        workdir=tmp_path,
        log_dir=tmp_path,
        kernel_language="flydsl",
    )
    success, _stdout, stderr, violation = worker._single_verification_pass(
        "import triton\n\n@triton.jit\ndef kernel():\n    return\n"
    )
    assert success is False
    assert violation
    assert "flydsl" in stderr
    assert worker.prompt_manager.kernel_language == "flydsl"


def test_starters_do_not_import_the_package():
    for name in (
        "examples/optimize_01_matvec_flydsl/input.py",
        "examples/optimize_02_rmsnorm_flydsl/input.py",
    ):
        text = (_REPO / name).read_text()
        assert "triton_kernel_agent" not in text


def test_matvec_starter_runs_without_pythonpath():
    pytest.importorskip("flydsl")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("FlyDSL starter check needs a GPU")

    from triton_kernel_agent.flydsl_env import ensure_flydsl_rocm_env

    ensure_flydsl_rocm_env()
    example = _REPO / "examples/optimize_01_matvec_flydsl"
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    script = r"""
import importlib.util
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1]) / "input.py"
spec = importlib.util.spec_from_file_location("starter_kernel", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
a = torch.ones(4, 32, device="cuda", dtype=torch.bfloat16)
b = torch.ones(32, 1, device="cuda", dtype=torch.bfloat16)
got = mod.kernel_function(a, b)
ref = torch.matmul(a.float(), b.float()).to(torch.bfloat16)
if not torch.equal(got, ref):
    raise SystemExit(f"mismatch max={(got.float() - ref.float()).abs().max()}")
print("PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(example)],
        check=False,
        cwd=example,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
