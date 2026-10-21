# Copyright (c) Meta Platforms, Inc. and affiliates.
#
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

from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.prompt_manager import PromptManager

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
    generated = manager.render_kernel_generation_prompt(
        "matvec", "def test_kernel():\n    pass\n"
    )
    assert "@flyc.kernel" in generated
    assert "@triton.jit" not in generated
