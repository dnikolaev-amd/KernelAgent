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

"""FlyDSL RMSNorm over dim 1 of an NCHW tensor.

One wavefront lane owns one (n, h, w) location, reduces the channels in
fp32, and writes y = x * rsqrt(mean(x^2) + eps). This is the starting
kernel for the ROCm FlyDSL search, not a tuned kernel.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu
from flydsl.expr.math import rsqrt

from triton_kernel_agent.flydsl_env import ensure_flydsl_rocm_env

ensure_flydsl_rocm_env()

BLOCK = 64


@flyc.kernel
def _rmsnorm(
    X: fx.Tensor,
    Y: fx.Tensor,
    N: fx.Int32,
    C: fx.Int32,
    H: fx.Int32,
    W: fx.Int32,
    eps: fx.Float32,
):
    spatial = N * H * W
    tid = gpu.block_idx.x * BLOCK + gpu.thread_idx.x
    if tid < spatial:
        w = tid % W
        tmp = tid // W
        h = tmp % H
        n = tmp // H
        acc = fx.Float32(0.0)
        for c in range(C):
            value = fx.Float32(X[n, c, h, w])
            acc = acc + value * value
        inv_rms = rsqrt(acc / fx.Float32(C) + eps)
        for c in range(C):
            value = fx.Float32(X[n, c, h, w])
            Y[n, c, h, w] = fx.BFloat16(value * inv_rms)


@flyc.jit
def _launch(
    X: fx.Tensor,
    Y: fx.Tensor,
    N: fx.Int32,
    C: fx.Int32,
    H: fx.Int32,
    W: fx.Int32,
    eps: fx.Float32,
):
    spatial = N * H * W
    _rmsnorm(X, Y, N, C, H, W, eps).launch(
        grid=((spatial + BLOCK - 1) // BLOCK,),
        block=(BLOCK,),
    )


def kernel_function(x, eps: float = 1e-5):
    y = torch.empty_like(x)
    n, c, h, w = x.shape
    _launch(x, y, n, c, h, w, float(eps))
    return y
