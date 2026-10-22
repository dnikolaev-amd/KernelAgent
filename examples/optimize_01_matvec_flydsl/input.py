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

"""FlyDSL matvec for C = A @ B, with B shaped (K, 1).

One wavefront lane owns one row and accumulates in fp32. This is the starting
kernel for the ROCm FlyDSL search, not a tuned kernel.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu

BLOCK = 64


@flyc.kernel
def _matvec(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, M: fx.Int32, K: fx.Int32):
    row = gpu.block_idx.x * BLOCK + gpu.thread_idx.x
    if row < M:
        acc = fx.Float32(0.0)
        for k in range(K):
            acc = acc + fx.Float32(A[row, k]) * fx.Float32(B[k, 0])
        C[row, 0] = fx.BFloat16(acc)


@flyc.jit
def _launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, M: fx.Int32, K: fx.Int32):
    _matvec(A, B, C, M, K).launch(
        grid=((M + BLOCK - 1) // BLOCK,),
        block=(BLOCK,),
    )


def kernel_function(A, B):
    C = torch.empty(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)
    _launch(A, B, C, A.shape[0], A.shape[1])
    return C
