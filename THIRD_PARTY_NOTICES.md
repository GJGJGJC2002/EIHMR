# Third-Party Notices

EIHMR's own code is released under the MIT License (see `LICENSE`). This repository also contains,
is derived from, or requires third-party components that are licensed under their own terms, listed
below. Where the terms differ, the third-party terms govern the corresponding files.

**Important:** several required components (GVHMR, PromptHMR, SMPL/SMPL-X) are licensed for
non-commercial research use only. Running the full EIHMR pipeline is therefore restricted to
non-commercial research use, regardless of the MIT License on EIHMR's own code.

## Code included in this repository

### TRAM — MIT License

Most of `lib/` was derived from a TRAM working tree (see `source_manifest.json`).
<https://github.com/yufu-wang/tram>

```
MIT License

Copyright (c) 2024 Yufu Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### GVHMR — GVHMR License (educational, research and non-profit use only)

Applies to `lib/gvhmr/` (vendored body-model, kinematics, geometry and rendering code).
<https://github.com/zju3dv/GVHMR>

```
Copyright 2022-2023 3D Vision Group at the State Key Lab of CAD&CG,
Zhejiang University. All Rights Reserved.

For more information see <https://github.com/zju3dv/GVHMR>
If you use this software, please cite the corresponding publications
listed on the above website.

Permission to use, copy, modify and distribute this software and its
documentation for educational, research and non-profit purposes only.
Any modification based on this work must be open-source and prohibited
for commercial use.
You must retain, in the source form of any derivative works that you
distribute, all copyright, patent, trademark, and attribution notices
from the source form of this work.


For commercial uses of this software, please send email to xwzhou@zju.edu.cn
```

The files in `lib/gvhmr/assets/` (joint regressors, SMPL-X to SMPL mapping, vertex segmentation) are
derived from the SMPL / SMPL-X body models and are additionally subject to the SMPL / SMPL-X licenses
(see below).

### PyTorch3D — BSD 3-Clause License

Applies to `lib/utils/rotation_conversions.py`.
<https://github.com/facebookresearch/pytorch3d>

```
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### PromptHMR — PromptHMR License (non-commercial scientific research use only)

The PromptHMR integration in `lib/pipeline/backends/hpe/` (`prompthmr.py`, `vitpose_compat.py`)
loads and runs a separately obtained PromptHMR checkout. Any portions of this repository derived
from PromptHMR remain subject to the PromptHMR License, which restricts use to non-commercial
scientific research and requires derivative works to be distributed under the same restrictions.
PromptHMR itself, its weights and its bundled components are not redistributed here.
<https://github.com/yufu-wang/PromptHMR>

## Components that must be obtained separately

These are not included in this repository. Each is governed by its own license, which you must
review and accept before use:

| Component | Used for | License (see upstream) |
|---|---|---|
| PromptHMR (code, weights, bundled Metric3D / ViTPose) | human mesh recovery | PromptHMR License, non-commercial |
| SMPL / SMPL-X body models | body models | SMPL / SMPL-X licenses, non-commercial |
| DROID-SLAM (`droid_backends`, `lietorch`) | camera tracking | BSD 3-Clause |
| PyTorch3D | geometry / rendering | BSD 3-Clause |
| Segment Anything (SAM) weights | person masks | Apache 2.0 |
| ZoeDepth / MiDaS | metric scale | MIT |
| Ultralytics YOLO (`ultralytics` package and weights) | person detection | AGPL-3.0 |
| Camera calibration checkpoint (`camcalib_sa_biased_l2.ckpt`) | intrinsics | see its source |
| EMDB dataset | evaluation | EMDB license, non-commercial |

Python packages installed from `requirements.txt` are distributed under their respective licenses.
