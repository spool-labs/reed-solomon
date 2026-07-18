# Third-Party Notices

`tape-reed-solomon` is licensed under Apache-2.0. It contains portions that are
a faithful port ("fork") of third-party code, reproduced here with attribution.

## reed-solomon-erasure (MIT)

The GF(2^8) field tables and arithmetic (`src/galois.rs`), the matrix
construction/inversion (`src/matrix.rs`), and the encode/verify/reconstruct
control flow (`src/reedsolomon.rs`) are ported from **reed-solomon-erasure**
version 6.0.0.

- Project: https://github.com/darrenldl/reed-solomon-erasure
- Copyright (c) 2016-2017 Darren Ldl and the reed-solomon-erasure contributors
- License: MIT

```
MIT License

Copyright (c) 2016-2017 Darren Ldl and the reed-solomon-erasure contributors

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

## SIMD kernels (original)

The GF(2^8) SIMD kernels under `src/gf/` (scalar reference plus the wasm128,
neon, and x86 kernels) are written from public technique — the nibble-split
`pshufb`/swizzle multiply and the GFNI `gf2p8affineqb` affine map documented in
Plank et al., Intel ISA-L, and Intel's GFNI whitepaper. The wasm128 kernel
derives from tape's own clean-room `gfsimd` prototype.
