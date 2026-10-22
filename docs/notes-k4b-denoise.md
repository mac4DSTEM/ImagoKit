# Notes for the merge: slice K4b, scikit-image denoising (2D, one channel)

Branch work only; the merging session folds these into `CHANGELOG.md`, `docs/parity.md`, `docs/status.md`, `docs/open-items.md`.

## CHANGELOG bullet (under Added)

- **Denoising** (`Denoise`, ImagoFilters): `totalVariationChambolle`, `totalVariationBregman` and `nonLocalMeans` for a 2D single-channel
  image on Float32 and Float64 (a float32 image stays float32), with scikit-image 0.26.0's names and defaults (`weight`, `eps`,
  `maxNumIter`, `isotropic`; `patchSize`, `patchDistance`, `h`, `fastMode`, `sigma`). Pinned against `denoise_tv_chambolle`,
  `denoise_tv_bregman` and `denoise_nl_means` (`tools/denoise/pins.py`, 536 cases on 12 images in both dtypes): Chambolle and
  non-local means (both modes) are bit-identical to the compiled package, Bregman agrees to 1e-14 (Float64) and 2.7e-6 (Float32)
  relative. Refuses a size mismatch, NaN or infinite samples, non-positive parameters (`DenoiseError`), and, for Bregman, an axis of
  length 1 (the compiled kernel reads outside the image there).

## docs/parity.md rows (replace the two `restoration` denoise rows at lines 133 and 134; the rest of line 134 stays missing)

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `denoise_tv_chambolle` (2D, one channel, `weight`, `eps`, `max_num_iter`) | present | `Denoise.totalVariationChambolle` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | 144 pins; bit-identical in both dtypes (NumPy pairwise sums replayed so the stop test sees the same energies); no `channel_axis`, no nD, no integer images |
| `denoise_tv_bregman` (2D, one channel, `weight`, `max_num_iter`, `eps`, `isotropic`) | present | `Denoise.totalVariationBregman` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | 108 pins; 1.0e-14 relative (f64), 2.7e-6 (f32); images with an axis of length 1 are refused (the compiled kernel reads outside the array there); no `channel_axis`, no integer images |
| `denoise_nl_means` (2D, one channel, `patch_size`, `patch_distance`, `h`, `fast_mode`, `sigma`) | present | `Denoise.nonLocalMeans` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | 284 pins; bit-identical in both modes and dtypes; the weight is the compiled kernel's Schraudolph exp approximation (see deviations below); `patch_size` 1 is refused; no `channel_axis`, no 3D/4D, no `preserve_range`, no integer images |

The line "`denoise_nl_means`, `denoise_bilateral`, ..." then loses its first entry and stays **missing** for the others. Counts: 3 rows
move to present (the Chambolle/Bregman row is split in two, so the table grows by one row), totals go from 87 rows to 88, present from
27 to 30.

## Measured agreement (for the test header and status)

Largest differences over all pins of a group (`denoiseAgreementReport` prints them):

| Group | Float64 | Float32 |
|---|---|---|
| Chambolle (72 pins per dtype) | 0 (bit-identical) | 0 (bit-identical) |
| Bregman (54) | 9.9e-15 relative, 2.3e-12 absolute | 2.7e-6 relative, 1.2e-3 absolute (the 1000-scale image) |
| NL-means fast and exact (142) | 0 (bit-identical) | 0 (bit-identical) |

Tolerance used in the tests: `assert_allclose` dtype defaults (rtol 1e-7 Float64, 1e-5 Float32, atol 0), as the task and
`docs/parity.md` prescribe.

## What the compiled scikit-image kernels do (and the kit reproduces)

`_denoise_tv_chambolle_nd` is NumPy and was ported line by line. `_denoise_tv_bregman` and `_nl_means_denoising` are compiled Cython with
no source on disk; they were recreated from the algorithm, then corrected against the compiled package and, where a probe could not
settle a detail, against the arm64 disassembly of the shipped `.so` (`otool -tV`). Things that are not the textbook algorithm:

1. **NL-means weights use a Schraudolph-style exp**, not `exp`. The kernel computes `Int32(-d * 2^20/ln 2)` (saturating, truncating),
   adds 1072632447 with 32-bit wrap-around and stores it as the high word of a double. The function is piecewise linear in d, halves
   its slope every ln 2, and gives 0.9711 at d = 0 (not 1). This was the cause of a 1e-3 mismatch that no tap-weight fit could explain;
   `.so` imports no `exp` at all. Distances above 5 are dropped (exact mode: tested at the start of each patch row, so a pair whose
   last row pushes the distance past 5 still gets a weight; fast mode: tested on the finished distance).
2. **Exact mode** (`fast_mode=False`): Gaussian patch weights with sigma (s-1)/4 normalised to sum 1/h^2, built in the image dtype with
   NumPy pairwise sums; the search window is clipped to the image; distance accumulation uses fused multiply-adds
   (`w * (diff^2 - 2 var)` then `distance + ...`), so the kit uses `addingProduct` in the same order; the result is accumulated with
   fused multiply-adds too. Float32 images compute in float32 throughout.
3. **Fast mode** (`fast_mode=True`): always computed in **Float64** (a float32 image is widened and rounded once at the end), with
   integral images. The patch it compares is `s - 1` pixels wide, rows and columns `pixel - s/2 + 1 ... pixel + s/2`, but normalised by
   `s^2`; it is not the symmetric s x s patch of the exact mode. Pairs are accumulated symmetrically (`alpha` 0.5 for `t_col == 0`).
4. **Borders** are `np.pad(..., mode='reflect')` of `s/2 + patch_distance + 1`, and NumPy's reflect is not the periodic triangle wave
   when the pad is wider than the image (it reflects repeatedly around the grown block; an axis of length 1 is filled with its value).
   The kit replays NumPy's index bookkeeping, checked against `np.pad` for n = 1...8 and pads 0...39.
5. **Bregman**: the Gauss-Seidel update uses `dx`, `bx` along the row index and `dy`, `by` along the column index, the border is the
   reflection of the second row/column (`image[1]`, `image[rows-1]`, ...), `dx, dy, bx, by` start at zero.

## Deviations (all refusals; no number differs where both run)

- API labels are `rows:` and `columns:` (the kit's rule), not `width:`/`height:`; the data is row-major as everywhere in the kit.
- `totalVariationBregman` refuses an image with fewer than 2 rows or 2 columns (`DenoiseError.tooSmall`): the compiled kernel reads
  `image[1, :]` / `image[:, 1]` to reflect the border and so reads outside a 1-pixel axis (undefined values; scikit-image returns
  something, but not reproducibly).
- `nonLocalMeans` refuses `patchSize` below 2 (with 1 the Gaussian window has sigma 0 and the exact mode divides 0 by 0; the fast mode
  compares empty patches), and `patchSize`/`patchDistance` above 1024.
- Infinite samples are refused as well as NaN (scikit-image returns NaN).
- Non-positive `weight`, `eps`, `maxNumIter`, `h`, negative `sigma` are refused (scikit-image accepts some of them and returns
  garbage or runs all iterations).
- For a 1 x N or N x 1 input scikit-image's `denoise_nl_means` returns a squeezed 1-D array; the kit returns the row-major samples
  of the same count (the pins compare flat values).
- The exact mode's treatment of huge distances is reproduced including the reference's wrap-around: a distance far above 5 that only
  appears in the last patch row (image values large against `h`) gives an arbitrary weight in scikit-image, and the same here (a
  1000-scale image with h = 0.1 is pinned and bit-identical).

## Open items

- `Tests/ImagoFiltersTests` has no test target in `Package.swift` (only a `Fixtures` folder, which the pins use through
  `FixtureDirs.fixtures(of: "ImagoFiltersTests")`). The closed-form and refusal tests therefore sit in
  `Tests/ImagoKitParityTests/Filters/DenoiseClosedFormTests.swift`; if the merging session adds a fast `ImagoFiltersTests` target,
  move that file there (they do not need fixtures).
- Not done by scope: `channel_axis`, 3D/4D, integer images, `preserve_range`, `denoise_bilateral`, wavelets, invariant denoisers.
- Performance: all three run on the CPU in the image's precision (the reference kernels are scalar); exact-mode NL-means is
  O(pixels x (2 patchDistance + 1)^2 x patchSize^2) like the reference. A Metal or vectorised path is not attempted
  (suite rule D16: only where size pays); the 512 x 512 default `fastMode` call has not been benchmarked.
- The pins use images up to 16 x 18; a larger planted-noise image with a known clean truth (for a `validation` badge on real
  denoising results) would be the next check.
- The Schraudolph weights and the s-1 fast-mode patch are properties of scikit-image 0.26.0's compiled kernels; if a later release
  fixes either, the pins and this port change together.
