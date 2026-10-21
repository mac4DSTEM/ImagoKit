# Parity rows, draft (suite tasks 32 and 33)

For `docs/parity.md` when the Mac session ports the functions; state is `missing` until then. Pins: `tools/precipitation/pins.py` ->
`Tests/ImagoCoreTests/Fixtures/threshold-morphology-pins.json` (task 32) and `regionprops-needle-pins.json` (task 33), scikit-image
0.26.0, NumPy 2.5.4, SciPy 1.18.1. Tolerance wording follows the existing rows: the 0.26.0 wheel ships no tests, so scikit-image
numbers are `assert_allclose` dtype defaults (rtol 1e-7 f64, 1e-5 f32); masks, labels and integers exact; SciPy's measurement tests
use decimal 6.

Facts found while pinning (a port must follow these):
- scikit-image 0.26.0 deprecates `morphology.binary_*`; they still run (default footprint: cross, `mode='ignore'`, the image border does not erode).
- `ndimage.binary_*` default structure is `generate_binary_structure(2, 1)` (cross) and `border_value=0`, so erosion eats the image border. Both are pinned; `erosion_border1` pins the `border_value=1` case.
- `remove_small_objects` / `remove_small_holes` take `max_size=N` in 0.26.0 (removes area <= N); the old `min_size=M` is `max_size=M-1`.
- `axis_major_length` is about 2/sqrt(3) x the length of a thin straight rod; `feret_diameter_max` is the end-to-end extent. They differ on every needle in the pins.

## scikit-image 0.26.0, `filters` (replaces the `threshold_*` row)

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `threshold_otsu` (`nbins` 256 and 64; uint8, uint16, int16, float32, float64) | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | 10 grey images; integer dtypes return exact integer thresholds |
| `threshold_li`, `threshold_yen`, `threshold_triangle`, `threshold_mean`, `threshold_isodata` | missing |  | same | study | same 10 images; `yen`/`triangle` histogram-based |
| `threshold_local` (`block_size` 15, `method` gaussian and mean, `offset`) | missing |  | same | study | 2 images, full threshold surface pinned |
| `apply_hysteresis_threshold` | missing |  | exact mask | study | low/high from Otsu +- 10 % of the range |

## SciPy 1.18.1 `ndimage` and scikit-image 0.26.0 `morphology`, `segmentation`

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `ndimage.binary_erosion`, `_dilation`, `_opening`, `_closing` (cross and square structure, `iterations` 1 and 2, `border_value`) | missing |  | exact masks | study | 13 binary cases incl. border-touching needles |
| `morphology.binary_erosion`, `_dilation`, `_opening`, `_closing` (default footprint) | missing |  | exact masks | study | deprecated in 0.26; `mode='ignore'` |
| `ndimage.binary_fill_holes` (cross and square structure), `generate_binary_structure` (rank 2 and 3) | missing |  | exact | study |  |
| `morphology.remove_small_objects`, `remove_small_holes` (`max_size` 0..40, connectivity 1 and 2) | missing |  | exact masks | study | `max_size` semantics above |
| `segmentation.clear_border` (`buffer_size` 0 and 2) | missing |  | exact masks | study |  |

## scikit-image 0.26.0 `measure`; SciPy 1.18.1 `ndimage.measurements`

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `label` connectivity 1 and 2 (needle masks) | present (0.6.0) | `LabelImage` | exact labels | study | extra needle cases beside `segmentation-pins.json` |
| `regionprops`: `area`, `bbox`, `centroid`, `orientation`, `axis_major_length`, `axis_minor_length` | present (0.6.0) | `RegionProps` | area, bbox exact; floats dtype defaults (rtol 1e-7) | study | re-pinned on needle shapes |
| `regionprops`: `feret_diameter_max` | missing |  | rtol 1e-7; check on the Mac against the 0.26.0 checkout | study | the needle length |
| `regionprops`: `perimeter`, `perimeter_crofton`, `eccentricity`, `solidity`, `equivalent_diameter_area`, `extent`, `euler_number`, `area_filled`, `area_convex` | missing |  | rtol 1e-7; integers exact | study | 251 objects at connectivity 2, connectivity 1 pinned too |
| `regionprops` `intensity_mean/min/max/std`, `centroid_weighted` | missing |  | rtol 1e-7 | study | one 64x64 image, 4 labels |
| `ndimage.sum`, `mean`, `standard_deviation`, `minimum`, `maximum`, `center_of_mass`, `minimum_position`, `maximum_position`, `find_objects` | missing |  | scipy/ndimage/tests/test_measurements.py: assert_array_almost_equal (decimal 6); positions and slices exact | study | same image as above |
