# Parity with the references' documented API

The checklist that makes "complete" a measured claim: one row per function or option group of the Python counterparts
a micrograph user reaches for, walked from the sources on disk (the pinned environment's site-packages: NumPy 2.5.4, SciPy
1.18.1, scikit-image 0.26.0; run 2026-10-21). States: **present** (ported, pinned, tested), **partial** (a subset, named),
**missing**. Priority: **study** (on Paul's path: the TEM study and the two apps' open kit asks), **useful**, **not planned**
(with the reason). The tolerance column is the one the counterpart's own test file uses for that function; the
scikit-image 0.26.0 wheel ships no test files, so there its functions are held to `assert_allclose` dtype defaults.
Numerics are never held stricter than that; readers, I/O, indices and metadata are exact.

**Totals: 93 rows of the Python counterparts: 36 present, 12 partial, 45 missing.** By priority: 39 study, 34 useful, 20 not planned. (At 0.7.0: 87 rows, 27 present, 11 partial, 49 missing; at 0.6.0: 77 rows, 17 present, 13 partial, 47 missing; at 0.5.0: 32 rows, 17 present, 8 partial, 7 missing.) The kit-only rows at the end are not counted.


## NumPy 2.5.4, `numpy.fft`

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `fft2`, `ifft2` on a real or complex 2D array, any shape | present | `FFTSurface.fft2/ifft2`, `FFT2D`, `FFT2DDouble` | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | study | exact N-point DFT for every length, Float32 and Float64; 81 pinned shape/axes/norm cases; Float64 lengths of prime factors 2, 3, 5, 7 not served by vDSP run on a mixed-radix plan (43 pinned lengths and 5 shapes, worst scaled difference 4e-14), Bluestein for the rest |
| `axes=` and `norm=` (backward, forward, ortho) on `fft2`/`ifft2` | present (0.6.0) | `FFTAxes`, `FFTNorm` | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | study | axes are axis 0, axis 1 or both |
| `s=` (resize before transforming) on `fft2`/`ifft2` | missing |  | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | useful | zero-pad or crop by the caller today |
| `rfft2`, `irfft2` (with `s`, `norm`) | present (0.8.0 real-input path) | `FFTSurface.rfft2/irfft2` | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | study | real-input path (two real rows per complex transform; 0.72 ms Float64 at 512x512, NumPy 0.65); 24 + 16 pinned shape/norm cases incl. `s` truncation and padding; worst scaled difference 9e-14 (Float64), 1.4e-5 (Float32; the test holds it to 1e-5 of the array maximum) |
| `fftshift`, `ifftshift` (2D, `axes`) | present (0.6.0) | `FFTSurface.fftshift/ifftshift` | numpy/fft/tests/test_helper.py: assert_array_almost_equal (decimal 6) | study | any element type; exact |
| `fftfreq` | present | `FFT2D.fftfreq`, `PowerSpectrum.frequencies` | exact (NumPy: test_helper.py decimal 6) | study |  |
| `rfftfreq` | missing |  | numpy/fft/tests/test_helper.py: assert_array_almost_equal (decimal 6) | useful | one line |
| `fft`, `ifft` on a complex 1D array | partial | `FFT1D`, `FFT1DDouble` | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | useful | radix-2 positive lengths only, no `axis=`, `n=`, `norm=`; the Double 1D runs on `vDSP_DFT_zop` |
| `rfft`, `irfft` (1D real; `n=`, `norm=`) | present (0.8.0) | `FFT1D.rfft/irfft`, `FFT1DDouble.rfft/irfft` (plan and static forms) | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | useful | power-of-two lengths only (the 1D plans'); `n=` is the plan length (static forms take `n:`); 94 pinned cases, worst scaled difference 1.9e-14 (Float64), 6.8e-6 (Float32) |
| `hfft`, `ihfft` (1D Hermitian) | missing |  | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | useful | ACOM (DiffractoKit) needs a 1D FFT of any length |
| `fftn`, `ifftn`, `rfftn`, `irfftn` (n-dimensional) | missing |  | numpy/fft/tests/test_pocketfft.py: atol 1e-6 (rtol 1e-7) | not planned | images are 2D; 3D/4D transforms belong to the consumers that need them |

## NumPy 2.5.4, other

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `histogram(a, bins=int, range=...)` on a float32 array | present | `IntensityHistogram.histogram` | exact counts (numpy/lib/tests/test_histograms.py) | study | array/string `bins`, `weights`, `density`, float64 input absent |
| `hanning(n)` | present | `SpotMeasurement.hanning` | numpy/lib/tests/test_function_base.py: assert_almost_equal (decimal 7) | study |  |

## SciPy 1.18.1, `scipy.ndimage` filters

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `gaussian_filter` order 0: `sigma` per axis, `truncate`, `mode` (reflect, constant, nearest, mirror, wrap), `cval` | present (0.6.0) | `SeparableFilters.gaussianFilter` | scipy/ndimage/tests/test_filters.py: assert_array_almost_equal (decimal 6) | study | Float32/Float64 (intermediate rounded to the input dtype as SciPy does); `grid-constant`/`grid-wrap` are the same as `constant`/`wrap` in a filter; 140 pinned cases |
| `gaussian_filter` derivative `order` 1 to 4, `radius=`, `axes=` | missing |  | scipy/ndimage/tests/test_filters.py: assert_array_almost_equal (decimal 6) | useful | gradients for GPA and edge work |
| `gaussian_filter1d`, `uniform_filter1d`, `correlate1d`, `convolve1d` | missing |  | scipy/ndimage/tests/test_filters.py: assert_array_almost_equal (decimal 6) | useful | the 2D separable passes exist privately |
| `uniform_filter` (`size` per axis, five modes, `cval`) | present (0.6.0) | `SeparableFilters.uniformFilter` | scipy/ndimage/tests/test_filters.py: assert_array_almost_equal (decimal 6) | study | origin 0 only; running sum; 120 pinned cases |
| `median_filter`, `percentile_filter`, `rank_filter` | partial | `HRTEMFilter` step A1 | scipy/ndimage/tests/test_filters.py: assert_array_almost_equal (decimal 6) | useful | only 3x3 nearest inside the HRTEM filter; no public general-size or footprint call |
| `maximum_filter`, `minimum_filter` (+1d) | partial | `PeakLocalMax` (private window maximum) | scipy/ndimage/tests/test_filters.py: assert_array_almost_equal (decimal 6) | useful | square window or a `PeakFootprint`, nearest mode, inside `peak_local_max` only |
| `correlate`, `convolve` (n-dimensional kernels) | missing |  | scipy/ndimage/tests/test_filters.py: assert_array_almost_equal (decimal 6) | useful |  |
| `sobel`, `prewitt`, `laplace`, `gaussian_laplace`, `gaussian_gradient_magnitude`, `generic_laplace`, `generic_gradient_magnitude` | missing |  | scipy/ndimage/tests/test_filters.py: assert_array_almost_equal (decimal 6) | useful |  |
| `generic_filter`, `generic_filter1d`, `vectorized_filter` | missing |  | scipy/ndimage/tests/test_filters.py: assert_array_almost_equal (decimal 6) | not planned | arbitrary Python callbacks have no Swift counterpart |
| `fourier_gaussian`, `fourier_uniform`, `fourier_ellipsoid` | missing |  | scipy/ndimage/tests/test_fourier.py: assert_array_almost_equal (decimal 6) | useful | Fourier-domain filters |
| `fourier_shift` | partial | `ImageShift.fourierShift`, `fourierShiftWithResidual` | scipy/ndimage/tests/test_fourier.py: decimal 6 | study | real(ifft2(fourier_shift(fft2(x), s))) of a real 2D image; `n=` and `axis=` (real-FFT input) absent |

## SciPy 1.18.1, `scipy.ndimage` interpolation

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `shift`, order 0 and 1, seven modes, `cval` | present | `ImageShift.shift` | scipy/ndimage/tests/test_interpolation.py: assert_array_almost_equal (decimal 6) | study | 2D, one real channel; `output=` absent |
| `shift`, order 2 to 5 (B-spline, `prefilter=True`) | partial (0.6.0) | `ImageShift.shift` with `ShiftOrder.quadratic…quintic` | scipy/ndimage/tests/test_interpolation.py: assert_array_almost_equal (decimal 6) | study | modes `grid-constant`, `nearest`, `reflect`, `mirror`, `grid-wrap` (500 pinned cases, held to decimal 6 = 1.5e-6 in both precisions since 0.6.1); `constant` and `wrap` are refused by name; `prefilter=False` absent |
| `map_coordinates` | partial | `LineProfile` (orders 0 to 5, five modes, line samples), `HRTEMFilter.radialProfile` | scipy/ndimage/tests/test_interpolation.py: assert_array_almost_equal (decimal 6) | useful | no public general coordinate array call |
| `affine_transform`, `geometric_transform` | missing |  | scipy/ndimage/tests/test_interpolation.py: assert_array_almost_equal (decimal 6) | useful |  |
| `zoom` (2D, `grid_mode`, orders 0 to 5, `prefilter=True`) | present (0.8.0) | `ImageResampling.zoom` | scipy/ndimage/tests/test_interpolation.py: assert_array_almost_equal (decimal 6) | study | modes `constant`, `grid-constant`, `nearest`, `reflect`, `mirror`, `grid-wrap` (`wrap` at orders 0 and 1 only); float32 pins held to 1.5e-6; `output=`, `prefilter=False`, complex and N-D missing |
| `rotate` (2D, `reshape`, orders 0 to 5) | present (0.8.0) | `ImageResampling.rotate` | scipy/ndimage/tests/test_interpolation.py: assert_array_almost_equal (decimal 6) | study | as `zoom`; the cosine and sine are Cephes' `cosdg`/`sindg` and the coordinates use fused multiply-add as SciPy's build does (an order-0 sample on an exact half pixel depends on it); `axes=`, N-D missing; 476 zoom and rotate cases pinned |
| `spline_filter`, `spline_filter1d` | partial | `LineProfile.SplineField` (package) | scipy/ndimage/tests/test_interpolation.py: assert_array_almost_equal (decimal 6) | useful | used by `profile_line` and spline shifts, not public |

## SciPy 1.18.1, `scipy.ndimage` measurements and morphology

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `label` (structure 4/8-connectivity) | present | `Segmentation.label` | scipy/ndimage/tests/test_measurements.py: assert_array_almost_equal / exact labels | study | 2D only |
| `sum`, `mean`, `standard_deviation`, `minimum`, `maximum`, `minimum_position`, `maximum_position`, `center_of_mass`, `find_objects` over a `LabelImage` (all objects at once, any same-sized image) | present (0.7.0) | `ObjectStatistics` | scipy/ndimage/tests/test_measurements.py: assert_array_almost_equal (decimal 6); positions and slices exact | study | 2D, labels 1...count; one pinned 64x64 image, 4 labels; ties for a position take the first row-major pixel |
| `variance`, `median`, `extrema`, `histogram`, `labeled_comprehension`, `value_indices`, an arbitrary `index=` list | missing |  | scipy/ndimage/tests/test_measurements.py: assert_array_almost_equal (decimal 6) | useful | `variance` is the square of `standardDeviation` |
| `binary_erosion`, `binary_dilation`, `binary_opening`, `binary_closing`, `binary_fill_holes`, `generate_binary_structure` (2D structures of odd size, `iterations` incl. until-stable, `border_value`; any rank for the structure) | present (0.7.0) | `BinaryMorphology`, `StructuringElement` | scipy/ndimage/tests/test_morphology.py: exact (boolean) | study | 13 planted masks (needles, ring, holes, noise) x cross and square, pinned against SciPy 1.18.1; `mask=`, `output=`, `origin=` and even structures are not ported |
| `binary_propagation`, `binary_hit_or_miss`, `iterate_structure` | missing |  | scipy/ndimage/tests/test_morphology.py: exact (boolean) | useful | no consumer yet |
| `grey_erosion`, `grey_dilation`, `grey_opening`, `grey_closing`, `morphological_gradient`, `morphological_laplace`, `white_tophat`, `black_tophat` | missing |  | scipy/ndimage/tests/test_morphology.py: exact (boolean) / decimal 6 | not planned | no consumer yet (`white_tophat` is a background tool; revisit with `rolling_ball`) |
| `distance_transform_edt` (2D, `sampling`, `return_indices`) | present (0.8.0) | `DistanceTransform.euclidean`, `euclideanWithIndices` | scipy/ndimage/tests/test_morphology.py: decimal 6 (held to 1e-12: exact squared distances); indices exact | study | an image with no background pixel is refused (SciPy returns distances to a point outside the image); `return_indices` is SciPy's Voronoi feature transform with its tie rule and the fused multiply-adds of the macOS arm64 build, which decide exact ties under a non-dyadic `sampling` (33 cases, `tools/edt-indices`, many ties); `return_distances=False`, `distances=`/`indices=` out arrays and N-D missing |
| `distance_transform_cdt`, `_bf`, `watershed_ift` | missing |  | scipy/ndimage/tests/test_morphology.py: exact (boolean) / decimal 6 | not planned | no consumer yet |

## scikit-image 0.26.0, `filters`

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `butterworth` (`cutoff_frequency_ratio`, `high_pass`, `order`, `squared_butterworth`, `npad`) | present (0.6.0) | `BandpassFilters.butterworth` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | 0.26.0's `npad` default is 0, not 64; `cutoffFrequencyRatio` 0 is refused (scikit-image answers NaN); `channel_axis` as `channels:` (channel-last, each channel alone; 16 more pins); 120 pinned cases; on the real-input transform (`irfftn(mask * rfftn(x))` as scikit-image), the mask built on the half spectrum |
| `difference_of_gaussians` (`low_sigma`, `high_sigma`, `mode`, `cval`, `truncate`) | present (0.6.0) | `BandpassFilters.differenceOfGaussians` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | `channel_axis` as `channels:` (channel-last, each channel alone; 9 pins); high < low refused |
| `window` (2D shape) | partial (0.6.0; all families since the next minor) | `Window.window2D`, `Window.window1D` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32); the 1D windows of the 14 later families at scipy/signal/tests/test_windows.py per family (rtol 4 sqrt(eps) by default; barthann rtol and atol 1e-15, exponential rtol 1e-14, cosine and general_gaussian atol 1e-12, lanczos atol 1e-9, chebwin atol 1e-8, dpss atol 1e-7), their 2D windows at the f64 default widened by that tolerance | study | every `get_window` family: boxcar, hann, hamming, blackman, blackmanharris, nuttall, flattop, bartlett, tukey, gaussian, kaiser, triang, parzen, bohman, barthann, cosine, exponential, taylor, lanczos, kaiser_bessel_derived, general_cosine, general_gaussian, general_hamming, dpss (`norm='approximate'`, one window), chebwin; the name aliases and `_periodic` suffixes, other `warp_kwargs`, n-D shapes absent |
| `gaussian` (`sigma`, `mode='nearest'`, `truncate`) | present (0.6.0) | `SeparableFilters.gaussianFilter(mode: .nearest)` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | same code as `ndimage.gaussian_filter`; `preserve_range`, `channel_axis` absent |
| `median`, `unsharp_mask` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful |  |
| `sobel`, `prewitt`, `scharr`, `farid`, `roberts` and their `_h`/`_v`/diagonal variants, `laplace` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | gradient kernels |
| `threshold_otsu`, `_li`, `_yen`, `_triangle`, `_mean`, `_isodata`, `_local` (gaussian and mean), `apply_hysteresis_threshold` | present (0.7.0) | `Thresholding` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32); integer images exact | study | 10 planted grey images (Float64, Float32, uint8, uint16, int16); the histogram follows NumPy's uniform-bin `histogram` and Float32 counts, the Float32 image computes in Float32; `_local` generic and median methods not ported |
| `threshold_minimum`, `_multiotsu`, `_niblack`, `_sauvola`, `try_all_threshold` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | no consumer yet |
| `frangi`, `meijering`, `sato`, `hessian`, `gabor`, `gabor_kernel`, `rank`, `rank_order`, `correlate_sparse` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned | ridge, texture and rank filters: no consumer |
| `wiener`, `LPIFilter2D`, `filter_forward`, `filter_inverse` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned | the HRTEM Wiener filter is `HRTEMFilter`; these are generic deconvolution |

## scikit-image 0.26.0, `feature`

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `peak_local_max` (`min_distance`, `threshold_abs`, `threshold_rel`, `exclude_border`, `num_peaks`, `footprint`, `labels`, `num_peaks_per_label`, `p_norm`) | present (0.6.0; footprint and labels since the next minor) | `PeakLocalMax.find` | exact indices (integer output) | study | n-D absent; 166 pinned cases plus 198 with `footprint=` and `labels=` (`PeakFootprint`, `labels:`, `numPeaksPerLabel:`) plus 7 with ±infinite pixels (0.6.1), coordinates exact; a NaN pixel is refused (`FilterError.parameter("NaN pixel")`): scikit-image answers by its NaN comparisons, which depend on the threshold, and no study input holds NaN. Registered deviation |
| `corner_peaks` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | built on `peak_local_max` |
| `match_template` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | atom columns by template |
| `blob_dog`, `blob_log`, `blob_doh` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | atom-column finding |
| `canny`, `corner_harris`, `corner_shi_tomasi`, `corner_foerstner`, `corner_kitchen_rosenfeld`, `corner_moravec`, `corner_fast`, `corner_orientations`, `corner_subpix` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned | natural-image detectors |
| `hessian_matrix`, `hessian_matrix_det`, `hessian_matrix_eigvals`, `structure_tensor`, `structure_tensor_eigenvalues`, `shape_index` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned |  |
| `ORB`, `SIFT`, `BRIEF`, `CENSURE`, `daisy`, `hog`, `local_binary_pattern`, `haar_like_feature`, `multiblock_lbp`, `fisher_vector`, `learn_gmm`, `Cascade`, `match_descriptors`, `graycomatrix`, `graycoprops`, `multiscale_basic_features`, `draw_*`, `plot_matched_features` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned | descriptors and texture: no consumer |

## scikit-image 0.26.0, `measure`

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `profile_line` (`order` 0 to 5, five modes, `linewidth`, `reduce_func`, `cval`) | present | `LineProfile` | decimal 6; measured 3.9e-15 f64, 4.7e-7 f32 of image maximum | study | multichannel as `sample(channels:)` and `unreduced(channels:)` (one Micrograph per channel; 360 pins at assert_allclose dtype defaults); width below 1 and degenerate segments throw |
| `label` | present | `Segmentation.label` | exact | study | n-D absent |
| `regionprops`: area, bbox, centroid, inertia tensor and eigenvalues, `axis_major_length`, `axis_minor_length`, `orientation` | present | `Segmentation.objectMoments` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | 8 properties, 2D |
| `regionprops`: `perimeter` (4), `perimeter_crofton` (4 directions), `eccentricity`, `solidity`, `equivalent_diameter_area`, `extent`, `euler_number`, `area_filled`, `area_convex`, `feret_diameter_max` | present (0.7.0) | `Segmentation.regionProperties` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32); integers exact | study | 13 planted masks, 8- and 4-connected, 251 objects against the 0.26.0 checkout; Feret is the maximum caliper over the convex hull's contour vertices (the needle length), `axis_major_length` stays 4√λ1; the orientation of an untilted blob may land on the other side of ±π/2 (same axis) |
| `regionprops`: `intensity_mean/min/max/std`, `centroid_weighted` | present (0.7.0) | `RegionIntensity` via `regionProperties(_:intensity:)` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | one 64x64 image, 4 labels |
| `regionprops_table` (CSV through the tidy results format) | present (0.7.0) | `RegionProperties.tidyRows/tidyEntries` | exact text | study | `regionprops_table` names (`bbox-0`, `centroid-1`); lengths in the caller's unit |
| `regionprops`: the remaining ~12 properties (`moments_*`, `inertia_tensor` as arrays, `image`, `image_convex`, 3D forms, `pixel_area` scaling) | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned | no consumer; 3D is not in the kit |
| `centroid`, `inertia_tensor`, `inertia_tensor_eigvals`, `moments`, `moments_central`, `moments_normalized`, `moments_coords`, `moments_coords_central` | partial | `Segmentation.objectMoments` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | per labelled object only; no free functions on an array |
| `block_reduce` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | binning |
| `find_contours`, `approximate_polygon`, `subdivide_polygon`, `points_in_poly`, `grid_points_in_poly`, `perimeter`, `perimeter_crofton`, `euler_number`, `moments_hu` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned | no consumer |
| `CircleModel`, `EllipseModel`, `LineModelND`, `ransac` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned | DiffractoKit fits ellipses to rings |
| `shannon_entropy`, `blur_effect`, `pearson_corr_coeff`, `manders_*`, `intersection_coeff`, `marching_cubes`, `mesh_surface_area` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned |  |

## scikit-image 0.26.0, `registration`

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `phase_cross_correlation` (real space, `upsample_factor`, `normalization`) | present | `PhaseCorrelation.register` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | integer upsample 1 to 1024 |
| `phase_cross_correlation` with `reference_mask`, `moving_mask`, `overlap_ratio` | present (0.6.0) | `PhaseCorrelation.registerMasked`, `maskedCorrelation` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | returns the shift only (scikit-image's error and phase difference are NaN there); 24 pinned cases; exact-tie maxima are not reproducible across FFT lengths |
| `phase_cross_correlation` `space='fourier'`, `disambiguate=True` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful |  |
| `optical_flow_ilk`, `optical_flow_tvl1` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned |  |

## scikit-image 0.26.0, `restoration`

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `denoise_tv_chambolle` (2D, one channel, `weight`, `eps`, `max_num_iter`) | present (0.8.0) | `Denoise.totalVariationChambolle` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | 144 pins; bit-identical in both dtypes (NumPy pairwise sums replayed so the stop test sees the same energies); no `channel_axis`, no nD, no integer images |
| `denoise_tv_bregman` (2D, one channel, `weight`, `max_num_iter`, `eps`, `isotropic`) | present (0.8.0) | `Denoise.totalVariationBregman` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | 108 pins; 1e-14 relative (f64), 2.7e-6 (f32); an axis of length 1 is refused (the compiled kernel reads outside the array there); no `channel_axis`, no integer images |
| `denoise_nl_means` (2D, one channel, `patch_size`, `patch_distance`, `h`, `fast_mode`, `sigma`) | present (0.8.0) | `Denoise.nonLocalMeans` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | 284 pins; bit-identical in both modes and dtypes; the weight is the compiled kernel's Schraudolph exp approximation; `patch_size` 1 is refused; no `channel_axis`, no 3D/4D, no `preserve_range`, no integer images |
| `denoise_bilateral`, `denoise_wavelet`, `denoise_invariant`, `calibrate_denoiser`, `cycle_spin`, `estimate_sigma` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful |  |
| `richardson_lucy`, `wiener`, `unsupervised_wiener` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | deconvolution |
| `rolling_ball`, `ball_kernel`, `ellipsoid_kernel` (2D, Float64 and Float32) | present (0.8.0) | `RollingBall.background`, `ballKernel`, `ellipsoidKernel` | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | `nansafe` as `nanSafe:` (21 more pins, `tools/rolling-ball-nansafe`; without it NaN input is refused), N-D missing; pinned on 8 cases incl. a kernel larger than the image; the `kernel=` path converts the kernel to the image type before its infinity test, as scikit-image does, so a finite value that overflows Float32 drops the cell instead of making the pixel -inf (#47, 3 cases, `tools/rolling-ball-kernel-overflow`) |
| `inpaint_biharmonic`, `unwrap_phase` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned |  |

## scikit-image 0.26.0, `transform`, `exposure`, `segmentation`, `morphology`

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| `transform.rotate`, `rescale`, `resize`, `warp`, `warp_coords` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | study | study row 3 (`ndimage.zoom` and `rotate` first) |
| `transform.warp_polar`, `downscale_local_mean`, `resize_local_mean` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | polar unwrap of diffraction rings |
| `transform.AffineTransform`, `EuclideanTransform`, `SimilarityTransform`, `ProjectiveTransform`, `PolynomialTransform`, `PiecewiseAffineTransform`, `ThinPlateSplineTransform`, `estimate_transform`, `matrix_transform`, `swirl` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned | no consumer |
| `transform.hough_*`, `radon`, `iradon`, `iradon_sart`, `frt2`, `ifrt2`, `pyramid_*`, `integral_image`, `integrate`, `order_angles_golden_ratio` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned |  |
| `exposure.rescale_intensity`, `adjust_gamma`, `adjust_log`, `adjust_sigmoid`, `equalize_hist`, `cumulative_distribution`, `match_histograms`, `is_low_contrast` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful | `DisplayNormalization` covers display only |
| `exposure.equalize_adapthist`, `exposure.histogram` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned |  |
| `segmentation.clear_border` | present (0.7.0) | `BinaryMorphology.clearBorder` | exact (mask) | study | 8-connected objects, `buffer_size` 0 and 2; the `mask=` and label-image forms are not ported |
| `segmentation.watershed` (2D, markers, `mask`, `connectivity` 1 and 2, `compactness`, `watershed_line`) | partial (0.8.0) | `Segmentation.watershed` (`compactness:`, `watershedLine:`), `Segmentation.markers` | scikit-image 0.26.0 ships no tests in its wheel: labels exact | study | pinned on 23 cases (smooth, integer-tied, masked, distance-transform markers; 12 tie-heavy stress cases check the heap order); `compactness` and `watershed_line` held (#46, `tools/watershed-compact`, labels exact, tie-heavy cases; the compact value is fused as the arm64 build computes it); `markers=None`/int, `offset` and N-D missing |
| `segmentation.expand_labels`, `relabel_sequential`, `find_boundaries`, `flood`, `flood_fill` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | useful |  |
| `segmentation.slic`, `felzenszwalb`, `quickshift`, `random_walker`, `chan_vese`, `active_contour`, level sets, `join_segmentations`, `mark_boundaries`, `inverse_gaussian_gradient` | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned |  |
| `morphology.remove_small_objects`, `remove_small_holes` (`max_size`, `connectivity` 1 and 2; a boolean mask) | present (0.7.0) | `BinaryMorphology.removeSmallObjects/removeSmallHoles` | exact (mask) | study | `max_size=N` removes area <= N (the pre-0.26 `min_size=M` is `N = M - 1`) |
| `morphology.binary_erosion`, `_dilation`, `_opening`, `_closing` (the deprecated 0.26 forms, default footprint, `mode`) | present (0.7.0) | `BinaryMorphology.skimageErosion` etc. | exact (mask) | study | `mode='ignore'` (a true border for erosion, false for dilation), `min`, `max`; the footprint sequences are not ported |
| `morphology.disk`, `square`, `skeletonize`, `thin`, `medial_axis`, `convex_hull_image`, and the rest | missing |  | scikit-image 0.26.0 ships no tests in its wheel: assert_allclose dtype defaults (rtol 1e-7 f64, 1e-5 f32) | not planned | revisit with the next consumer |

## Not from a Python package

| Quantity | State | Truth |
|---|---|---|
| distance and angle between marks (anisotropic scales, Kahan cross product) | done (0.2.0) | `ImageMeasurement`: planted angles, closed forms and 100 NumPy float64 triples |
| display normalisation, log contrast, thumbnails, colour mapping | done (0.2.0) | `DisplayNormalization`, `Thumbnail`, `DisplayMapping`: Float32 asserted exact today against a NumPy reading of the definition (within the written tolerance); the log to the last Float32 bit |
| scale-bar numbers (1-2-5 length, unit ladder, text, drawn length) | done (0.2.0) | `ScaleBarMath`: exact on 250 random and 108 usual calibrations, plus the ladder grid across mantissa boundaries |
| square-region, bin and decade geometry, shown units and aspect | done (0.2.0) | `FFTRegion`, `SpectrumBin`, `RadialChartScale`, `FFTUnits`, `SpectrumShape`: exact against a transcribed definition on random cases |
| log-power spectrum display and its default window | done (0.2.0) | `SpectrumDisplay`: values to the last Float32 bit, window within a 2048th of the range of the (⌈n/2⌉)-th smallest value |
| the four display colour tables (viridis, inferno, gray, rdbu) | done (unreleased) | `Colormap`, `DisplayMapping.sampleRGB`: anchor tables, not matplotlib's maps (matplotlib is not in the pinned environment); bytes exact against a NumPy reading and an application's table bytes |
| radial-chart samples (drawable bins, nearest bin, decade marks) and the spot-mask defaults | done (unreleased) | `RadialChartScale.samples`, `nearest`, `powerAxisMarks`, `FFTMask`: bit for bit against a NumPy reading |
| CSV text: `%.17g` numbers, header values, cells, the radial-profile, spots and tidy results files | done (unreleased) | `CSVText`, `RadialProfileCSV`, `SpotsCSV`, `TidyResultsCSV`: byte for byte against an application's own writers on 46 seeded cases |
| Line-profile CSV text (`# key: value` header, `%.17g` rows) | done (0.5.0) | `ProfileCSV`: the lines and order of macTEM's former writer, byte for byte |
| areal number density with the Miles–Lantuéjoul edge weight, length and width summaries, `N_V = N_A / t` with σ propagated (`validation: "none"`) | done (0.7.0) | `ObjectStatistics.arealDensity`, `edgeWeight`, `numberDensity`: the closed forms of mac4DSTEM's former `PrecipitateStatistics.density` (same arithmetic order), and planted truth: the edge-corrected count of a box averages to exactly 1 over every placement; no calibration, no density |
| particle size summary (total, counted, edge, small, median and mean length, mean width) | done (0.7.0) | `ObjectStatistics.sizeSummary`: macTEM's former `ParticleCompute.summary` rules |
| window applied to an image; the strict threshold mask and the fraction level | done (0.7.0) | `WindowFunction.apply`, `Thresholding.mask/level`: held to the closed forms written out in the tests (the product, the strict compare, low + fraction x range) |
| `peak_local_max` on a centrosymmetric spectrum: centre exclusion and one of each mirror pair | done (0.7.0) | `PeakLocalMax.find(symmetry:)` with `SpectrumSymmetry`: a kit extension (not a scikit-image option), equal to macTEM's former `FFTSpotFinder` pruning on 25 random spectra |
| axis as offset, scale, unit (RosettaSciIO convention) | done (0.2.0) | `value(at: 0) == offset`; round trip with `origin` |
| Å per pixel, Å⁻¹ per pixel (Å⁻¹, nm⁻¹, mrad), canonical unit spellings | done (0.2.0) | closed forms (1 nm = 10 Å, mrad = 1/(1000 λ) Å⁻¹); every spelling pinned to the transcribed consumer table |
| radial profile against physical \|q\| | done | a planted ring at known \|q\| |
| spot spacing d = 1/\|q\| and the angle between two spots | done | planted lattices, closed forms |
| masked inverse FFT (Bragg filter) | done | NumPy's `ifft2(ifftshift(fftshift(fft2(x)) · M))` and a planted two-lattice image |
| HRTEM ABSF and Wiener filters (Kilaas 1998, Mitchell's script) | done | a float64 NumPy/SciPy reading of the same steps; never compared with DigitalMicrograph |
| GPA strain (Hÿtch, Snoeck and Kilaas 1998) | done (0.8.0), validation none | `GeometricPhaseAnalysis`; planted strain fields (see the GPA row above) |
| atom columns (Atomap, GPL-3.0) | absent | planned, its own GPL module |


## Geometric phase analysis (no upstream package)

| Function or option group | State | Kit symbol | Counterpart tolerance (source) | Priority | Notes |
|---|---|---|---|---|---|
| GPA strain from two reflections of a lattice image (Hytch, Snoeck and Kilaas 1998) | present, validation none (0.8.0) | `GeometricPhaseAnalysis.analyze`, `findReflections` | the planted field of `tools/gpa/planted.py` (no upstream package): 2x the plain-numpy selfcheck error per case (uniform 9.8e-4, bump 3.12e-3, bump_oblique 2.97e-3, perfect 1e-6), floor 1e-6, interior only | study | Float64 CPU only; measured errors equal the numpy GPA's (4.89e-4, 1.56e-3, 1.48e-3, 2.5e-11 at size 128); badged `validation: "none"` until one dataset with truth passes |
