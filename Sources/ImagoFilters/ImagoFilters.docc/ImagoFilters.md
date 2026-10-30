# ``ImagoFilters``

Filters for high-resolution TEM images, and the image filters, thresholds, peak finding, denoising and strain mapping
around them.

``HRTEMFilter`` implements the average background subtraction filter (ABSF) and the Wiener filter of R. Kilaas, J. Microsc.
190 (1998) 45–51, in the steps D. R. G. Mitchell's DigitalMicrograph script describes: an iterative estimate of the amorphous
background from the rotational average of the spectrum, two masks built from it, a Butterworth low-pass, and the masked
inverse transforms. The result carries every stage. The outputs have never been compared with DigitalMicrograph; the
kit's tests hold them to a float64 NumPy/SciPy reading of the same steps.

## Topics

### The HRTEM filters

- ``HRTEMFilter``
- ``HRTEMFilterError``

### Smoothing, band-pass filters and windows

SciPy's `gaussian_filter` and `uniform_filter`, scikit-image's `butterworth` and `difference_of_gaussians`, SciPy's
`get_window`.

- ``SeparableFilters``
- ``BoundaryMode``
- ``FilterError``
- ``BandpassFilters``
- ``WindowFunction``
- ``WindowType``

### Thresholding

scikit-image's global thresholds (Otsu, Li, Yen, triangle, mean, ISODATA), the local threshold and hysteresis.

- ``Thresholding``
- ``ThresholdPixels``
- ``ThresholdError``

### Peaks

scikit-image's `peak_local_max`, and the symmetry of a spectrum's peaks.

- ``PeakLocalMax``
- ``PeakCoordinate``
- ``PeakBorder``
- ``SpectrumSymmetry``

### Denoising

scikit-image's total-variation (Chambolle, Bregman) and non-local-means denoisers, 2D single-channel.

- ``Denoise``
- ``DenoiseError``

### Strain

Geometric phase analysis (Hytch, Snoeck and Kilaas 1998) from two reflections. Checked on planted lattices only: not
validated on a real dataset.

- ``GeometricPhaseAnalysis``
- ``GeometricPhaseAnalysisError``
