# ``ImagoAlignment``

Drift and stack alignment: the translation between two images, the drift of a stack, a sub-pixel shift, and the sum of a
stack after it is aligned.

The translation comes from phase cross-correlation with sub-pixel refinement (scikit-image's `phase_cross_correlation`
without masks), the shifts are SciPy's `ndimage.shift` of order 0 and 1 and the Fourier shift, and every one runs on a
Float32 path (Accelerate) and a Float64 reference path. A shift is the translation to apply to the moving image so that it
registers onto the reference; it is in pixels, row first.

## Topics

### Registering two images

- ``PhaseCorrelation``
- ``ImageRegistration``
- ``Normalization``
- ``PixelShift``

### Shifting an image

- ``ImageShift``
- ``ShiftOrder``
- ``ShiftMode``

### Drift and stack sums

- ``DriftSeries``
- ``DriftReference``
- ``AlignedSum``
- ``AlignedStack``
- ``AlignmentMethod``

### Errors

- ``AlignmentError``
