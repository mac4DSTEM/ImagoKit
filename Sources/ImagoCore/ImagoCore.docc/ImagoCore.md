# ``ImagoCore``

The image every other module reads, its calibration, and the exact-length 2D Fourier transform.

A ``Micrograph`` is one Float32 image, row-major with x fastest, with an ``AxisCalibration`` per axis exactly as the file
states it. ``FFT2D`` evaluates the exact N-point DFT of any size on Accelerate (radix-2, exact radix-3 and radix-5 plans,
vDSP's DFT, and Bluestein for every other length) in Float32; ``FFT2DDouble`` is the same algorithm in Double, the
reference path. Every product in the kit takes a ``Precision`` and records it: speed never moves a number silently.

## Topics

### The image

- ``Micrograph``
- ``AxisCalibration``
- ``PixelCrop``
- ``MicrographError``

### Units

- ``LengthUnit``

### Measurement

- ``ImageMeasurement``
- ``PixelPoint``
- ``IntensityHistogram``
- ``HistogramError``
- ``LineProfile``
- ``ProfileMode``
- ``ProfileOrder``
- ``ProfileReduction``
- ``ProfileArithmetic``
- ``LineProfileError``

### Display

- ``DisplayNormalization``
- ``Thumbnail``
- ``ScaleBarMath``
- ``ScaleBarPlan``
- ``DisplayMapping``

### The Fourier transform

- ``FFT2D``
- ``FFT2DDouble``
- ``FFT2DTransform``
- ``Precision``
