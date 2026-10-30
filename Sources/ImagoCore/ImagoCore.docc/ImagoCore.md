# ``ImagoCore``

The image every other module reads, its calibration, the exact-length 2D Fourier transform, and the binary and labelled
image tools (morphology, the distance transform, segmentation, object statistics, the rolling-ball background).

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
- ``Colormap``

### The Fourier transform

- ``FFT2D``
- ``FFT2DDouble``
- ``FFT2DTransform``
- ``Precision``
- ``FFTSurface``
- ``ComplexImage``
- ``FFTNorm``
- ``FFTAxes``
- ``FFTSurfaceError``

### Morphology and the distance transform

SciPy's `ndimage` binary morphology and `distance_transform_edt` (2D).

- ``BinaryMorphology``
- ``StructuringElement``
- ``MorphologyMode``
- ``MorphologyError``
- ``DistanceTransform``

### Segmentation and objects

Connected-component labels, scikit-image's marker-controlled `watershed`, `regionprops` and the per-object statistics,
areal and number densities.

- ``Segmentation``
- ``RegionProperties``
- ``RegionIntensity``
- ``ObjectStatistics``
- ``ArealDensityError``

### Background

scikit-image's `restoration.rolling_ball`, Float64 and Float32.

- ``RollingBall``
- ``RestorationError``

### Export

- ``CSVText``
- ``ProfileCSV``
- ``TidyResultsCSV``
