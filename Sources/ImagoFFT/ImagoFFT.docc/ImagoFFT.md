# ``ImagoFFT``

What a micrograph's Fourier transform tells: the centred spectrum, its radial profile, lattice spacings and the angles
between spots, and the image rebuilt from chosen spots.

The conventions are NumPy's: `fftshift(fft2(x))` with the zero frequency at index n/2 on each axis, the frequency axes
`fftshift(fftfreq(n, d))` with each axis's own scale, nothing windowed or padded unless a function says so. A spot's
spacing comes from a Hann-windowed, mean-removed spectrum, snapped to the largest bin near the click and refined by a 5 × 5
power-weighted centroid; the angle between two spots is taken from their physical q vectors, so anisotropic sampling is
accounted for.

## Topics

### Complex 1D transforms

Both directions are unnormalised; divide inverse output by the length for NumPy's default inverse convention.

- ``FFT1D``
- ``FFT1DDouble``

### The spectrum

- ``PowerSpectrum``
- ``SpectrumValues``
- ``PowerSpectrumError``
- ``RadialProfile``
- ``RadialProfileError``

### Spots, spacings and angles

- ``SpotMeasurement``
- ``LatticePair``
- ``SpotMeasurementError``

### Bragg filtering

- ``MaskedInverseFFT``
- ``MaskedInverseFFTError``

### Drawing a spectrum

- ``SpectrumDisplay``
- ``FFTRegion``
- ``SpectrumBin``
- ``RadialChartScale``
- ``SpectrumShape``
- ``FFTUnits``
- ``FFTMask``

### Export

- ``FFTExportHeader``
- ``FFTExportText``
- ``RadialProfileCSV``
- ``SpotsCSV``
