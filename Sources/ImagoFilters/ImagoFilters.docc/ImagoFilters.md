# ``ImagoFilters``

Filters for high-resolution TEM images.

``HRTEMFilter`` implements the average background subtraction filter (ABSF) and the Wiener filter of R. Kilaas, J. Microsc.
190 (1998) 45–51, in the steps D. R. G. Mitchell's DigitalMicrograph script describes: an iterative estimate of the amorphous
background from the rotational average of the spectrum, two masks built from it, a Butterworth low-pass, and the masked
inverse transforms. The result carries every stage. The outputs have never been compared with DigitalMicrograph; the
kit's tests hold them to a float64 NumPy/SciPy reading of the same steps.

## Topics

### The HRTEM filters

- ``HRTEMFilter``
- ``HRTEMFilterError``

### Thresholding

- ``Thresholding``

### Restoration and strain

- ``Denoise``
- ``GeometricPhaseAnalysis``
