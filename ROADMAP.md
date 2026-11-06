# Roadmap

ImagoKit is pre-1.0: a minor version may change the API, and its CHANGELOG entry says how to migrate. Open work is in the GitHub
issues; the function-by-function state and the known limits are in [`docs/parity.md`](docs/parity.md).

## Next

- The option gaps left in shipped functions: watershed, EDT, rolling ball, denoise and GPA options (#10), the remaining
  scikit-image and SciPy options (#12), mixed-radix plans for radices 11 and 13 and for Float32.

## Later

- Metal Float32 paths for the FFT and the separable filters, chosen per call behind the same API, each checked against the
  Float64 CPU path.
- Atom-column finding, in a module of its own.
- Validation of GPA strain on a dataset with known strain (until then its results carry `validation: "none"`).

## 1.0

- One naming convention: every image API in `rows` and `columns` (the older `Micrograph`, `LabelImage`, morphology and
  segmentation API still say `width` and `height`).
- `LocalizedError` texts for every error enum.
- A stable public API; from then on breaking changes only in a major version.
