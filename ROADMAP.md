# Roadmap

ImagoKit is pre-1.0: a minor version may change the API, and its CHANGELOG entry says how to migrate. The detailed list of
gaps is [`docs/open-items.md`](docs/open-items.md); the function-by-function state is [`docs/parity.md`](docs/parity.md).

## Next

- Speed where the kit is slower than Python at 512²: connected-component `label`, the upsampled phase correlation (Float32).
- Close gaps in shipped functions: `watershed` `compactness` and `watershed_line`, `rolling_ball` `nansafe`, more `get_window`
  families, spline shifts in the `constant` and `wrap` modes, mixed-radix plans for radices 11 and 13 and for Float32.
- Benchmark the 0.7.0 segmentation functions (thresholds, morphology, object statistics) against the Python packages.

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
