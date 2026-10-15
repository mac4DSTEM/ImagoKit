# ImagoKit status

## Now (2026-10-15 16:05 CEST, `5a40748`): slice 1a done, the 0.0.1 candidate

- **Slice 1a** (tier 1, `docs/registration-slice-1.md` with amendments 1–6): `ImagoCore` (`Micrograph`, `AxisCalibration`,
  `PixelCrop`, `LengthUnit`, `Precision`, `FFT2D` on vDSP Float32, `FFT2DDouble` the reference path) and `ImagoFFT`
  (`PowerSpectrum`, `SpectrumValues`, `RadialProfile`, `SpotMeasurement` with the angle between spots, `LatticePair`).
- **Gate** `tools/test.sh`: green and stamped at `5a40748`; 24 test functions (135 parameterised cases), 0 warnings, both
  precisions on every pinned case. **Full set** `tools/slice-1/full-parity.sh`: green (15 spectrum and profile, 41 spot, 30
  angle cases). **Apps parity** `tools/slice-1/apps-parity/run.sh`: the Float32 path equals macTEM `3d17311`'s TEMCore bit
  for bit, 580,392 of 580,392 values, no string differs. **Consumer check**: 25 checks, Debug, no warnings. **Hook proof**:
  passes.
- **Pins** `tools/slice-1/pins.py` (under memcap, peak 169 MB): macTEM's fft- and spot-parity harnesses at `3d17311` plus 30
  angle plants; committed 88 files, 6.06 MB; the rest in `.build/pins-full`.
- **Mutants**: 35 on the frozen commit, 33 caught by the committed gate (after the follow-up tests), 2 argued equivalent;
  5 of 5 Gate B follow-ups caught (`docs/slice-1-mutants-2026-10-15.md`).
- **Gate B** (Sonnet refuter on `eb042aa`): upheld with findings, no wrong number; findings 1, 2, 4 (part), 8 fixed in
  `f51d1f8`; the rest recorded in amendment 5.
- **Oracle**: `References/oracle-venv` (a link to the org folder's `References/ImagoKit/`), CPython 3.14.7, NumPy 2.5.4,
  SciPy 1.18.1, scikit-image 0.26.0, wheels `--no-deps` from `tools/oracle/references-lock.txt`, `pip check` clean.
- **Consumers**: `docs/consumers-0.0.1.md` (what macTEM deletes at its switch: the FFT2D copy, PowerSpectrum, RadialProfile,
  SpotMeasure; mac4DSTEM needs no change). The suite rule: apps re-pin at 0.1.0.

## Next

1. Slice 1b (registered): the masked inverse FFT (`ImagoFFT`) and the HRTEM filters (`ImagoFilters`), pins from macTEM's
   fft-parity `ifft` cases and hrtem-filter-parity (41 plants), apps parity extended, mutants, one refuter.
2. The real-data check, when a raw HRTEM image exists (amendment 4).
3. Slice 2's registration (`ImagoAlignment`, scope in registration section 10).
4. 0.1.0: the API pass of amendment 5 item 7, DocC, an example executable, the README consumer, the 0.1.0 bar.

## Waiting on the owner

- Confirm or change the name (ImagoKit; runner-up MicrografoKit; `docs/naming.md`).
- Confirm the BSD-3-Clause licence: the slice-1 seed is the owner's own code from GPL-3.0 macTEM and mac4DSTEM, relicensed
  here (`NOTICE`). Otherwise GPL-3.0 like the other kits (replace `LICENSE` and the `NOTICE` paragraph).
- Create the remote and push: `gh repo create mac4DSTEM/ImagoKit --public --source . --remote origin --push` (or `--private`).
- Tag 0.0.1 at `5a40748` when the name and licence stand: `git tag -a 0.0.1 5a40748 -m "ImagoKit 0.0.1: slice 1a" && git push origin 0.0.1`.
- Enable Actions; submit the URL to the Swift Package Index once public.
- A raw HRTEM image (Velox `.emd` or DM4, counts) in `data/tem/` for the real-data check: the 630 kx and 1.05 Mx images there
  are 8-bit RGB TIFF renders.

## Open (amendment 5)

- Float32 centroid and angle bands (1e-6 px, 1e-6°) are claims on the planted cases; a 2000:1 lattice pair gave 1.3e-6°.
- The "printed, not gated" weak-bin distribution is not implemented.
- API pass for 0.1.0: `FFT2D.transform` traps on a size mismatch; `angle(qa:qb:)` range beyond about 1e±150; a subnormal
  scale counts as calibrated; a spot on non-finite pixels returns NaN; which helpers stay public; the unsigned angle documented.

## Unverified

- CI has never run (no remote). macOS 15 and iOS 18 are declared; built and tested on macOS 27 only.
