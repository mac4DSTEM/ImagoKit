# CLAUDE.md — start here

ImagoKit: the macEM suite's image-analysis library (the sixth kit, the owner's decision of 2026-10-15 15:25), a Swift port
of what the apps compute on a micrograph: the exact-length 2D FFT and the centred power spectrum, the radial profile,
FFT-spot spacings and angles, the masked inverse FFT, the HRTEM filters; later drift and stack alignment, GPA strain and
atom-column finding. NumPy, SciPy and scikit-image are its references. BSD-3-Clause (README "Licence" and `NOTICE` say why
and what would change it). Modules: `ImagoCore` (image, calibration, the FFT engine), `ImagoFFT` (spectrum, profile, spots,
masked inverse), `ImagoFilters` (HRTEM filters), `ImagoAlignment` (slice 2, a placeholder with its scope). Rules only here;
state in `docs/status.md`. Tests are Swift Testing (`import Testing`, `@Test`, `#expect`, `#require`).

## Read first

1. The suite's `../CLAUDE.md` (board, tiers, how we work) and `../ROADMAP.md`, "Kits" (the direction, the oracles, the
   licence rule, the Apple-infrastructure requirements, the 0.1.0 bar, the kit loop).
2. `docs/registration-slice-1.md`: what slice 1 makes, the definitions, the pins, the planted truth, the gates, slice 2's scope.
3. `docs/status.md`: what is live, the hand-off, what is unverified.
4. `../CuboKit/CLAUDE.md` and `../DiffractoKit/CLAUDE.md`: the kit repos this one follows.

## Hard rules

- **mac4DSTEM and macTEM are read-only from here**: `git -C ../<repo> --no-optional-locks show <commit>:<path>`, never their
  working trees. Never build, test or write in them. Slice 1's seeds were read at macTEM `3d17311` and mac4DSTEM `7a0f727f`
  (their heads on 2026-10-15 15:30 CEST). The apps' code may be compiled from `git show` into `.build/` for the apps-parity
  check (registration section 6), never from their trees.
- **Registration before code** for anything tier 1 (every number a scientist reads): `docs/registration-<slice>.md`, with the
  definitions, the pins, the planted truth, the gates and the planned mutants, committed before the first Swift line; a change
  after code starts is a dated amendment with its reason.
- **The gate is `tools/test.sh`** (`swift test -c release "$@"`). Never run `swift build --build-tests -c release` (fails on
  this toolchain). The manifest carries no unsafe flags, so a consumer builds the kit optimised itself.
- **Green stamp**: a full, passing `tools/test.sh` run on a clean tree (no untracked file under Sources or Tests either) writes
  `.build/last-green` (commit, UTC time); `tools/last-green.sh` prints it and says whether HEAD equals it.
- **Hook**: `.claude/settings.json` runs `tools/hooks/no-sleep-poll.sh` before every Bash call; it refuses a sleep of 30 s or
  more, any `while`/`until` loop with a sleep in it, `killall`, and `pkill` without `-P <pid>` or a scratchpad pattern (it
  matches text: write a commit message that names them to a file). Kill by pid only. Wait on background-task notifications or
  Monitor. Proof: `tools/hooks/test-no-sleep-poll.sh`.
- **A reference for every number.** Pins: NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0 on CPython 3.14.7
  (`tools/oracle/references-lock.txt`, the same releases macTEM's FFT, spot and HRTEM-filter harnesses were measured on),
  installed by `tools/oracle/provision.sh` into the gitignored `References/oracle-venv` (a link to the org folder's
  `References/ImagoKit/`; the freeze is `References/oracle-venv-freeze.txt`). Oracles run as
  `../tools/memcap.sh -- References/oracle-venv/bin/python -I -B <script>`, one at a time, inputs under 2 GB. An unpinned
  copy is never a reference. Agreement with NumPy is agreement, not truth: truth is a planted lattice, a closed form or a
  stated invariant.
- **Pins are generated before the Swift code** (`tools/slice-<n>/pins.py` writes `Tests/<Module>Tests/Fixtures/`); the Swift
  tests read the pins and the planted truth; a difference is a failure of the port, never a tolerance to widen.
- **Speed never moves a number silently.** The Float32 vDSP path and the Float64 reference path are both public; the
  registration states the tolerance between them and against the oracle. No path is chosen by the hardware.
- **Public API only, no app types.** Nothing an app needs is `package`. A kit type that replaces an app original keeps its
  name (`FFT2D`, `PowerSpectrum`, `RadialProfile`, `MaskedInverseFFT`, `HRTEMFilter`, `PixelCrop`: the app deletes its own at
  the switch); a type the app keeps has a different kit name (`Micrograph` and `AxisCalibration`, not the app's
  `CalibratedImage` and `ImageAxis`). No message, comment or doc names an app or the suite (provenance lives in `NOTICE`).
  `tools/consumer-check` is a separate package that depends on this one by path and compiles only while every symbol it names
  is `public`.
- **Licence boundary.** The package is BSD-3-Clause. Code ported from a GPL source (Atomap, HyperSpy, py4DSTEM) never enters
  `ImagoCore`, `ImagoFFT`, `ImagoFilters` or `ImagoAlignment`; it goes into its own GPL-3.0 module (planned: `ImagoAtoms`) with
  its licence named in `Package.swift`, `NOTICE` and the README, or into a separate package, decided at that slice's
  registration. Nothing of D. R. G. Mitchell's `HRTEM_Filter.s` (no licence stated) is copied: the filters are a functional
  re-implementation of the registered steps.
- **Rigour follows risk** (`../CLAUDE.md`): tier named in every commit; tier 1 is registration, every new test broken once,
  one Gate B refuter on the frozen commit, Gate D when a cause is unproven, a fixture.
- **Names** (the suite's standard): the library is domain + `Kit`, modules carry its prefix: `ImagoCore`, `ImagoFFT`,
  `ImagoFilters`, `ImagoAlignment`. The name check is `docs/naming.md`.
- **Pushing, tagging and creating the remote are the owner's.** One session per repo; `main` only; commit by explicit paths at
  every green checkpoint. At each tag, `docs/consumers-x.y.z.md` says what each consumer changes.
- **The owner's data** lives in the suite's `data/` folder, read as `$MACEM_DATA` (default `../data`): never copied or
  committed here; a manifest line in `../data/README.md` per file used (the coordinator commits that file).
- **Suite board**: this repo's row is `lanes/imagokit`; how a session opens and closes it is in `../CLAUDE.md`.
