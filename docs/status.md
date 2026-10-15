# ImagoKit status

## Now (2026-10-15)

- **Skeleton** (tier 3): the package with four modules (`ImagoCore`, `ImagoFFT`, `ImagoFilters`, `ImagoAlignment`), each a
  placeholder naming its registration; one skeleton test (Swift Testing); the gate `tools/test.sh` (release, the last-green
  stamp); the hook with its proof (`tools/hooks`: no sleep or poll loops, no kill by name); the consumer check
  (`tools/consumer-check`, a separate package, 4 checks); CI (`.github/workflows/ci.yml`, macos-26, not enabled: the owner's);
  `.spi.yml`; `docs/naming.md` (ImagoKit chosen, MicrografoKit runner-up); `docs/parity.md`.
- **The oracle**: `References/oracle-venv` (a link to the org folder's `References/ImagoKit/`), CPython 3.14.7 with NumPy
  2.5.4, SciPy 1.18.1, scikit-image 0.26.0 and its closure, wheels only, `--no-deps` from `tools/oracle/references-lock.txt`,
  `pip check` clean, provisioned under `../tools/memcap.sh` (peak 294 MB); freeze in `References/oracle-venv-freeze.txt`.
- **Licence**: BSD-3-Clause, decided at creation (`NOTICE` gives the reason); the owner confirms relicensing the app code here
  before the first push.

## Waiting on the owner

- Confirm or change the name (ImagoKit; runner-up MicrografoKit; `docs/naming.md`).
- Confirm the BSD-3-Clause licence: the slice-1 seed is the owner's own code from GPL-3.0 macTEM and mac4DSTEM, relicensed
  here. Otherwise GPL-3.0 like the other kits (replace `LICENSE` and the `NOTICE` paragraph).
- Create the remote and push: `gh repo create mac4DSTEM/ImagoKit --public --source . --remote origin --push` (or `--private`).
- Enable Actions; submit the URL to the Swift Package Index once public.

## Unverified

- CI has never run (no remote). macOS 15 and iOS 18 are declared, built on macOS 27 only.
