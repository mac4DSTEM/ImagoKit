#!/bin/zsh
# apps-parity: extract macTEM 3d17311's five originals by `git show` (never its working tree) into Sources/MacTEMRef
# (gitignored), build in release and compare with the kit's Float32 path bit for bit on .build/pins-full (written by
# tools/slice-1/pins.py). Exit 0 only when every value and string is identical. Run at every frozen commit of slice 1.
set -euo pipefail
here=${0:A:h}
root=${here:h:h:h}
commit=3d17311
mkdir -p $here/Sources/MacTEMRef
for f in Sources/TEMCore/Compute/FFT2D.swift Sources/TEMCore/Compute/PowerSpectrum.swift \
         Sources/TEMCore/Compute/RadialProfile.swift Sources/TEMCore/Compute/SpotMeasure.swift \
         Sources/TEMCore/Data/CalibratedImage.swift; do
  git -C $root/../macTEM --no-optional-locks show $commit:$f > $here/Sources/MacTEMRef/${f:t}
done
cd $here
swift build -c release --jobs 4 --product apps-parity
"$(swift build -c release --show-bin-path)/apps-parity" "$root/.build/pins-full"
