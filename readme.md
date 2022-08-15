# sf-nnue
Nodchip's original NNUE impementation (efficiently updateable neural network)...optimized

https://github.com/nodchip/Stockfish/releases/tag/stockfish-nnue-2020-06-09

Windows 64-bit bmi2, avx2, & (fast) blas executables included
Use the 'blas' binaries to reduce training time 50% or more.

Please see
readme.txt &
stockfish.md
for more info

Use nnue-gui:
https://github.com/FireFather/nnue-gui
It's a basic GUI to keep track of settings, paths, UCI options, command line parameters, etc.
It will also launch the various binaries needed for all 3 phases: gen training data, gen validation data, and learning (training).
