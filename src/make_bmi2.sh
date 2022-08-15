#!/bin/bash
# make_bmi2.sh

make profile-nnue ARCH=x86-64-bmi2
strip stockfish.exe
mv stockfish.exe sf+nnue-aio.010822.halfkp_256x2-32-32.x64.bmi2.exe
make clean
