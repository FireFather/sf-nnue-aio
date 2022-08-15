#!/bin/bash
# make_avx2.sh

make profile-nnue ARCH=x86-64-avx2
strip stockfish.exe
mv stockfish.exe sf+nnue-aio.010822.halfkp_256x2-32-32.x64.avx2.exe
make clean
