#!/bin/bash
# make_avx2-blas.sh

make nnue-learn ARCH=x86-64-avx2
strip stockfish.exe
mv stockfish.exe sf+nnue-aio.140822.halfkp_256x2-32-32.x64.blas.avx2.blas.exe
make clean 
