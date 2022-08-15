#!/bin/bash
# make_all-blas.sh

make nnue-learn ARCH=x86-64-bmi2
strip stockfish.exe
mv stockfish.exe sf+nnue-aio.140822.halfkp_256x2-32-32.x64.bmi2.blas.exe
make clean 

make nnue-learn ARCH=x86-64-avx2
strip stockfish.exe
mv stockfish.exe sf+nnue-aio.140822.halfkp_256x2-32-32.x64.avx2.blas.exe
make clean 

