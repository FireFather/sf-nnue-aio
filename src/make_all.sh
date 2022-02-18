#!/bin/bash
# make_all.sh

make profile-nnue ARCH=x86-64-bmi2
strip stockfish.exe
mv stockfish.exe sf+nnue-aio.250720.halfkp_256x2-32-32.x64.bmi2.exe
make clean 

make profile-nnue ARCH=x86-64-avx2
strip stockfish.exe
mv stockfish.exe sf+nnue-aio.250720.halfkp_256x2-32-32.x64.avx2.exe
make clean 

make profile-nnue ARCH=x86-64-sse42
strip stockfish.exe
mv stockfish.exe sf+nnue-aio.250720.halfkp_256x2-32-32.x64.popc.exe
make clean 
