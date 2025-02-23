#!/bin/bash

cd ut
make clean
rm -r out
make DEFINES="-DBUILD=DEBUG $*"
mkdir -p out/cores
mkdir -p out/logs
cd ..