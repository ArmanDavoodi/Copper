#!/bin/bash

rm -r ut/out/cores
rm -r ut/out/logs
mkdir -p ut/out/cores
mkdir -p ut/out/logs
# ulimit -c unlimited

# export CORE_DUMP_DIR=ut/out/cores/core.$(date +%Y_%m_%d_%H_%M_%S).$(pidof -s your_executable)

# echo $CORE_DUMP_DIR | sudo tee /proc/sys/kernel/core_pattern

./ut/build/bin/$1 > ut/out/logs/$1.log

# echo "|/usr/share/apport/apport -p%p -s%s -c%c -d%d -P%P -u%u -g%g -- %E\n" | sudo tee /proc/sys/kernel/core_pattern

# unset CORE_DUMP_DIR

# todo run ut all
# todo run ut multiple files