#!/usr/bin/env bash
L=48
for i in {1..16}; do
    id=${RANDOM}
    TMPFILE=`mktemp --tmpdir=/home/jkott/perm/tmp`
    cp run_a100.sh $TMPFILE

    echo "julia thermalize.jl --fp64 --rng=$id $L 1.0" >> $TMPFILE
    echo "rm $TMPFILE" >> $TMPFILE

    echo $TMPFILE

    chmod +x $TMPFILE
    bsub < $TMPFILE
done
