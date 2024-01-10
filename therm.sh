#!/usr/bin/env bash
L=48
for i in {1..8}; do
    id=${RANDOM}
    TMPFILE=`mktemp --tmpdir=/home/josh/tmp`
    out="~/tmp/log_${L}_$i.out"
    err="~/tmp/log_${L}_$i.err"

    echo "echo \`date\` >> $out" >> $TMPFILE
    echo "julia --project thermalize.jl --rng=$id $L 1.0 1>> $out 2> $err" >> $TMPFILE
    echo "echo \`date\` >> $out" >> $TMPFILE
    echo "rm $TMPFILE" >> $TMPFILE

    echo $TMPFILE

    chmod +x $TMPFILE
    echo $TMPFILE | batch
done
