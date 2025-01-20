#!/usr/bin/env bash
L=48
taus=(0.01 0.0299466 0.0566751 0.0887668)
init_ids=(1 3 6 9)

idx=1
n=4

files=`ls ~/perm/modelH/thermalized/tau/${init_ids[idx]}/thermalized_L_${L}_id_*.jld2 | sort -R | head -$n`
for file in $files; do

    id=${RANDOM}
    TMPFILE=`mktemp --tmpdir=/home/jkott/perm/tmp`
    cp run_h100.sh $TMPFILE

    echo "julia tau_measure.jl --fp64 --H0 --init=$file --mass=${taus[idx]} --rng=$id $L 0.1" >> $TMPFILE
    echo "rm $TMPFILE" >> $TMPFILE

    echo $TMPFILE

    chmod +x $TMPFILE
    bsub < $TMPFILE

done
#done
