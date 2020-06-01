#!/bin/bash
path=./ckpt/
FILES=$path/*
for f in $FILES; do
    echo "==========================" ${f##*/}
    python -u eval.py $path${f##*/} 1
done
