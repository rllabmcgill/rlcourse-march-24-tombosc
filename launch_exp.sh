#!/bin/sh


N="300"
T="10"
EPS="0.1"
GAMMA="0.99"
NAME="taxi"

# HLS: separate loop as lambda > 0
for l in $(LANG=en_us seq 0.1 0.1 1.0)
do
python gym_train.py -m "$NAME" -l hls -n "$N" --gamma "$GAMMA" --lambda "$l" --eps "$EPS" --n-trials "$T"
done



for l in $(LANG=en_us seq 0.0 0.1 1.0)
do
for a in $(LANG=en_us seq 0.1 0.1 1.0)
do
	python gym_train.py -m "$NAME" -l sarsa -n "$N" --gamma "$GAMMA" --alpha="$a" --lambda "$l" --eps "$EPS" --n-trials "$T"
	#python gym_train.py -m "$NAME" -l expsarsa -n "$N" --gamma "$GAMMA" --alpha="$a" --lambda "$l" --eps "$EPS" --n-trials "$T"
	#python gym_train.py -m "$NAME" -l q -n "$N" --gamma "$GAMMA" --alpha="$a" --lambda "$l" --eps "$EPS" --n-trials "$T"
done
done





