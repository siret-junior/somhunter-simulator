#!/bin/sh

if [ "$#" -ne 1 ]; then
	echo "Illegal number of parameters"
	exit 1
fi


split -l 100 $1 $1_part.

for file in `ls $1_part.*`;
do
	python run_simulations.py -t $file -o $file.output &>$file.stdout &
done

exit 0
