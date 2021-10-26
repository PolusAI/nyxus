#!/bin/bash 

#=== Request echoing of each command executed 
set -x

cd build-4-linux

rm -r OUTPUT-ratbrain/*
mkdir -p OUTPUT-ratbrain

./nyxus.exe --verbosity=12 --features=*all* --intDir=/home/ec2-user/work/data-ratbrain/int --segDir=/home/ec2-user/work/data-ratbrain/seg --outDir=/home/ec2-user/work/data-ratbrain/OUTPUT-ratbrain --filePattern=.* --csvFile=singlecsv --rotations=15,30,45,67.8,77 --loaderThreads=2 --reduceThreads=4 

cd ..
