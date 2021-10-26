#!/bin/bash 

#=== Request echoing of each command executed 
set -x

cd build-4-linux

./nyxus.exe --intDir=/home/ec2-user/work/data/hamda-deep/int --segDir=/home/ec2-user/work/data/hamda-deep/seg --outDir=/home/ec2-user/work/data/OUTPUT-hamda --filePattern=.* --csvFile=separatecsv --loaderThreads=2 --reduceThreads=8 

cd ..
