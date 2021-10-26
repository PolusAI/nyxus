#!/bin/bash 

#=== Request echoing of each command executed 
set -x

cd build-4-linux

rm -r /home/ec2-user/work/data/OUTPUT-tissuenet
mkdir -p /home/ec2-user/work/data/OUTPUT-tissuenet

./nyxus.exe --verbosity=0 --features=*all* --intDir=/home/ec2-user/work/data/tissuenet/intensity --segDir=/home/ec2-user/work/data/tissuenet/labels --outDir=/home/ec2-user/work/data/OUTPUT-tissuenet --filePattern=.* --csvFile=separatecsv --loaderThreads=2 --reduceThreads=8 

cd ..
