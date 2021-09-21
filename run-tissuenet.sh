#!/bin/bash 

#=== Request echoing of each command executed 
set -x

cd build-4-linux
rm -r ./output-tissuenet/*
mkdir -p output-tissuenet

date > timing.txt

./nyxus.exe --intDir=/home/ec2-user/work/tissuenet/intensity --segDir=/home/ec2-user/work/tissuenet/labels --outDir=./output-tissuenet --filePattern=* --csvFile=singlecsv --rotations=15,30,45,67.8,77

date >> timing.txt

ls -l ./output-tissuenet
cd ..
