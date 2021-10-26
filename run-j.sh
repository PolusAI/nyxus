#!/bin/bash 

#=== Request echoing of each command executed 
set -x

cd build-4-linux
rm -r ./output-j/*
mkdir -p output-j

./nyxus.exe --features=*all* --intDir=/home/ec2-user/work/data-jayapriya/intensity --segDir=/home/ec2-user/work/data-jayapriya/labels --outDir=./output-j --filePattern=.* --csvFile=singlecsv --rotations=15,30,45,67.8,77 --loaderThreads=2 --reduceThreads=8

cd ..

