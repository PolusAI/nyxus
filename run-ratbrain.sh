/home/ec2-user/work/data_polus_tissuenet/intensity

#!/bin/bash 

#=== Request echoing of each command executed 
set -x

cd rebuild
rm -r output-tissuenet/*
mkdir -p output-tissuenet

date > timing.txt

./nyxus.exe --intDir=/home/ec2-user/work/data-ratbrain/int --segDir=/home/ec2-user/work/data-ratbrain/seg --outDir=/home/ec2-user/work/data-ratbrain --filePattern=* --csvFile=singlecsv --rotations=15,30,45,67.8,77 --loaderThreads=4 --reduceThreads=16 

date >> timing.txt

ls -l output-tissuenet
cd ..
