#!/bin/bash 

#=== Request echoing of each command executed 
set -x

cd build
mkdir -p output-jayapria
rm output-jayapria/*
./nyxus.exe ~/work/data-jayapriya/intensity ~/work/data-jayapriya/label ./output-jayapria
ls -l output-jayapria
cd ..
