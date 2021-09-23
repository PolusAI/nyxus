#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/nyxus:${version}