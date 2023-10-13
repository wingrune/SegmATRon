#!/bin/bash

echo "Building container"
docker build .. \
    -f Dockerfile \
    -t segmatron:latest \
    --progress plain 

