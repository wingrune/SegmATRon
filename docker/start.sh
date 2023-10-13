#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus all --name segmatron  segmatron:latest "/bin/bash"
