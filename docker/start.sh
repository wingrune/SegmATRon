#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus all --name segmatron --volume /home/wingrune/segmatron_dev:/segmatron_dev segmatron:latest "/bin/bash"