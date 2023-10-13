Run this command to build the docker image for SegmATRon.
```
bash ./build.sh
```

Next, run this command to start a docker container from the builded docker image.
```
bash ./start.sh
```

To get into the SegmATRon docker container run.
```
bash ./into.sh
```
You may need to change the paths inside the "into.sh" script to make them correspond to your local paths.

Finally, setup CUDA Kernel for MSDeformAttn.
```
cd /segmatron/models/oneformer/modeling/pixel_decoder/ops
sudo env "PATH=$PATH" sh make.sh
cd ../../../../..
```
