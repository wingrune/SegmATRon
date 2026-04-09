# Segmatron: Embodied Adaptive Semantic Segmentation

## Installation

The evaluation of SegmATRon can be performed inside a docker container. One can find all necessary files and scripts to build a docker image and run a docker container inside "docker/" folder. Read the README.md inside the "docker/" directory. Prepare data and checkpoints before building the docker image.

Alternatively, you can create a conda environment with Python 3.8, CUDA 11.3 and PyTorch 1.11.0. Then:

Install OpenCV:
```
pip3 install -U opencv-python
```
Install detectron2:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install other requirements:

```
docker/requirements.txt ./
pip install -r docker/requirements.txt
```

Install ninja:
```
sudo wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
sudo gunzip /usr/local/bin/ninja.gz
sudo chmod a+x /usr/local/bin/ninja
```

Setup CUDA Kernel for MSDeformAttn.
```
cd models/oneformer/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../..
```
## Data and pretrained checkpoints

Test data and pretrained checkpoints can be downloaded by using the following links from our Anonymous Google Drive:

- [Mask2Former (Single Frame Baseline)](https://drive.google.com/file/d/14SXE0rZU7H9NAxOfPSOvhZsgW_2MFiXK/view?usp=drive_link)
- [MaskDINO (Single Frame Baseline)](https://drive.google.com/file/d/1BaKlgArItOKgXyKQ-6lldl1O_m-DH3He/view?usp=drive_link)
- [SegmATRon (MaskFormer)](https://drive.google.com/file/d/1fndbrlwdcJDGRNaXjxAfGWpdiRpbf5g0/view?usp=drive_link)
- [SegmATRon (MaskDINO)](https://drive.google.com/file/d/1veThLJDk_WPEwISTz9vaQOr_yh2dajqW/view?usp=drive_link)
- [SegmATRon Habitat data](https://drive.google.com/file/d/10oYGLo9d8xso5M5XDxWjuq-cXzZ4Q3sv/view?usp=drive_link)
- [SegmATRon AI2-Thor data](https://drive.google.com/file/d/1-sWox5ezZBcF1DRLLYT7CUC01C7buhCy/view?usp=drive_link)

Expected data and pretrained checkpoints structure:
    segmatron/
        checkpoints/
            mask2former_single_frame.pt
            maskdino_single_frame.pt
            segmatron_mask2former.pt
            segmatron_maskdino.pt
        data/
            segmatron_ai2thor/
                annotations/
                test/
                test_mask/
            segmatron_habitat/
                annotations/
                val/
                val_mask/

## Evaluation

Evaluation of the SegmATRon (1 step) model and OneFormer (Single Frame baseline) can be performed by running 

Mask2Former Single Frame baseline on AI2-THOR dataset:

``python evaluate.py --config=configs/mask2former/mask2former_single_frame_baseline_r50_ai2thor.yaml``.

Mask2Former Single Frame baseline on Habitat dataset:

``python evaluate.py --config=configs/mask2former/mask2former_single_frame_baseline_r50_habitat.yaml``.

SegmATRon (Mask2Former) on AI2-THOR dataset:

``python evaluate.py --config=configs/mask2former/segmatron_mask2former_4_steps_r50_ai2thor.yaml``.

SegmATRon (Mask2Former) on Habitat dataset:

``python evaluate.py --config=configs/mask2former/segmatron_mask2former_4_steps_r50_habitat.yaml``.

MaskDINO Single Frame baseline on AI2-THOR dataset:

``python evaluate.py --config=configs/maskdino/maskdino_single_frame_baseline_r50_ai2thor.yaml``.

MaskDINO  Single Frame baseline on Habitat dataset:

``python evaluate.py --config=configs/maskdino/maskdino_single_frame_baseline_r50_habitat.yaml``.

SegmATRon (MaskDINO) on AI2-THOR dataset:

``python evaluate.py --config=configs/maskdino/segmatron_maskdino_4_steps_r50_ai2thor.yaml``.

SegmATRon (MaskDINO) on Habitat dataset:

``python evaluate.py --config=configs/maskdino/segmatron_maskdino_4_steps_r50_habitat.yaml``.

The code will automatically take over current GPU device.

The evaluator will output visualizations and results in a folder called
`evaluation_results/`. 

Note: the evaluation results can be slightly different depending on the specific random actions chosen by SegmATRon (1 step). For demo purposes we set random seeds.


