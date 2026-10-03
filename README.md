<h1 align="center">GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2604.11585"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://geomprompt.github.io/"><img src="https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white" alt="Project Page"></a>
</p>

<p align="center"><b>CVPR 2026 URVIS Workshop</b></p>

GeomPrompt learns a geometric prompt from RGB; GeomPrompt-Recovery corrects degraded depth. Both use segmentation supervision with a frozen RGB-D segmenter.

## 📦 Installation

Python 3.11 or 3.12. Install PyTorch for your CUDA version, then the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## 🤖 Pretrained Models

DFormer-Base and GeminiFusion MiT-B3 source is included. Download their **SUN RGB-D** checkpoints and place them in `checkpoints/`:

- [DFormer-Base](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8): `SUNRGBD_DFormer_Base.pth`
- [GeminiFusion MiT-B3](https://github.com/JiaDingCN/GeminiFusion/releases/download/SUN_v2/mit-b3.pth.tar): `mit-b3.pth.tar`

## 📂 Dataset

Use the [SUN RGB-D data prepared for DFormer](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl), with `train.txt` and `test.txt` under the dataset root. Each line contains three paths relative to that root:

```text
RGB/train_0.jpg labels/train_0.png Depth/train_0.png
```

Labels use 0 for void and 1–37 for classes. Depth uses the dataset's uint8 representation or DFormer's uint16 encoding (divided by 256). DFormer RGB-only prompting can also load `wyrx/SUNRGBD_seg` automatically when `--data-root` is omitted.

## 🏋️ Training

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --segmenter dformer \
  --segmenter-checkpoint checkpoints/SUNRGBD_DFormer_Base.pth \
  --data-root /path/to/SUNRGBD --output-dir runs/prompt_dformer
```

For GeminiFusion, use `--segmenter geminifusion --segmenter-checkpoint checkpoints/mit-b3.pth.tar`. For recovery, add `--method recovery`. A single GPU uses `python train.py` with the same arguments; gradient accumulation automatically targets an effective batch of 32.

Defaults follow the paper: 300 epochs, AdamW, 10 warmup epochs, polynomial decay, 480×480 random crops, residual scale 15→80, and a factor-2 low-pass projection. Recovery samples one of seven depth corruptions at severity 0.1–0.9, keeping clean depth with probability 0.2. Training saves `best.pth` (EMA selected on full validation) and `last.pth`; resume with `--resume runs/prompt_dformer/last.pth`.

## ⚡ Inference

```bash
python evaluate.py --segmenter dformer \
  --segmenter-checkpoint checkpoints/SUNRGBD_DFormer_Base.pth \
  --data-root /path/to/SUNRGBD \
  --prompt-checkpoint runs/prompt_dformer/best.pth
```

DFormer uses native-resolution inference at scales {0.5, 0.75, 1, 1.25, 1.5} with horizontal flips. GeminiFusion uses a single 480×480 pass. Evaluation reports mIoU and pixel accuracy; `--output outputs/results.json` saves metrics.

For recovery, add `--mode recovery --corruption noise --severity 0.9` and supply the recovery checkpoint. Use `--mode broken` with the same corruption and severity for the degraded-depth control. `--mode rgb` and `--mode gt` evaluate zero depth and clean depth. Supported corruptions: `quantize`, `hole`, `dropout`, `noise`, `blur`, `banding`, `scale_shift`.

## 🤝 Acknowledgements

Our code builds on [DFormer](https://github.com/VCIP-RGBD/DFormer), [GeminiFusion](https://github.com/JiaDingCN/GeminiFusion), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), and [timm](https://github.com/huggingface/pytorch-image-models). We thank the authors for their code.

## ⚖️ License

Upstream license notices are retained under `geomprompt/segmenters/`, including the NVIDIA license for the SegFormer-derived code.

## 📜 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{jaganathan2026geomprompt,
  title   = {GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth},
  author  = {Jaganathan, Krishna and Vela, Patricio},
  journal = {arXiv preprint arXiv:2604.11585},
  year    = {2026}
}
```
