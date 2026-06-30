# 🎻 VIOLIN: Spatial Priors via Space Filling Curves for Small and Limited Data Vision Transformers

Official implementation of [ICML'26 paper](https://arxiv.org/abs/2606.14757): Spatial Priors via Space Filling Curves for Small and Limited Data Vision Transformers.




<div align="center">
  <img width="100%" alt="VIOLIN illustration" src="assets/VIOLIN.gif">
  <p><strong>Video:</strong> Scanning patterns for Snake, Zig-Zag, Hilbert and Peano curves.</p>
</div>

## 🔍 Overview

**VIOLIN** enhances spatial awareness in Vision Transformers (ViTs) by integrating **Space Filling Curves (SFCs)** into masked attention.  
It is compatible with standard ViT architectures and can be used during either **pretraining** or **fine-tuning**, with minimal compute overhead.

- 🔁 Uses 8 SFCs (Snake, Zig-Zag, Peano, Hilbert and their transposes)
- 🔧 Drop-in replacement for standard attention (no architectural changes)
- ⚙️ Works with **DeiT**, and **DINO** backbones
- 🚀 Improves both supervised and self-supervised performance
- 🔌 Plug-and-play: can be used directly during fine-tuning
- ⚡ Minimal computational and memory overhead

## Setup & Usage

This repo builds on existing ViT training frameworks. Please follow the original repositories (e.g., [DeiT](https://github.com/facebookresearch/deit), [DINO](https://github.com/facebookresearch/dino)) for environment setup and dependencies.  

For fine-tuning, please follow [VTAB Evaluation Code](https://github.com/BenediktAlkin/vtab1k-pytorch).

### Run VIOLIN-enhanced models

Each folder contains training scripts with VIOLIN attention:

- `DeiT/`: Supervised training (DeiT + VIOLIN)
- `DINO/`: Self-supervised training (DINO + VIOLIN)
- `logs/`: Logs for all experiments

Checkpoints of all models will be available soon. 

## Cite as:

```
@inproceedings{candogan2026spatial,
title={Spatial Priors via Space Filling Curves for Small and Limited Data Vision Transformers},
author={Leyla Naz Candogan and Arshia Afzal and Pol Puigdemont and Volkan Cevher},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
}
```
