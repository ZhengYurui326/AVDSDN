# AVDSDN

<div align="center">

**Synergistic Dual-stream Neural Network Based on Adaptive Variational Mode Decomposition Denoising for Automatic Modulation Classification**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-RML2016.10a-4C8CBF)](https://www.deepsig.ai/datasets/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **AVDSDN** for robust automatic modulation classification (AMC) from raw I/Q samples.

</div>

## Overview

AVDSDN is designed for modulation recognition in noisy wireless channels. It turns variational mode decomposition (VMD) from a fixed preprocessing operation into a task-oriented denoising stage and combines it with a parallel CNN–Transformer architecture.

The model has three main components:

- **Adaptive VMD denoising (AVD):** learns point-wise weights for five VMD modes and reconstructs a signal optimized directly by the classification objective.
- **Synergistic dual-stream feature extraction:** a CNN stream captures robust local and spatial patterns, while a Transformer stream models long-range temporal dependencies with deformable attention.
- **Cross-stream attention fusion (CSAF):** adaptively combines the complementary features before classification.

<p align="center">
  <img src="doc/avdsdn_architecture.png" width="95%" alt="AVDSDN architecture">
</p>

## Highlights

- End-to-end, task-oriented adaptive denoising for non-stationary I/Q signals.
- Decoupled CNN and Transformer streams for complementary local–spatial and global–temporal representations.
- Pretrained checkpoint and a complete evaluation pipeline for RML2016.10a.
- Strong robustness in the low-SNR regime.
- Pretrained RML2016.10a checkpoint included for evaluation.

## Paper-reported results

The following results are reported in the manuscript. MAA and MAF denote mean average accuracy and mean average F1 score over all SNR levels. Inference latency was measured with FP32 on an NVIDIA RTX 4090.

| Dataset | Accuracy at SNR <= 0 dB | Accuracy at SNR > 0 dB | MAA | MAF | Parameters | FLOPs | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| RML2016.10a | **36.12%** | **93.36%** | **64.75%** | **67.42%** | 268.346K | 21.50M | 11.66 ms |
| RML22 | **57.17%** | **94.90%** | **76.93%** | **76.56%** | 268.089K | 21.50M | 11.82 ms |
| HisarMod2019.1 | **76.86%** | **98.91%** | **87.89%** | **85.46%** | 373.641K | 348.80M | 12.56 ms |

<p align="center">
  <img src="doc/rml2016_accuracy.png" width="68%" alt="Accuracy versus SNR on RML2016.10a">
</p>

The AVD module consistently improves the tested AMC backbones, with its largest gains appearing in the low-to-medium SNR range.

<p align="center">
  <img src="doc/avd_ablation.png" width="75%" alt="Effectiveness of adaptive VMD denoising">
</p>

> [!NOTE]
> This release currently provides the model, preprocessing pipeline, pretrained checkpoint, training code, and evaluation code for **RML2016.10a**. Dataset-specific pipelines for RML22 and HisarMod2019.1, as well as the TensorRT INT8 deployment pipeline discussed in the manuscript, are not included in the current release.

## Repository structure

```text
AVDSDN/
├── data/
│   └── Dataprocess_VMD.py       # Offline VMD preprocessing and data split
├── doc/                         # README figures (tracked by Git)
├── model/
│   ├── AVDSDN.py                # Complete network and classification head
│   ├── Signal_Augment/          # Adaptive VMD mode weighting
│   ├── Conv_Stage/              # CNN stream
│   ├── Transformer_Stage/       # Deformable-attention Transformer stream
│   └── Feature_Fusion/          # Cross-stream attention fusion
├── weights/rml16a/
│   └── weights_rml16.pth        # Pretrained RML2016.10a checkpoint
├── main.py                      # Training and evaluation entry point
├── train.py                     # Training loop and checkpoint selection
├── predict.py                   # Evaluation, metrics, and visualization
├── tools.py                     # Normalization and confusion-matrix utilities
└── requirements.txt
```

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/ZhengYurui326/AVDSDN.git
cd AVDSDN

conda create -n avdsdn python=3.10 -y
conda activate avdsdn
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. If you need a specific CUDA build, install the matching PyTorch package from the [official installation guide](https://pytorch.org/get-started/locally/) before installing the remaining dependencies.

## Dataset preparation

### Option A: download the preprocessed dataset (recommended)

The VMD-preprocessed dataset (1.3 GB) is available as a release asset:

```bash
mkdir -p dataset
wget -O dataset/RML2016.10a_vmd_float32.pkl \
    https://github.com/ZhengYurui326/AVDSDN/releases/download/v1.0.0/RML2016.10a_vmd_float32.pkl
```

### Option B: generate it yourself

#### 1. Download RML2016.10a

Download `RML2016.10a_dict.pkl` from the [DeepSig dataset page](https://www.deepsig.ai/datasets/). The dataset contains 220,000 examples from 11 modulation classes, with SNR values from -20 dB to 18 dB and 128 I/Q samples per example.

#### 2. Generate the VMD representation

VMD is performed offline because decomposing the complete dataset on CPU is computationally expensive. From the repository root, run:

```bash
mkdir -p dataset
python -c "from data.Dataprocess_VMD import process_with_vmd; process_with_vmd('/path/to/RML2016.10a_dict.pkl', 'dataset/RML2016.10a_vmd_float32.pkl')"
```

The preprocessing configuration used by this release is:

| Parameter | Value |
|---|---:|
| Number of modes (`K`) | 5 |
| VMD bandwidth constraint (`alpha`) | 10 |
| Noise tolerance (`tau`) | 0 |
| Initialization | Uniformly distributed center frequencies |
| Convergence tolerance | `1e-7` |

The generated file is intentionally excluded from Git because of its size.

## Evaluation

The repository includes `weights/rml16a/weights_rml16.pth`. Evaluation is enabled by default in `main.py`:

```bash
python main.py --data dataset/RML2016.10a_vmd_float32.pkl
```

The script automatically uses `cuda:0` when CUDA is available and otherwise falls back to CPU. Evaluation produces:

- overall accuracy and mean F1 score;
- accuracy for every SNR level;
- per-modulation accuracy;
- overall and per-SNR confusion matrices;
- CSV metric files under `acc/<timestamp>/`;
- figures under `figure/<timestamp>/`.

## Training

Training and evaluation are controlled near the top of `main.py`:

```python
train_enabled = True
eval_enabled = True
```

Then run the same entry point:

```bash
python main.py --data dataset/RML2016.10a_vmd_float32.pkl
```

Default training configuration:

| Setting | Value |
|---|---:|
| Train / validation / test split | 60% / 20% / 20% per modulation–SNR block |
| Random seed | 3409 |
| Epochs | 200 |
| Batch size | 256 |
| Optimizer | AdamW |
| Initial learning rate | 0.005 |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 6 |
| Loss | Cross-entropy |
| Early-stopping patience | 100 epochs |

The best checkpoint is selected by validation loss and saved to `weights/rml16a/<timestamp>/`. Training logs are written to `training_loss/<timestamp>/`.

## Reproducibility notes

- Inputs are L2-normalized over the complete sample before being passed to AVDSDN.
- Labels are stored as one-hot vectors by the data loader and converted to class indices during training and evaluation.
- The pretrained checkpoint is a plain PyTorch `state_dict` and is loaded with an exact architecture match.
- The current classifier is configured for the 11 classes in RML2016.10a.
- VMD preprocessing is deterministic for fixed input data and configuration; the train/validation/test split is controlled by the seed in `main.py`.

## Citation

If you find this project useful in your research, please cite the manuscript:

```bibtex
@misc{zheng2026avdsdn,
  title  = {Synergistic Dual-stream Neural Network Based on Adaptive Variational Mode Decomposition Denoising for Automatic Modulation Classification},
  author = {Zheng, Yurui and Huang, Sai and Zhang, Pengcheng and Zhang, Yifan and Feng, Zhiyong},
  year   = {2026},
  note   = {Manuscript}
}
```

Publication metadata and a paper link will be added after the final version becomes publicly available.

## Acknowledgements

This implementation builds on ideas and open-source resources from the broader AMC and deformable-attention communities. In particular:

- RML2016.10a is provided by [DeepSig](https://www.deepsig.ai/datasets/).
- The data-processing workflow was adapted from [leena201818/radiom](https://github.com/leena201818/radiom).
- The deformable-attention design is inspired by [DAT: Vision Transformer with Deformable Attention](https://github.com/LeapLabTHU/DAT).

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions about the code or paper, please open a GitHub issue or contact the authors at Beijing University of Posts and Telecommunications.
