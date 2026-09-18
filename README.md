# AVDSDN

Official PyTorch implementation of **"AVDSDN: [Paper Title]"** — Automatic Modulation Classification (AMC) on the RadioML 2016.10a dataset.

> TODO: replace `[Paper Title]` with the final paper title and add the publication venue.

## Overview

AVDSDN classifies modulation schemes from I/Q signal frames by combining:

- **VMD-based signal augmentation** (`model/Signal_Augment/`): learnable weighted combination of Variational Mode Decomposition modes.
- **Multi-frequency conv stage** (`model/Conv_Stage/`): depthwise-separable convolutions with ECA channel attention.
- **Deformable-attention transformer stage** (`model/Transformer_Stage/`): 1D deformable attention over sequence features.
- **Feature fusion** (`model/Feature_Fusion/`): channel + spatial attention fusion of the conv and transformer branches.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0 (CUDA recommended)

```bash
pip install -r requirements.txt
```

## Dataset preparation

1. Download **RML2016.10a** from [DeepSig](https://www.deepsig.ai/datasets/) (`RML2016.10a_dict.pkl`).
2. Generate the VMD-decomposed dataset (this runs Variational Mode Decomposition over the whole dataset and is slow):

```bash
python -c "from data.Dataprocess_VMD import process_with_vmd; process_with_vmd('RML2016.10a_dict.pkl', 'RML2016.10a_vmd_float32.pkl')"
```

3. Place the generated file at `dataset/RML2016.10a_vmd_float32.pkl` (default location), or point to it anywhere with `--data`:

```bash
python main.py --data /path/to/RML2016.10a_vmd_float32.pkl
```

If the dataset is not found, `main.py` exits early with instructions.

## Usage

All configuration is at the top of `main.py`:

| Variable | Meaning |
|---|---|
| `train_enabled` | Set `True` to train; weights and logs are saved to `weights/rml16a/<timestamp>/` and `training_loss/<timestamp>/` |
| `eval_enabled` | Set `True` to evaluate; metrics and figures are saved to `acc/<timestamp>/` and `figure/<timestamp>/` |
| `eval_weight_filepath` | Checkpoint to load for evaluation (a pre-trained model is provided at `weights/rml16a/weights_rml16.pth`) |
| `--data` | Path to the VMD-preprocessed dataset (default: `dataset/RML2016.10a_vmd_float32.pkl`) |

Run:

```bash
python main.py
```

Training uses AdamW with `ReduceLROnPlateau` scheduling, early stopping (`stop_num` epochs without validation-loss improvement), and keeps the best-validation-loss checkpoint. Evaluation reports overall accuracy, per-SNR confusion matrices, per-modulation accuracy and F1 scores.

## Results

> TODO: add the main results table from the paper.

## Citation

If you find this code useful, please cite:

```bibtex
@article{zhengyurui2026avdsdn,
  title  = {AVDSDN: [Paper Title]},
  author = {Zheng, Yurui},
  year   = {2026}
}
```

## Acknowledgements

- Data processing is adapted from [leena201818/radiom](https://github.com/leena201818/radiom).
- The deformable attention implementation is adapted from the DAT release ([Vision Transformer with Deformable Attention](https://github.com/LeapLabTHU/DAT)).

## License

This project is released under the [MIT License](LICENSE).
