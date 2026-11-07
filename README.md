# CS5494 Assignment 1: MNIST Digit Generation

## Assignment requirements

Train a Fully Visible Sigmoid Belief Network (FVSBN) and a neural autoregressive model on binarized MNIST, then train a GAN on the original grayscale MNIST. The report documents the models, training progress, generated samples, and training and inference times, with code and data links for reproduction.

## Models

| Task | Model | Method |
| --- | --- | --- |
| 1.1 | FVSBN | A logistic regression for each pixel predicts its Bernoulli probability from preceding pixels in row-major order. A strictly lower-triangular mask enforces the ordering. |
| 1.2 | PixelCNN | A 32-channel CNN uses a Mask A input convolution and three Mask B layers to predict binary pixels without seeing the current or future input pixels. |
| 1.3 | MLP GAN | A generator maps 64-dimensional Gaussian noise to 28 × 28 grayscale images. A discriminator learns to distinguish real and generated images. |

## Experimental design

- **Autoregressive comparison.** Train FVSBN and PixelCNN on the same 55,000 binarized images and select checkpoints using a 5,000-image validation split. Compare test negative log-likelihood, generated samples, and training and sampling times.
- **PixelCNN ablations.** Vary the number of Mask B layers (3 vs. 1), channel width (32 vs. 16), and training set size (55,000 vs. 10,000) one at a time. Compare validation likelihood and runtime.
- **GAN stability.** Compare the baseline GAN with a feature-matching variant using the same architecture and training budget. Inspect loss curves and fixed-noise sample grids across epochs. Use a classifier trained on real MNIST to estimate digit coverage on 2,000 generated samples per variant.

## Reproduce the experiments

1. Save [train.py](train.py), [analyze.py](analyze.py), [plot_results.py](plot_results.py), and [mnist-data.zip](mnist-data.zip) in the same directory.
2. Use Python with a CUDA-enabled PyTorch build, matching torchvision, and matplotlib. The original runs used Python 3.13.5, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, and matplotlib 3.10.0. On Colab, select a GPU runtime.
3. Run these commands from that directory:

```bash
python -m zipfile -e mnist-data.zip .
python train.py all
python analyze.py
python plot_results.py
```

The dataset extracts to `data/MNIST/raw/`. Training saves checkpoints, loss histories, timing summaries, and samples to `runs/`. Analysis and plotting produce coverage statistics and figures. For a quick installation check, run `python train.py all --smoke --output smoke`.
