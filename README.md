# CS5494 Assignment 1: MNIST Digit Generation

## Assignment requirements

Train a Fully Visible Sigmoid Belief Network (FVSBN) and a neural autoregressive model on binarized MNIST, then train a GAN on the original grayscale MNIST. The report documents the models, training progress, generated samples, and training and inference times.

## Models and experiments

| Task | Model | Method and evaluation |
| --- | --- | --- |
| 1.1 | FVSBN | A masked logistic regression per pixel estimates its Bernoulli probability from preceding pixels in row-major order. Compare 8 and 24 epochs. |
| 1.2 | PixelCNN | A 32-channel masked CNN uses a Mask A input convolution and three Mask B layers. Vary depth (3 vs. 1 Mask B layers), width (32 vs. 16 channels), training set size (55,000 vs. 10,000), and the first-layer kernel (3 × 3 vs. 7 × 7). |
| 1.3 | MLP GAN | Generate 28 × 28 grayscale digits from 64-dimensional Gaussian noise. Compare the baseline objective and a feature-matching variant across three training seeds. |

The autoregressive runs share a fixed MNIST split and binarization rule. Validation NLL selects checkpoints. Test NLL, uncurated samples, and runtime characterize the primary comparisons. For GANs, inspect loss curves and fixed-noise grids, and evaluate predicted digit coverage using the same classifier and 2,000 latent draws per checkpoint.

## Reproduce the experiments

Clone the repository or save [train.py](train.py), [analyze.py](analyze.py), [plot_results.py](plot_results.py), and [mnist-data.zip](mnist-data.zip) in one directory. Use Python with a CUDA-enabled PyTorch build, matching torchvision, and matplotlib. The runs used Python 3.13.5, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, and matplotlib 3.10.0. On Colab, select a GPU runtime.

```bash
python -m zipfile -e mnist-data.zip .
python train.py all
python analyze.py
python plot_results.py
```

The dataset extracts to `data/MNIST/raw/`. Training saves checkpoints, histories, timings, and sample grids to `runs/`. For a quick installation check, run `python train.py all --smoke --output smoke`.

### Follow-up experiments

Run the primary experiments first, then run the commands below from the same directory. The original GAN pair uses seed 5494. The two new pairs vary training randomness while keeping the data split and evaluation latents fixed.

```bash
python train.py fvsbn --ar-epochs 24 --output runs/supplemental/fvsbn-24
python train.py pixelcnn --first-kernel 7 --output runs/supplemental/pixel-7
python train.py gan --seed 5495 --output runs/supplemental/seed-5495
python train.py gan_features --seed 5495 --output runs/supplemental/seed-5495
python train.py gan --seed 5496 --output runs/supplemental/seed-5496
python train.py gan_features --seed 5496 --output runs/supplemental/seed-5496
python analyze.py --extra-gan-root runs/supplemental
```

See [supplemental-results.json](supplemental-results.json) for measured likelihood, timing, and per-seed coverage results.


