# CS5494 Assignment 1

Reproduce the FVSBN, PixelCNN and GAN experiments on MNIST.

1. Download the three Python files and the separate `mnist-data.zip` file into one directory.
2. From that directory, run `python -m zipfile -e mnist-data.zip .`. This creates `data/MNIST/raw/`.
3. Use Python with a CUDA-enabled PyTorch and matching torchvision, plus matplotlib. The report used Python 3.13.5, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128 and matplotlib 3.10.0.
4. Run the commands below in order. The training script defaults to CUDA. In Colab, select a GPU runtime.

    python train.py all
    python analyze.py
    python plot_results.py

The training command writes checkpoints, loss histories, timing summaries and samples to `runs/`. The analysis and plotting commands produce the coverage statistics and figures. For a short installation check, run `python train.py all --smoke --output smoke`.
