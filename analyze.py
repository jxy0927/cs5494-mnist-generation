"""Evaluate generated digit coverage with a fixed MNIST classifier."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.datasets import MNIST

from train import ROOT, SEED, LATENT, make_gan, rng_on, save_json

RUNS = ROOT / "runs"


@torch.inference_mode()
def predictions(model, images):
    return torch.cat([model(batch).softmax(dim=1) for batch in images.split(256)])


def describe(probabilities):
    confidence, labels = probabilities.max(dim=1)
    counts = labels.bincount(minlength=10)
    shares = counts.float() / len(labels)
    return {
        "counts": counts.tolist(),
        "entropy_bits": float(-torch.special.xlogy(shares, shares).sum() / torch.log(torch.tensor(2.0, device=shares.device))),
        "top_share": float(shares.max()),
        "mean_confidence": float(confidence.mean()),
    }


def probe(extra_gan_root=None):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = MNIST(ROOT / "data", train=True, download=True)
    held_out = MNIST(ROOT / "data", train=False, download=True)
    split = torch.randperm(len(original), generator=rng_on("cpu", 0))[5000:]
    x = original.data[split].to(device).float().div(255)
    y = original.targets[split].to(device)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                          nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    shuffle = rng_on(device, 10)
    for _ in range(5):
        for indices in torch.randperm(len(x), device=device, generator=shuffle).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
    model.eval()
    actual = held_out.data.to(device).float().div(255)
    real_probabilities = predictions(model, actual)
    accuracy = (real_probabilities.argmax(1).cpu() == held_out.targets).float().mean()
    results = {"test_accuracy": float(accuracy), "real": describe(real_probabilities)}
    with torch.inference_mode():
        gen, _ = make_gan()
        gen.to(device).eval()
        variants = ("gan", "gan_features")
        # Use the same 2,000 independent latents for every checkpoint.
        noise = torch.randn(2000, LATENT, device=device, generator=rng_on(device, 11))

        def coverage(folder):
            scores = {}
            for run in variants:
                checkpoint = torch.load(folder / run / "model.pt", map_location=device)
                gen.load_state_dict(checkpoint["generator"])
                generated = (gen(noise) + 1) / 2
                scores[run] = describe(predictions(model, generated))
            return scores

        results.update(coverage(RUNS))
        if extra_gan_root is not None:
            repeated = {str(SEED): {run: results[run] for run in variants}}
            for folder in sorted(extra_gan_root.glob("seed-*")):
                seed = int(folder.name.removeprefix("seed-"))
                repeated[str(seed)] = coverage(folder)
            save_json(extra_gan_root / "gan_seed_probe.json", repeated)
    save_json(RUNS / "gan_probe.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-gan-root", type=Path,
                        help="Optional directory containing paired GAN runs in seed-* folders.")
    probe(parser.parse_args().extra_gan_root)


