"""Reproduce the CS5494 image-generation experiments."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent
SEED = 5494
BATCH = 128
SIDE = 28
PIXELS = SIDE ** 2
LATENT = 64
COUNT = 25
CONFIGS = {
    "fvsbn": (0, 0, 55000),
    "pixelcnn": (32, 3, 55000),
    "pixelcnn_shallow": (32, 1, 55000),
    "pixelcnn_narrow": (16, 3, 55000),
    "pixelcnn_subset": (32, 3, 10000),
}


class FVSBN(nn.Module):
    """A raster-ordered logistic regression per output pixel."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(PIXELS, PIXELS))
        self.bias = nn.Parameter(torch.zeros(PIXELS))
        self.register_buffer("mask", torch.ones(PIXELS, PIXELS).tril(-1))

    def forward(self, x):
        return F.linear(x.flatten(1), self.weight * self.mask, self.bias).reshape_as(x)

    @torch.inference_mode()
    def sample(self, count, rng):
        x = self.bias.new_zeros(count, PIXELS)
        for i in range(PIXELS):
            logit = (x[:, :i] * self.weight[i, :i]).sum(1) + self.bias[i]
            x[:, i] = torch.bernoulli(logit.sigmoid(), generator=rng)
        return x.reshape(count, 1, SIDE, SIDE)


class MaskedConv(nn.Conv2d):
    def __init__(self, inputs, outputs, include_center):
        super().__init__(inputs, outputs, 3, padding=1)
        mask = torch.tensor([[1, 1, 1], [1, int(include_center), 0], [0, 0, 0]])
        self.register_buffer("mask", mask.float().reshape(1, 1, 3, 3))

    def forward(self, x):
        # Apply the mask on every pass, including during optimization.
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=1)


class PixelCNN(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        layers = [MaskedConv(1, channels, False), nn.ReLU()]
        for _ in range(depth):
            layers.extend([MaskedConv(channels, channels, True), nn.ReLU()])
        self.layers = nn.Sequential(*layers, nn.Conv2d(channels, 1, 1))

    def forward(self, x):
        return self.layers(x)

    @torch.inference_mode()
    def sample(self, count, rng):
        x = next(self.parameters()).new_zeros(count, 1, SIDE, SIDE)
        for row in range(SIDE):
            for column in range(SIDE):
                probability = self(x)[:, :, row, column].sigmoid()
                x[:, :, row, column] = torch.bernoulli(probability, generator=rng)
        return x


def rng_on(device, offset):
    return torch.Generator(device=device).manual_seed(SEED + offset)


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def load_data(path, device, binary, smoke):
    source = MNIST(path, train=True, download=True)
    test = MNIST(path, train=False, download=True).data
    permutation = torch.randperm(len(source), generator=rng_on("cpu", 0))
    train, valid = source.data[permutation[5000:]], source.data[permutation[:5000]]
    if smoke:
        train, valid, test = train[:256], valid[:128], test[:128]

    def transform(images):
        images = images.unsqueeze(1).to(device)
        return (images >= 128).float() if binary else images.float().div(127.5).sub(1)

    return tuple(transform(part) for part in (train, valid, test))


def batches(data, rng):
    order = torch.randperm(len(data), device=data.device, generator=rng)
    for indices in order.split(BATCH):
        yield data[indices]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_samples(draw, device, folder):
    draw(1)
    times = []
    for _ in range(3):
        sync(device)
        start = perf_counter()
        images = draw(COUNT)
        sync(device)
        times.append(perf_counter() - start)
    save_image(images, folder / "samples.png", nrow=5)
    return sorted(times)[1]


@torch.inference_mode()
def check_causality(model, inputs):
    # Include corners, a row boundary and an interior pixel.
    for i in (0, 27, 28, 405, 782, 783):
        altered = inputs.clone()
        tail = altered.flatten(1)[:, i:]
        tail.copy_(1 - tail)
        torch.testing.assert_close(
            model(inputs).flatten(1)[:, i], model(altered).flatten(1)[:, i],
            atol=1e-6, rtol=1e-5,
        )


@torch.inference_mode()
def evaluate_nll(model, data):
    total = data.new_zeros(())
    for batch in data.split(BATCH):
        total += F.binary_cross_entropy_with_logits(model(batch), batch, reduction="sum")
    return total.item() / len(data)


def train_ar(name, data, device, folder, epochs):
    channels, depth, limit = CONFIGS[name]
    train, valid, test = data
    train = train[:limit]
    model = FVSBN() if name == "fvsbn" else PixelCNN(channels, depth)
    model = model.to(device=device, memory_format=torch.channels_last)
    check_causality(model, valid[:2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    rng = rng_on(device, 1)
    mixed = channels > 0 and device.type == "cuda"
    main_config = name in {"fvsbn", "pixelcnn"}
    best = float("inf")
    history = []
    seconds = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        sync(device)
        start = perf_counter()
        total = train.new_zeros(())
        for batch in batches(train, rng):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=mixed):
                loss = F.binary_cross_entropy_with_logits(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(batch) * PIXELS
        sync(device)
        seconds += perf_counter() - start
        model.eval()
        row = {"epoch": epoch, "train_nll": total.item() / len(train),
               "val_nll": evaluate_nll(model, valid)}
        history.append(row)
        if row["val_nll"] < best:
            best = row["val_nll"]
            best_epoch = epoch
            torch.save(model.state_dict(), folder / "model.pt")
        if main_config and epoch in {1, max(1, epochs // 2), epochs}:
            images = model.sample(10, rng_on(device, 2))
            save_image(images, folder / f"epoch_{epoch:02d}.png", nrow=5)
        save_json(folder / "history.json", history)
        print(name, row, flush=True)

    model.load_state_dict(torch.load(folder / "model.pt", map_location=device))
    check_causality(model, valid[:2])
    sample_rng = rng_on(device, 3)
    latency = measure_samples(lambda n: model.sample(n, sample_rng), device, folder)
    if channels == 0:
        active = PIXELS * (PIXELS - 1) // 2 + PIXELS
    else:
        active = 5 * channels + depth * (5 * channels ** 2 + channels) + channels + 1
    return {
        "train_size": len(train), "valid_size": len(valid), "test_size": len(test),
        "active_parameters": active, "channels": channels, "mask_b_layers": depth,
        "epochs": epochs, "steps": epochs * ((len(train) + BATCH - 1) // BATCH),
        "best_epoch": best_epoch, "val_nll": best,
        "test_nll": evaluate_nll(model, test) if main_config else None,
        "train_seconds": seconds, "sample_seconds": latency, "bf16_training": mixed,
    }


def make_gan():
    generator = nn.Sequential(
        nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, PIXELS), nn.Tanh(), nn.Unflatten(1, (1, SIDE, SIDE)),
    )
    discriminator = nn.Sequential(
        nn.Flatten(), nn.Linear(PIXELS, 512), nn.LeakyReLU(0.2),
        nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1),
    )
    return generator, discriminator


def train_gan(data, device, folder, epochs, feature_weight):
    train = data[0]
    gen, disc = (module.to(device) for module in make_gan())
    features = disc[:-1]
    gen_opt = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    rng = rng_on(device, 1)
    fixed = torch.randn(4, LATENT, device=device, generator=rng_on(device, 2))
    stages = sorted({0, round(epochs / 4), round(epochs / 2), round(epochs * 3 / 4), epochs})

    @torch.inference_mode()
    def render(z):
        return (gen(z) + 1) / 2

    progress = [render(fixed)]
    history = []
    seconds = 0.0
    bce = F.binary_cross_entropy_with_logits
    for epoch in range(1, epochs + 1):
        sync(device)
        start = perf_counter()
        d_total, g_total = train.new_zeros(()), train.new_zeros(())
        for real in batches(train, rng):
            noise = torch.randn(len(real), LATENT, device=device, generator=rng)
            fake = gen(noise)
            disc_opt.zero_grad(set_to_none=True)
            real_logit, fake_logit = disc(real), disc(fake.detach())
            d_loss = (bce(real_logit, torch.ones_like(real_logit))
                      + bce(fake_logit, torch.zeros_like(fake_logit))) / 2
            d_loss.backward()
            disc_opt.step()

            # Keep the gradient to fake images while freezing D for the G update.
            disc.requires_grad_(False)
            gen_opt.zero_grad(set_to_none=True)
            logit = disc(fake)
            g_loss = bce(logit, torch.ones_like(logit))
            if feature_weight:
                # Match minibatch feature means to encourage broader coverage.
                with torch.no_grad():
                    real_mean = features(real).mean(dim=0)
                fake_mean = features(fake).mean(dim=0)
                g_loss = g_loss + feature_weight * F.l1_loss(fake_mean, real_mean)
            g_loss.backward()
            gen_opt.step()
            disc.requires_grad_(True)
            d_total += d_loss.detach() * len(real)
            g_total += g_loss.detach() * len(real)
        sync(device)
        seconds += perf_counter() - start
        row = {"epoch": epoch, "d_loss": d_total.item() / len(train),
               "g_loss": g_total.item() / len(train)}
        history.append(row)
        if epoch in stages:
            progress.append(render(fixed))
        save_json(folder / "history.json", history)
        print("gan", row, flush=True)

    torch.save({"generator": gen.state_dict(), "discriminator": disc.state_dict()},
               folder / "model.pt")
    save_image(torch.cat(progress), folder / "progress.png", nrow=4)
    sample_rng = rng_on(device, 3)

    def draw(count):
        return render(torch.randn(count, LATENT, device=device, generator=sample_rng))

    latency = measure_samples(draw, device, folder)
    return {"train_size": len(train), "epochs": epochs, "stages": stages, "feature_weight": feature_weight,
            "generator_parameters": sum(p.numel() for p in gen.parameters()),
            "discriminator_parameters": sum(p.numel() for p in disc.parameters()),
            "train_seconds": seconds, "sample_seconds": latency}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=[*CONFIGS, "gan", "gan_features", "all"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--ar-epochs", type=int, default=8)
    parser.add_argument("--gan-epochs", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use a CUDA PyTorch build or pass --device cpu.")
    if min(args.ar_epochs, args.gan_epochs) < 1:
        parser.error("Epoch counts must be positive.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    names = ["fvsbn", "pixelcnn", "gan", "pixelcnn_shallow", "pixelcnn_narrow",
             "pixelcnn_subset", "gan_features"] if args.experiment == "all" else [args.experiment]
    current_kind = None
    for name in names:
        binary = name in CONFIGS
        if current_kind != binary:
            data = load_data(args.data, device, binary, args.smoke)
            current_kind = binary
        torch.manual_seed(SEED)
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        epochs = 1 if args.smoke else args.ar_epochs if binary else args.gan_epochs
        if binary:
            metrics = train_ar(name, data, device, folder, epochs)
        else:
            metrics = train_gan(data, device, folder, epochs, 10.0 if name == "gan_features" else 0.0)
        metrics.update({"experiment": name, "seed": SEED, "batch": BATCH,
                        "sample_count": COUNT, "smoke": args.smoke,
                        "device": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
                        "python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__})
        save_json(folder / "metrics.json", metrics)
        print(name, metrics, flush=True)


if __name__ == "__main__":
    main()


