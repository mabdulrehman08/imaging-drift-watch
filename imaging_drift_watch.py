"""Train a tiny CPU CNN and replay a simulated deployment."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.stats import ks_2samp
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from reporting import MONITORS, alert_history, format_summary, write_reports


WEEKS = 8
DRIFT_WEEK = 5
BATCH_SIZE = 48
REVIEW_SIZE = 24
LABEL_DELAY = 1
BASE_PREVALENCE = 0.60
DRIFT_PREVALENCE = 0.30
PREDICTION_CUTOFF = 0.50
REFERENCE_SIZE = 300
BOOTSTRAPS = 400
PCA_COMPONENTS = 4


@dataclass
class Split:
    images: torch.Tensor
    labels: np.ndarray


def make_split(images, labels, pixel_max):
    return Split(torch.as_tensor(images).float().unsqueeze(1) / pixel_max,
                 np.asarray(labels).reshape(-1).astype(np.int64))


def load_splits(dataset, cache, seed):
    fallback = None
    if dataset != "digits":
        from medmnist import PneumoniaMNIST

        cache.mkdir(parents=True, exist_ok=True)
        previous_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(30)
            train = PneumoniaMNIST(split="train", root=str(cache), download=True)
            reference = PneumoniaMNIST(split="val", root=str(cache))
            stream = PneumoniaMNIST(split="test", root=str(cache))
        except (OSError, RuntimeError, ValueError) as error:
            if dataset == "pneumoniamnist":
                raise
            detail = " ".join(str(error).split()).encode("ascii", "backslashreplace").decode("ascii")
            fallback = f"{type(error).__name__}: {detail}"
            print(f"PneumoniaMNIST unavailable. Using sklearn digits. {fallback}",
                  file=sys.stderr, flush=True)
        else:
            return ("pneumoniamnist", fallback,
                    [make_split(part.imgs, part.labels, 255)
                     for part in (train, reference, stream)])
        finally:
            socket.setdefaulttimeout(previous_timeout)

    digits = load_digits()
    labels = digits.target % 2  # Odd is the positive class.
    indices = np.arange(len(labels))
    development, stream = train_test_split(
        indices, test_size=0.35, stratify=labels, random_state=seed)
    train, reference = train_test_split(
        development, test_size=0.30, stratify=labels[development], random_state=seed)
    return ("digits", fallback,
            [make_split(digits.images[idx], labels[idx], 16)
             for idx in (train, reference, stream)])


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(64, 24), nn.ReLU(),
        )
        self.classifier = nn.Linear(24, 1)

    def forward(self, images):
        embeddings = self.features(images)
        return self.classifier(embeddings).squeeze(1), embeddings


def train_cnn(train, epochs, seed):
    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    labels = torch.from_numpy(train.labels).float()
    loader = DataLoader(TensorDataset(train.images, labels), batch_size=64,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(1 - labels.mean()) / labels.mean())
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, targets in loader:
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        print(f"Epoch {epoch}/{epochs} | train loss {total_loss / len(labels):.4f}",
              flush=True)
    return model.eval()


@torch.inference_mode()
def predict(model, images):
    logits, embeddings = model(images)
    return logits.sigmoid().numpy(), embeddings.numpy()


def make_reference(labels, rng):
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    # Respect the available reference cases in both datasets.
    size = min(REFERENCE_SIZE, int(len(positive) / BASE_PREVALENCE),
               int(len(negative) / (1 - BASE_PREVALENCE)))
    n_positive = round(size * BASE_PREVALENCE)
    return np.concatenate((rng.choice(positive, n_positive, replace=False),
                           rng.choice(negative, size - n_positive, replace=False)))


def weekly_cases(labels, rng):
    pools = [rng.permutation(np.flatnonzero(labels == label)) for label in (0, 1)]
    offsets = [0, 0]
    for week in range(1, WEEKS + 1):
        prevalence = BASE_PREVALENCE if week < DRIFT_WEEK else DRIFT_PREVALENCE
        n_positive = round(BATCH_SIZE * prevalence)
        cases = []
        for label, count in enumerate((BATCH_SIZE - n_positive, n_positive)):
            stop = offsets[label] + count
            if stop > len(pools[label]):
                raise ValueError(f"Insufficient class {label} cases for week {week}")
            cases.append(pools[label][offsets[label]:stop])
            offsets[label] = stop
        yield week, rng.permutation(np.concatenate(cases))


def scanner_shift(images, rng):
    contrast = 0.65 * images.pow(1.8) + 0.12
    blurred = F.avg_pool2d(F.pad(contrast, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    noise = torch.from_numpy(rng.normal(0, 0.08, size=tuple(images.shape))
                             .astype(np.float32))
    return (blurred + noise).clamp(0, 1)


def unlabeled_metrics(probabilities, projected, reference_probs, reference_projected):
    confidence = np.maximum(probabilities, 1 - probabilities)
    reference_confidence = np.maximum(reference_probs, 1 - reference_probs)
    return {
        "confidence_ks": float(ks_2samp(confidence, reference_confidence).statistic),
        "positive_rate": float(np.mean(probabilities >= PREDICTION_CUTOFF)),
        "embedding_ks": float(max(
            ks_2samp(batch_pc, reference_pc).statistic
            for batch_pc, reference_pc in zip(projected.T, reference_projected.T))),
    }


def review_metrics(probabilities, labels):
    predictions = probabilities >= PREDICTION_CUTOFF
    both_classes = np.unique(labels).size == 2
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "auroc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "kappa": (float(cohen_kappa_score(labels, predictions))
                  if both_classes or np.any(predictions != labels) else None),
    }


def calibrate(reference_probs, reference_projected, labels, rng):
    bootstrap = {key: [] for key in MONITORS}
    for _ in range(BOOTSTRAPS):
        batch = rng.integers(len(labels), size=BATCH_SIZE)
        review = rng.integers(len(labels), size=REVIEW_SIZE)
        metrics = {
            **unlabeled_metrics(reference_probs[batch], reference_projected[batch],
                                reference_probs, reference_projected),
            **review_metrics(reference_probs[review], labels[review]),
        }
        for key, value in metrics.items():
            if value is not None:
                bootstrap[key].append(value)
    return {
        key: {bound: float(np.quantile(bootstrap[key], quantile))
              for bound, quantile in limits.items()}
        for key, limits in {
            "confidence_ks": {"high": 0.95},
            "positive_rate": {"low": 0.025, "high": 0.975},
            "embedding_ks": {"high": 0.95},
            "accuracy": {"low": 0.05},
            "auroc": {"low": 0.05},
            "kappa": {"low": 0.05},
        }.items()
    }


def simulate(model, stream, pca, reference_probs, reference_projected, rngs):
    case_rng, scanner_rng, review_rng = rngs
    pending = {}
    weeks = []
    for week, indices in weekly_cases(stream.labels, case_rng):
        images = stream.images[indices]
        if week >= DRIFT_WEEK:
            images = scanner_shift(images, scanner_rng)
        probabilities, embeddings = predict(model, images)
        metrics = unlabeled_metrics(probabilities, pca.transform(embeddings),
                                    reference_probs, reference_projected)
        review = review_rng.choice(BATCH_SIZE, REVIEW_SIZE, replace=False)
        pending[week + LABEL_DELAY] = (week, probabilities[review], indices[review])
        weeks.append({
            "week": week,
            "batch": {"n": len(indices), "metrics": metrics},
            "review": release_review(stream, pending.pop(week, None)),
        })
    # The final batch can only be scored when its labels arrive.
    for week, payload in sorted(pending.items()):
        weeks.append({"week": week, "batch": None,
                      "review": release_review(stream, payload)})
    return weeks


def release_review(stream, payload):
    if payload is None:
        return None
    source_week, probabilities, indices = payload
    labels = stream.labels[indices]
    return {"source_week": source_week, "n": len(indices),
            "metrics": review_metrics(probabilities, labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("auto", "pneumoniamnist", "digits"),
                        default="auto", help="auto tries PneumoniaMNIST, then digits")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.seed < 2**32:
        parser.error("epochs must be positive and seed must be between 0 and 2**32 - 1")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    seeds = np.random.SeedSequence(args.seed).spawn(5)
    ref_rng, calibration_rng, *stream_rngs = map(np.random.default_rng, seeds)

    dataset, fallback, (train, reference, stream) = load_splits(
        args.dataset, args.cache_dir, args.seed)
    print(f"Dataset: {dataset} | seed {args.seed} | CPU", flush=True)
    model = train_cnn(train, args.epochs, args.seed)
    reference_indices = make_reference(reference.labels, ref_rng)
    reference_labels = reference.labels[reference_indices]
    reference_probs, reference_embeddings = predict(model, reference.images[reference_indices])
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
    pca.fit(reference_embeddings)
    reference_projected = pca.transform(reference_embeddings)
    limits = calibrate(reference_probs, reference_projected, reference_labels, calibration_rng)
    weeks = simulate(model, stream, pca, reference_probs, reference_projected, stream_rngs)
    report = {
        "dataset": dataset, "fallback_reason": fallback, "seed": args.seed,
        "epochs": args.epochs, "training_cases": len(train.labels),
        "reference_cases": len(reference_indices),
        "reference_metrics": {
            "positive_rate": float(np.mean(reference_probs >= PREDICTION_CUTOFF)),
            **review_metrics(reference_probs, reference_labels),
        },
        "config": {
            "drift_week": DRIFT_WEEK, "deployment_weeks": WEEKS,
            "batch_size": BATCH_SIZE, "review_size": REVIEW_SIZE,
            "label_delay_weeks": LABEL_DELAY,
            "target_prevalence_before": BASE_PREVALENCE,
            "target_prevalence_after": DRIFT_PREVALENCE,
            "prediction_cutoff": PREDICTION_CUTOFF, "bootstrap_draws": BOOTSTRAPS,
            "embedding_pca_components": PCA_COMPONENTS,
        },
        "thresholds": limits,
        "weeks": weeks,
        "monitors": alert_history(weeks, limits, DRIFT_WEEK),
    }
    write_reports(report, args.out_dir)
    print(format_summary(report))
    print(f"Wrote {args.out_dir / 'report.md'}, {args.out_dir / 'report.json'}, "
          f"and {args.out_dir / 'timeline.png'}")


if __name__ == "__main__":
    main()
