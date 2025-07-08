#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from botorch.acquisition.monte_carlo import qPosteriorStandardDeviation
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

torch.set_default_dtype(torch.double)


@dataclass
class Param:
    spec: Dict

    def __post_init__(self):
        t = self.spec["type"]
        if t in ("range", "int"):
            self.dim = 1
        elif t == "choice" and not self.spec.get("is_ordered", False):
            self.dim = len(self.spec["values"])
        else:
            self.dim = 1

    @property
    def name(self):
        return self.spec["name"]

    @property
    def kind(self):
        return self.spec["type"]

    def encode(self, v) -> torch.Tensor:
        if self.kind in ("range", "int"):
            lo, hi = self.spec["bounds"]
            return torch.tensor([(v - lo) / (hi - lo)])
        vals = self.spec["values"]
        idx = vals.index(v)
        if self.spec.get("is_ordered", False):
            denom = max(len(vals) - 1, 1)
            return torch.tensor([idx / denom])
        vec = torch.zeros(len(vals), dtype=torch.double)
        vec[idx] = 1.0
        return vec

    def decode(self, t: torch.Tensor):
        x = t if self.dim > 1 else t.item()
        if self.kind == "range":
            lo, hi = self.spec["bounds"]
            return x * (hi - lo) + lo
        if self.kind == "int":
            lo, hi = self.spec["bounds"]
            return int(round(x * (hi - lo) + lo))
        vals = self.spec["values"]
        if self.spec.get("is_ordered", False):
            idx = int(round(x * (len(vals) - 1)))
        else:
            idx = int(torch.argmax(t).item())
        return vals[idx]


def encode_sample(params: List[Param], s: Dict) -> torch.Tensor:
    return torch.cat([p.encode(s[p.name]) for p in params])


def decode_vector(params: List[Param], v: torch.Tensor) -> Dict:
    out, i = {}, 0
    for p in params:
        out[p.name] = p.decode(v[i : i + p.dim])
        i += p.dim
    return out


def sobol_batch(params: List[Param], n: int, seed: int) -> List[Dict]:
    dim = len(params)
    X = SobolEngine(dim, scramble=True, seed=seed).draw(n)
    return [decode_vector(params, row) for row in X]


def bayesopt_batch(
    params: List[Param],
    X: torch.Tensor,
    Y: torch.Tensor,
    Yvar: torch.Tensor,
    n: int,
    seed: int,
) -> List[Dict]:
    dim = X.shape[-1]
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

    model = SingleTaskGP(X, Y, Yvar)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(torch.Size([512]), seed=seed)
    acq = qPosteriorStandardDeviation(model, sampler=sampler)

    X_batch, _ = optimize_acqf(
        acq,
        bounds=bounds,
        q=n,
        num_restarts=10,
        raw_samples=4096,
        options={"seed": seed},
        sequential=True,
    )
    return [decode_vector(params, x) for x in X_batch.squeeze(0)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose new samples for a DOE experiment."
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the JSON file containing the DOE space and samples.",
    )
    parser.add_argument(
        "--samples", type=int, default=1, help="Number of samples to propose."
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=1e-3,
        help="Noise level for the Gaussian Process model.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    torch.manual_seed(seed=args.seed)
    file_path = Path(args.file)
    data = json.loads(file_path.read_text())

    params = [Param(p) for p in data["parameters"]]
    samples = data["samples"]

    if any([s.get("result") is None for s in samples]):
        raise ValueError(
            f"Fill all `result` fields in {file_path} before running this script."
        )

    if not samples:
        new = sobol_batch(params, args.samples, args.seed)
    else:
        X = torch.stack([encode_sample(params, s) for s in samples])
        Y = torch.tensor([[float(s["result"])] for s in samples])
        Yvar = torch.tensor([[args.noise]] * len(samples))
        new = bayesopt_batch(params, X, Y, Yvar, args.samples, args.seed)

    samples.extend([n | {"result": None} for n in new])
    data["samples"] = samples
    file_path.write_text(json.dumps(data, indent=2))
    print(f"Appended {len(new)} sample(s) → {file_path}")


if __name__ == "__main__":
    main()
