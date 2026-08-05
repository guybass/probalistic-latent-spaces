"""CPU-only smoke test on real Pythia checkpoints.

This script measures exact sectional curvature of the final decoded intervention
manifold and scores ambient output-space composition under three connections.
It intentionally uses Pythia-14M by default; larger checkpoints are a later
confirmation step, not a prerequisite for validating the method.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from predictive_geometry.benchmark import evaluate_quadrilateral
from predictive_geometry.softmax import SoftmaxHessianGeometry


FACTORIALS = (
    {
        "name": "number_x_time",
        "p00": "Today the cat is",
        "p10": "Today the cats are",
        "p01": "Yesterday the cat was",
        "p11": "Yesterday the cats were",
    },
    {
        "name": "gender_x_number",
        "p00": "The man is",
        "p10": "The men are",
        "p01": "The woman is",
        "p11": "The women are",
    },
    {
        "name": "animal_x_number",
        "p00": "The cat is",
        "p10": "The cats are",
        "p01": "The dog is",
        "p11": "The dogs are",
    },
)


def extract(
    model: Any,
    tokenizer: Any,
    text: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]], float]:
    encoded = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        output = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[-1][0, -1]
        logits = output.logits[0, -1]
        reconstructed = model.get_output_embeddings()(hidden)
    reconstruction_error = float(torch.max(torch.abs(logits - reconstructed)))
    probabilities = torch.softmax(logits.double(), dim=-1)
    top_probabilities, top_indices = torch.topk(probabilities, k=5)
    top_tokens = [
        {
            "token": tokenizer.decode([int(index)]),
            "probability": float(probability),
        }
        for probability, index in zip(top_probabilities, top_indices, strict=True)
    ]
    return (
        hidden.detach().cpu().double().numpy(),
        probabilities.detach().cpu().numpy(),
        top_tokens,
        reconstruction_error,
    )


def run_checkpoint(
    model_name: str,
    revision: str,
    cache_dir: Path,
    random_planes: int,
    seed: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
        dtype=torch.float32,
    )
    model.eval()
    unembedding = (
        model.get_output_embeddings().weight.detach().cpu().double().numpy()
    )
    rng = np.random.default_rng(seed)
    checkpoint_results: list[dict[str, Any]] = []

    for design in FACTORIALS:
        extracted = {
            corner: extract(model, tokenizer, design[corner])
            for corner in ("p00", "p10", "p01", "p11")
        }
        h00, p00, top_tokens, reconstruction_error = extracted["p00"]
        h10, p10, _, _ = extracted["p10"]
        h01, p01, _, _ = extracted["p01"]
        _, p11, _, _ = extracted["p11"]
        direction_a = h10 - h00
        direction_b = h01 - h00

        geometry: SoftmaxHessianGeometry | None = None
        try:
            geometry = SoftmaxHessianGeometry(unembedding, p00)
            semantic = geometry.sectional_curvature(
                direction_a,
                direction_b,
            )
            random_curvatures = []
            for _ in range(random_planes):
                random_u, random_v = geometry.random_fisher_orthonormal_plane(rng)
                random_curvatures.append(
                    geometry.sectional_curvature(
                        random_u,
                        random_v,
                    ).sectional_curvature
                )
            curvature: dict[str, Any] = {
                "feasible": True,
                "semantic_sectional_curvature": semantic.sectional_curvature,
                "semantic_plane_gram_determinant": semantic.gram_determinant,
                "metric_condition_number": semantic.metric_condition_number,
                "metric_min_eigenvalue": semantic.metric_min_eigenvalue,
                "metric_max_eigenvalue": semantic.metric_max_eigenvalue,
                "solve_relative_residual": semantic.solve_relative_residual,
                "random_plane_curvatures": random_curvatures,
                "random_plane_mean": (
                    float(np.mean(random_curvatures))
                    if random_curvatures
                    else None
                ),
            }
        except ValueError as error:
            curvature = {"feasible": False, "failure_reason": str(error)}

        entropy = float(-np.dot(p00, np.log(p00)))
        checkpoint_results.append(
            {
                "design": design,
                "base_entropy_nats": entropy,
                "top_next_tokens": top_tokens,
                "logit_reconstruction_max_abs_error": reconstruction_error,
                "decoded_manifold_curvature": curvature,
                "ambient_connection_composition": evaluate_quadrilateral(
                    p00,
                    p10,
                    p01,
                    p11,
                ),
            }
        )
        if geometry is not None:
            del geometry
        gc.collect()

    result = {
        "model": model_name,
        "revision": revision,
        "hidden_dim": int(unembedding.shape[1]),
        "vocabulary_size": int(unembedding.shape[0]),
        "random_seed": seed,
        "random_planes_per_context": random_planes,
        "results": checkpoint_results,
    }
    del model, tokenizer, unembedding
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-14m")
    parser.add_argument(
        "--revisions",
        nargs="+",
        default=["step0", "step143000"],
    )
    parser.add_argument("--random-planes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/hf_cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia_14m_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_planes < 0:
        raise ValueError("--random-planes cannot be negative")
    torch.set_num_threads(args.threads)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "exact final-decoder Fisher curvature and connection transfer",
        "device": "cpu",
        "checkpoints": [
            run_checkpoint(
                args.model,
                revision,
                args.cache_dir,
                args.random_planes,
                args.seed + index,
            )
            for index, revision in enumerate(args.revisions)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
