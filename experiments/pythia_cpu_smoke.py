"""CPU-only smoke test on real Pythia checkpoints.

This script measures exact sectional curvature of the final decoded intervention
manifold and scores ambient output-space composition under three connections.
It intentionally uses Pythia-14M by default; larger checkpoints are a later
confirmation step, not a prerequisite for validating the method.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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


def stable_control_seed(base_seed: int, revision: str, design_name: str) -> int:
    """Derive an order-independent control seed for one checkpoint and design."""

    key = f"{base_seed}\0{revision}\0{design_name}".encode("utf-8")
    digest = hashlib.blake2b(
        key,
        digest_size=8,
        person=b"predgeom",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def run_checkpoint(
    model_name: str,
    revision: str,
    cache_dir: Path,
    random_planes: int,
    seed: int,
    control_mode: str,
    metric_eigenvalue_rtol: float,
    plane_gram_rtol: float,
    solve_residual_rtol: float,
    solve_forward_error_rtol: float,
    same_plane_curvature_rtol: float,
    vocabulary_chunk_size: int,
    chunking_curvature_rtol: float,
    eigenspace_rtol: float,
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
    checkpoint_results: list[dict[str, Any]] = []

    for design in FACTORIALS:
        control_seed = stable_control_seed(seed, revision, design["name"])
        rng = np.random.default_rng(control_seed)
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
            geometry = SoftmaxHessianGeometry(
                unembedding,
                p00,
                eigenvalue_rtol=metric_eigenvalue_rtol,
                plane_gram_rtol=plane_gram_rtol,
                solve_residual_rtol=solve_residual_rtol,
                solve_forward_error_rtol=solve_forward_error_rtol,
            )
            semantic = geometry.sectional_curvature(
                direction_a,
                direction_b,
            )
            orthonormal_u, orthonormal_v = geometry.fisher_orthonormalize_plane(
                direction_a,
                direction_b,
            )
            orthonormal_semantic = geometry.sectional_curvature(
                orthonormal_u,
                orthonormal_v,
            )
            same_plane_error = abs(
                semantic.sectional_curvature
                - orthonormal_semantic.sectional_curvature
            )
            same_plane_scale = max(abs(semantic.sectional_curvature), 1.0)
            if same_plane_error > same_plane_curvature_rtol * same_plane_scale:
                raise ValueError(
                    "sectional curvature changed after same-plane "
                    "Fisher orthonormalization: "
                    f"{same_plane_error:.3e} > "
                    f"{same_plane_curvature_rtol * same_plane_scale:.3e}"
                )
            chunked_semantic = geometry.sectional_curvature_chunked(
                direction_a,
                direction_b,
                chunk_size=vocabulary_chunk_size,
            )
            chunking_error = abs(
                semantic.sectional_curvature
                - chunked_semantic.sectional_curvature
            )
            chunking_scale = max(
                abs(semantic.sectional_curvature),
                abs(chunked_semantic.sectional_curvature),
                1.0,
            )
            if chunking_error > chunking_curvature_rtol * chunking_scale:
                raise ValueError(
                    "full and vocabulary-chunked curvature disagree: "
                    f"{chunking_error:.3e} > "
                    f"{chunking_curvature_rtol * chunking_scale:.3e}"
                )
            control_curvatures = []
            for _ in range(random_planes):
                if control_mode == "spectrum-block":
                    random_u, random_v = geometry.spectrum_matched_control_plane(
                        direction_a,
                        direction_b,
                        rng,
                        eigenspace_rtol=eigenspace_rtol,
                    )
                else:
                    random_u, random_v = geometry.random_fisher_orthonormal_plane(
                        rng
                    )
                control_curvatures.append(
                    geometry.sectional_curvature(
                        random_u,
                        random_v,
                    ).sectional_curvature
                )
            greater_count = sum(
                value >= semantic.sectional_curvature
                for value in control_curvatures
            )
            less_count = sum(
                value <= semantic.sectional_curvature
                for value in control_curvatures
            )
            if control_curvatures:
                greater_tail_rank = (greater_count + 1) / (
                    len(control_curvatures) + 1
                )
                less_tail_rank = (less_count + 1) / (
                    len(control_curvatures) + 1
                )
                two_sided_tail_rank = min(
                    1.0,
                    2.0 * min(greater_tail_rank, less_tail_rank),
                )
            else:
                greater_tail_rank = None
                less_tail_rank = None
                two_sided_tail_rank = None
            curvature: dict[str, Any] = {
                "feasible": True,
                "semantic_sectional_curvature": semantic.sectional_curvature,
                "semantic_plane_gram_determinant": semantic.gram_determinant,
                "semantic_plane_normalized_gram_determinant": (
                    semantic.normalized_gram_determinant
                ),
                "metric_condition_number": semantic.metric_condition_number,
                "metric_min_eigenvalue": semantic.metric_min_eigenvalue,
                "metric_max_eigenvalue": semantic.metric_max_eigenvalue,
                "solve_relative_residual": semantic.solve_relative_residual,
                "solve_condition_weighted_residual": (
                    semantic.solve_condition_weighted_residual
                ),
                "same_plane_orthonormalization_abs_error": same_plane_error,
                "same_plane_orthonormalization_scaled_error": (
                    same_plane_error / same_plane_scale
                ),
                "vocabulary_chunk_size": vocabulary_chunk_size,
                "chunked_sectional_curvature": (
                    chunked_semantic.sectional_curvature
                ),
                "chunking_curvature_abs_error": chunking_error,
                "chunking_curvature_scaled_error": (
                    chunking_error / chunking_scale
                ),
                "control_mode": control_mode,
                "control_seed": control_seed,
                "eigenspace_rtol": (
                    eigenspace_rtol
                    if control_mode == "spectrum-block"
                    else None
                ),
                "eigenspace_band_sizes": (
                    [
                        stop - start
                        for start, stop in geometry.eigenvalue_bands(
                            eigenspace_rtol=eigenspace_rtol
                        )
                    ]
                    if control_mode == "spectrum-block"
                    else None
                ),
                "control_plane_curvatures": control_curvatures,
                "control_plane_mean": (
                    float(np.mean(control_curvatures))
                    if control_curvatures
                    else None
                ),
                "control_tail_rank_greater": greater_tail_rank,
                "control_tail_rank_less": less_tail_rank,
                "control_tail_rank_two_sided": two_sided_tail_rank,
                "tail_rank_calibration_condition": (
                    "calibrated as a randomization p-value only if the "
                    "semantic-plane statistic is conditionally invariant under "
                    "the declared block-orthogonal control group"
                    if control_mode == "spectrum-block"
                    else "calibrated as a p-value only under the declared null "
                    "that the semantic plane is Fisher-Haar distributed"
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
        "control_planes_per_context": random_planes,
        "control_mode": control_mode,
        "metric_eigenvalue_rtol": metric_eigenvalue_rtol,
        "plane_gram_rtol": plane_gram_rtol,
        "solve_residual_rtol": solve_residual_rtol,
        "solve_forward_error_rtol": solve_forward_error_rtol,
        "same_plane_curvature_rtol": same_plane_curvature_rtol,
        "vocabulary_chunk_size": vocabulary_chunk_size,
        "chunking_curvature_rtol": chunking_curvature_rtol,
        "eigenspace_rtol": eigenspace_rtol,
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
    parser.add_argument(
        "--control-mode",
        choices=("spectrum-block", "fisher-haar"),
        default="spectrum-block",
    )
    parser.add_argument("--metric-eigenvalue-rtol", type=float, default=1e-12)
    parser.add_argument("--plane-gram-rtol", type=float, default=1e-4)
    parser.add_argument("--solve-residual-rtol", type=float, default=1e-10)
    parser.add_argument("--solve-forward-error-rtol", type=float, default=1e-4)
    parser.add_argument("--same-plane-curvature-rtol", type=float, default=1e-10)
    parser.add_argument("--vocabulary-chunk-size", type=int, default=4096)
    parser.add_argument("--chunking-curvature-rtol", type=float, default=1e-7)
    parser.add_argument("--eigenspace-rtol", type=float, default=1e-6)
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
    if args.vocabulary_chunk_size <= 0:
        raise ValueError("--vocabulary-chunk-size must be positive")
    positive_tolerances = {
        "--metric-eigenvalue-rtol": args.metric_eigenvalue_rtol,
        "--plane-gram-rtol": args.plane_gram_rtol,
        "--solve-residual-rtol": args.solve_residual_rtol,
        "--solve-forward-error-rtol": args.solve_forward_error_rtol,
        "--same-plane-curvature-rtol": args.same_plane_curvature_rtol,
        "--chunking-curvature-rtol": args.chunking_curvature_rtol,
    }
    for name, value in positive_tolerances.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not np.isfinite(args.eigenspace_rtol)
        or args.eigenspace_rtol < 0.0
        or args.eigenspace_rtol >= 1.0
    ):
        raise ValueError("--eigenspace-rtol must be finite and in [0, 1)")
    torch.set_num_threads(args.threads)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": (
            "secondary finite-chord final-decoder Fisher curvature "
            "and connection diagnostic"
        ),
        "device": "cpu",
        "checkpoints": [
            run_checkpoint(
                args.model,
                revision,
                args.cache_dir,
                args.random_planes,
                args.seed,
                args.control_mode,
                args.metric_eigenvalue_rtol,
                args.plane_gram_rtol,
                args.solve_residual_rtol,
                args.solve_forward_error_rtol,
                args.same_plane_curvature_rtol,
                args.vocabulary_chunk_size,
                args.chunking_curvature_rtol,
                args.eigenspace_rtol,
            )
            for revision in args.revisions
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
