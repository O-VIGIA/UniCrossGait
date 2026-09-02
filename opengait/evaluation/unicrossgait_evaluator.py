"""SUSTech1K cross-modal evaluator for UniCrossGait.

The implementation follows the standard SUSTech1K protocol used in the paper:
state-varying sequences of one modality are probes and normal ``00-nm``
sequences of the opposite modality are the gallery.  Identical-view pairs are
excluded by default.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from utils import get_msg_mgr


SUSTECH1K_PROBES = {
    "Normal": ("01-nm",),
    "Bag": ("bg",),
    "Clothing": ("cl",),
    "Carrying": ("cr",),
    "Umbrella": ("ub",),
    "Uniform": ("uf",),
    "Occlusion": ("oc",),
    "Night": ("nt",),
    "Overall": ("01", "02", "03", "04"),
}
SUSTECH1K_GALLERY = ("00-nm",)


def _contains_any(values: np.ndarray, tokens: Iterable[str]) -> np.ndarray:
    values = values.astype(str)
    mask = np.zeros(values.shape, dtype=bool)
    for token in tokens:
        mask |= np.char.find(values, str(token)) >= 0
    return mask


def _part_cosine_distance(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Mean cosine distance over OpenGait's local parts."""

    if probe.ndim == 2:
        probe = probe[:, :, None]
    if gallery.ndim == 2:
        gallery = gallery[:, :, None]
    if probe.ndim != 3 or gallery.ndim != 3:
        raise ValueError("embeddings must be [samples, channels, parts]")
    if probe.shape[1:] != gallery.shape[1:]:
        raise ValueError("probe and gallery embedding dimensions must match")

    eps = np.finfo(np.float32).eps
    probe = probe.astype(np.float32, copy=False)
    gallery = gallery.astype(np.float32, copy=False)
    probe = probe / np.maximum(np.linalg.norm(probe, axis=1, keepdims=True), eps)
    gallery = gallery / np.maximum(
        np.linalg.norm(gallery, axis=1, keepdims=True), eps
    )
    similarity = np.einsum("ncp,mcp->nmp", probe, gallery).mean(axis=2)
    return 1.0 - similarity


def _rank_accuracy(
    probe_features: np.ndarray,
    gallery_features: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    rank: int,
) -> float:
    distance = _part_cosine_distance(probe_features, gallery_features)
    effective_rank = min(int(rank), gallery_features.shape[0])
    nearest = np.argsort(distance, axis=1)[:, :effective_rank]
    matches = gallery_labels[nearest] == probe_labels[:, None]
    return float(matches.any(axis=1).mean() * 100.0)


def evaluate_unicrossgait(
    data,
    dataset,
    metric="cos",
    modes: Sequence[str] = ("2d3d", "3d2d"),
    ranks: Sequence[int] = (1, 5),
    exclude_identical_view=True,
):
    """Evaluate both UniCrossGait retrieval directions on SUSTech1K."""

    if dataset != "SUSTech1K":
        raise KeyError(
            "This reference evaluator implements SUSTech1K only; adapt the "
            "probe/gallery masks for other datasets."
        )
    if metric != "cos":
        raise ValueError("UniCrossGait inference uses cosine similarity")

    features = {
        "2d": np.asarray(data["embeddings_2d"]),
        "3d": np.asarray(data["embeddings_3d"]),
    }
    labels = np.asarray(data["labels"])
    sequence_types = np.asarray(data["types"]).astype(str)
    views = np.asarray(data["views"]).astype(str)
    view_values = sorted(np.unique(views).tolist())

    directions = {"2d3d": ("2d", "3d"), "3d2d": ("3d", "2d")}
    gallery_sequence_mask = _contains_any(sequence_types, SUSTECH1K_GALLERY)
    results = {}
    messages = []

    for mode in modes:
        if mode not in directions:
            raise ValueError("modes may contain only '2d3d' and '3d2d'")
        probe_modality, gallery_modality = directions[mode]
        for condition, tokens in SUSTECH1K_PROBES.items():
            condition_mask = _contains_any(sequence_types, tokens)
            for rank in ranks:
                view_pair_scores = []
                for probe_view in view_values:
                    probe_mask = condition_mask & (views == probe_view)
                    if not probe_mask.any():
                        continue
                    for gallery_view in view_values:
                        if exclude_identical_view and probe_view == gallery_view:
                            continue
                        gallery_mask = gallery_sequence_mask & (views == gallery_view)
                        if not gallery_mask.any():
                            continue
                        view_pair_scores.append(
                            _rank_accuracy(
                                features[probe_modality][probe_mask],
                                features[gallery_modality][gallery_mask],
                                labels[probe_mask],
                                labels[gallery_mask],
                                rank,
                            )
                        )

                if not view_pair_scores:
                    continue
                score = float(np.mean(view_pair_scores))
                key = "scalar/test_accuracy/{}/{}@R{}".format(
                    mode, condition, int(rank)
                )
                results[key] = score
                messages.append("{} {}@R{}: {:.2f}%".format(mode, condition, rank, score))

    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("=== UniCrossGait cross-modal evaluation ===")
    msg_mgr.log_info("\n".join(messages))
    return results


__all__ = ["evaluate_unicrossgait"]
