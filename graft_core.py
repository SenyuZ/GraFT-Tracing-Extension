#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graft_core — importable pipeline stages for the GraFT tracer.

This module exposes the GraFT pipeline as small, reusable functions so other
front-ends (notably ``gui_app.py``) can drive it in-process without shelling out
to the CLI. It does **not** modify the validated workflow or algorithm code: it
only *calls* functions in ``utilsGraFT`` and reproduces the exact stage ordering
of ``GraFT_workflow_still_improved_iterations1.main()``.

The pipeline is split so the expensive part can be cached:

    build_graph(image, params)  ->  Built     # segmentation + graph (slow)
    trace(built, params)        ->  graphTagg  # one constrained DFS  (fast)
    render_trace(built, graph)  ->  Figure     # matplotlib drawing
    evaluate(built, graph, gt)  ->  (metrics, fig)

A GUI can cache ``build_graph`` keyed on the image + segmentation/graph params,
so moving a *tracing* slider only re-runs the fast ``trace`` step.

NOTE: ``trace`` uses the fixed "arbitrary" tolerances (``arb_intensity_tol`` /
``arb_thickness_tol`` with percentiles 0/0), i.e. the same single trace the CLI
produces with ``--arb-only``. The 121-combo percentile sweep is intentionally
out of scope here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy as sp
from scipy import ndimage
import networkx as nx
from skimage import io, color, img_as_ubyte
from skimage.morphology import dilation, square, skeletonize

import matplotlib
matplotlib.use("Agg")  # headless: figures are returned, never shown
import matplotlib.pyplot as plt

import utilsGraFT

# skimage emits FutureWarnings (remove_small_objects' min_size, square()); the
# calls are correct for the pinned version. Silence the category so the GUI/
# console stays clean (these also appear in the unchanged CLI).
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Default parameters — mirror build_params() in the CLI so the GUI starts from
# the same validated defaults.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    # Segmentation
    "preprocess": "full",
    "sigma": 1.0,
    "small": 50.0,
    "thresh_top": 0.5,
    # Graph construction
    "size": 6,
    "eps": 200,
    # DFS constraint coefficients
    "angleA": 140.0,
    "overlap": 4,
    "intensity_coeff": 3,
    "thickness_coeff": 3,
    "angle_coeff": 5,
    "score_threshold": 8,
    "apply_angle_penalty": True,
    "either_constraint_coeff": 0,
    # Node-extension knobs (only used when node_extension is True)
    "node_extension": False,
    "min_edge_pixels": 5,
    "merge_radius": 10,
    "thickness_factor": 1.5,
    "intensity_factor": 1.5,
    "thickness_thresh": 2.0,   # inert (recomputed internally); kept for signature
    "intensity_thresh": 0.25,  # inert
    # Fixed tolerances for the single ("arbitrary") trace
    "arb_intensity_tol": 0.17,
    "arb_thickness_tol": 0.27,
}

# Params that affect Stage A (segmentation + graph). A GUI cache for build_graph
# should key on these (plus the image); the rest only affect the fast trace.
# (thickness_thresh / intensity_thresh are intentionally excluded: they are inert
# — insert_nodes_by_thickness_intensity_dynamic recomputes them internally — so
# they neither belong in the cache key nor need to be supplied by a caller.)
BUILD_PARAM_KEYS = (
    "preprocess", "sigma", "small", "thresh_top", "size", "eps",
    "node_extension", "min_edge_pixels", "merge_radius",
    "thickness_factor", "intensity_factor",
)


@dataclass
class Built:
    """Everything Stage A produces that later stages need."""
    graph_s: nx.Graph
    lgG_V: nx.Graph
    imgBlR: np.ndarray
    posL: np.ndarray
    imageAnnotated: np.ndarray
    image_padded: np.ndarray   # single-padded original, for drawing + image_shape
    image_shape: tuple


# ---------------------------------------------------------------------------
# Small helpers (faithful copies of the CLI's, so results match exactly)
# ---------------------------------------------------------------------------
def load_grayscale(path_or_bytes) -> np.ndarray:
    """Read an image to a 2-D grayscale array.

    Accepts a filesystem path or a file-like object (e.g. a Streamlit upload).
    RGB(A) inputs are collapsed to grayscale so the downstream 2-D pipeline works;
    images already grayscale are returned unchanged (matching the CLI).
    """
    img = io.imread(path_or_bytes)
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = img_as_ubyte(color.rgb2gray(img))
    return img


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Linearly rescale to [0, 255] (copy of the CLI's normalize_image)."""
    lo, hi = image.min(), image.max()
    return 255.0 * (image - lo) / (hi - lo)


def _fit_to_shape(arr: np.ndarray, target_shape) -> np.ndarray:
    """Centre-align ``arr`` to ``target_shape`` (copy of the CLI's fit_to_shape)."""
    result = arr
    for axis, (cur, tgt) in enumerate(zip(result.shape, target_shape)):
        if cur < tgt:
            before = (tgt - cur) // 2
            after = tgt - cur - before
            pad = [(0, 0)] * result.ndim
            pad[axis] = (before, after)
            result = np.pad(result, pad, mode="constant")
        elif cur > tgt:
            start = (cur - tgt) // 2
            sl = [slice(None)] * result.ndim
            sl[axis] = slice(start, start + tgt)
            result = result[tuple(sl)]
    return result


# ---------------------------------------------------------------------------
# Stage A — segmentation + graph (expensive; cache this)
# ---------------------------------------------------------------------------
def build_graph(image_array: np.ndarray, params: dict) -> Built:
    """Run load→segment→nodes→VW→(insert)→graph→line-graph.

    Mirrors sections 2–8 of the CLI's ``main()`` exactly (same padding and same
    arguments to the same ``utilsGraFT`` functions), so the resulting graph is
    identical to a CLI run with the same parameters.

    Raises ValueError if no filamentous structures are detected.
    """
    p = {**DEFAULT_PARAMS, **params}

    # 2. Load + pad
    image_padded = np.pad(image_array, 1, mode="constant")
    image_normalized = _normalize_image(image_padded)

    # 3. Segmentation + skeletonization
    seg_fn = (utilsGraFT.segmentation_skeleton if p["preprocess"] == "full"
              else utilsGraFT.segmentation_skeleton_short)
    (imageTubeness, imageCleaned,
     imageHysteresis, imageHysteresisCleaned) = seg_fn(
        image_normalized, p["sigma"], p["small"], p["thresh_top"]
    )

    image_normalized = np.pad(image_normalized, 1, mode="constant")
    imageTubeness = np.pad(imageTubeness, 1, mode="constant")
    imageCleaned = np.pad(imageCleaned, 1, mode="constant")
    imageHysteresis = np.pad(imageHysteresis, 1, mode="constant")
    imageCleaned = (imageCleaned * 1) > 0

    # 4. Node detection + VW placement
    imageNodes = utilsGraFT.node_find(imageCleaned)
    node_initial = imageCleaned + imageNodes
    imF, imgBl = utilsGraFT.project_edges(node_initial, p["eps"], p["size"])

    # 5. Optional node extension
    if p["node_extension"]:
        ihc_padded = np.pad(imageHysteresisCleaned, 1, mode="constant")
        distance_map = ndimage.distance_transform_edt(ihc_padded)
        pixel_widths = distance_map * 2
        pixel_widths_skeletonized = skeletonize(pixel_widths > 0)
        thickness_map = pixel_widths * pixel_widths_skeletonized
        imF = utilsGraFT.insert_nodes_by_thickness_intensity_dynamic(
            imF, thickness_map, image_normalized,
            min_edge_pixels=p["min_edge_pixels"], merge_radius=p["merge_radius"],
            thickness_factor=p["thickness_factor"], intensity_factor=p["intensity_factor"],
            thickness_thresh=p["thickness_thresh"], intensity_thresh=p["intensity_thresh"],
        )

    # 6. Build the graph
    mask, index_list = utilsGraFT.project_mask(imF)
    ones = np.ones((3, 3))
    imageNodeCondense = utilsGraFT.node_condense(
        imF - imageCleaned, imageCleaned, np.ones((p["size"], p["size"]))
    )
    imgInt = dilation((node_initial > 1).astype(int), square(p["size"]))
    imgBlR = (((imgBl > 0).astype(int) - imgInt) > 0).astype(int)
    df_pos = utilsGraFT.condense_mask(index_list, imageNodeCondense, mask, p["size"])

    imgReLab, _ = sp.ndimage.label(imageNodeCondense, structure=ones)
    imageAnnotated = imgReLab + imageCleaned

    if len(df_pos) == 0:
        raise ValueError(
            "No filamentous structures detected. Try adjusting 'small', 'sigma', "
            "'thresh_top', or switching the preprocessing mode."
        )

    gBo, posL = utilsGraFT.make_graph_mask(
        imageAnnotated, image_normalized, mask, df_pos, imageHysteresisCleaned
    )
    gBu = utilsGraFT.unify_graph(gBo)
    graph_s = utilsGraFT.test_connectivity(gBu)

    # 8. Line graph (reused by every trace)
    graphD = utilsGraFT.dangling_edges(graph_s.copy())
    lgG = nx.line_graph(graph_s.copy())
    lgG_V = utilsGraFT.lG_edgeVal(lgG.copy(), graphD, posL)

    return Built(
        graph_s=graph_s, lgG_V=lgG_V, imgBlR=imgBlR, posL=posL,
        imageAnnotated=imageAnnotated, image_padded=image_padded,
        image_shape=image_padded.shape,
    )


def count_nodes(built: Built) -> int:
    """Number of graph nodes (handy for a GUI status line)."""
    return built.graph_s.number_of_nodes()


# ---------------------------------------------------------------------------
# Stage B — one constrained DFS (fast)
# ---------------------------------------------------------------------------
def trace(built: Built, params: dict) -> nx.Graph:
    """Trace filaments with the fixed 'arbitrary' tolerances (== CLI --arb-only)."""
    p = {**DEFAULT_PARAMS, **params}
    return utilsGraFT.dfs_constrained(
        built.graph_s.copy(), built.lgG_V.copy(), built.imgBlR, built.posL,
        p["angleA"], p["overlap"],
        p["arb_intensity_tol"], p["arb_thickness_tol"],
        intensity_coeff=p["intensity_coeff"], thickness_coeff=p["thickness_coeff"],
        angle_coeff=p["angle_coeff"], score_threshold=p["score_threshold"],
        percentile_intensity=0, percentile_thickness=0,
        apply_angle_penalty=p["apply_angle_penalty"],
        either_constraint_coeff=p["either_constraint_coeff"],
    )


# ---------------------------------------------------------------------------
# Stage C — render
# ---------------------------------------------------------------------------
def render_trace(built: Built, graphTagg: nx.Graph, title: str = "Traced filaments") -> plt.Figure:
    """Matplotlib figure of the tagged trace over the input image."""
    return utilsGraFT.draw_graph_filament_nocolor(
        built.image_padded, graphTagg, built.posL, title, "filament"
    )


# ---------------------------------------------------------------------------
# Stage D — evaluate against ground truth
# ---------------------------------------------------------------------------
def _predicted_labels(built: Built, graphTagg: nx.Graph):
    """Build the predicted-label image (mirrors evaluate_prediction's head)."""
    graphTagF_adjusted, _ = utilsGraFT.adjust_filament_tags(graphTagg)
    node_coords = np.transpose(np.where(built.imageAnnotated > 1))
    pos_dict = {i: (int(node_coords[i, 0]), int(node_coords[i, 1]))
                for i in range(len(node_coords))}
    filament_coords = utilsGraFT.extract_filament_coordinates_from_graph(
        graphTagF_adjusted, pos_dict
    )
    predicted_labels = utilsGraFT.create_predicted_label_image(
        filament_coords, built.image_shape
    )
    ids = np.unique(predicted_labels)
    return predicted_labels, ids[ids > 0]


def load_ground_truth(gt_folder: str, image_shape) -> tuple:
    """Load + relabel + centre-align the GT masks to ``image_shape``."""
    gt_list = utilsGraFT.prepare_ground_truth_layers(gt_folder)
    layers = np.stack(gt_list, axis=0)
    gt_labels, gt_ids = utilsGraFT.relabel_ground_truth_layers(layers)
    if gt_labels.shape != tuple(image_shape):
        gt_labels = _fit_to_shape(gt_labels, image_shape)
    return gt_labels, gt_ids


def evaluate(built: Built, graphTagg: nx.Graph,
             gt_folder: Optional[str]) -> tuple[Optional[dict], Optional[plt.Figure]]:
    """Compare a trace to ground truth. Returns (metrics, pred-vs-gt figure).

    metrics carries the same flattened keys as the CLI: ``overall_*`` (strict +
    ``*_overlap`` lenient) and ``MCC_*`` (incl. over-segmentation). Returns
    (None, None) when no GT folder is given.
    """
    if not gt_folder:
        return None, None

    gt_labels, gt_ids = load_ground_truth(gt_folder, built.image_shape)
    predicted_labels, pred_ids = _predicted_labels(built, graphTagg)

    confusion = utilsGraFT.compute_confusion_matrix_multi_layer(
        predicted_labels, gt_labels, pred_ids, gt_ids
    )
    matches = utilsGraFT.match_filaments(
        confusion, predicted_labels, gt_labels, pred_ids, gt_ids
    )
    _, overall = utilsGraFT.calculate_metrics(
        predicted_labels, gt_labels, matches, confusion, pred_ids, gt_ids
    )
    mcc = utilsGraFT.calculate_metrics_with_mcc1(
        confusion, matches, pred_ids, gt_ids, noise_threshold=10
    )

    metrics = {}
    metrics.update({f"overall_{k}": v for k, v in overall.items()})
    metrics.update({f"MCC_{k}": v for k, v in mcc.items()})

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].imshow(predicted_labels, cmap="jet")
    axes[0].set_title("Predicted")
    axes[0].axis("off")
    axes[1].imshow(gt_labels, cmap="jet")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")
    fig.tight_layout()
    return metrics, fig
