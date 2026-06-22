#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraFT — Graph-based Filament Tracing
======================================
Segments filament-like structures in a grayscale image, builds a graph
representation, traces individual filaments via a constrained DFS, and
optionally evaluates the result against a ground-truth dataset.

Usage
-----
    python GraFT_workflow_still_improved_iterations1.py \\
        --image  path/to/image.png \\
        --output path/to/output_dir \\
        [--gt    path/to/ground_truth_folder]

Run with ``--help`` for a full list of tunable parameters.

Author: senyuz
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import skimage
import skimage.io as io
from scipy import ndimage
from skimage.morphology import dilation, square

import utilsGraFT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GraFT: Graph-based Filament Tracing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required I/O
    p.add_argument("--image",  required=True,
                   help="Path to the input image (PNG / TIFF / JPG).")
    p.add_argument("--output", required=True,
                   help="Output directory (created if it does not exist).")
    p.add_argument("--gt",     default=None,
                   help="Ground-truth folder (optional). "
                        "Each file should be a binary mask of one filament.")

    # Segmentation
    p.add_argument(
        "--preprocess",
        choices=["full", "binary"],
        default="full",
        help=(
            "Preprocessing pipeline. "
            "'full' = Gaussian + Frangi tubeness + CLAHE + median + Otsu "
            "hysteresis + skeletonize (use for real fluorescence microscopy). "
            "'binary' = skip preprocessing and only skeletonize + prune "
            "(use for synthetic / hand-drawn / already-binarised inputs)."
        ),
    )
    p.add_argument("--sigma",      type=float, default=1.0,
                   help="Gaussian sigma for tubeness filtering (full only).")
    p.add_argument("--small",      type=float, default=50.0,
                   help="Minimum connected-component size (pixels) to keep.")
    p.add_argument("--thresh_top", type=float, default=0.5,
                   help="Hysteresis lower-threshold ratio (full only).")

    # Graph construction
    p.add_argument("--size",    type=int,   default=6,
                   help="Kernel size for node condensation.")
    p.add_argument("--eps",     type=int,   default=200,
                   help="Epsilon for the Visvalingam–Whyatt node-placement "
                        "algorithm. Larger values simplify more aggressively, "
                        "placing FEWER bend nodes.")

    # Node extension (thickness / intensity discontinuity inserts).
    # All four reduce the number of inserted nodes when increased.
    p.add_argument("--min-edge-pixels", type=int,   default=5,
                   help="Minimum pixel spacing between inserted nodes along an "
                        "edge. Larger => fewer inserted nodes.")
    p.add_argument("--merge-radius",    type=int,   default=10,
                   help="Radius for merging a new node onto a nearby existing "
                        "node. Larger => fewer distinct nodes.")
    p.add_argument("--thickness-factor", type=float, default=1.5,
                   help="Multiplier on the per-component thickness std used as "
                        "the insertion threshold. Larger => fewer inserted nodes.")
    p.add_argument("--intensity-factor", type=float, default=1.5,
                   help="Multiplier on the per-component intensity std used as "
                        "the insertion threshold. Larger => fewer inserted nodes.")
    p.add_argument("--node-extension", action="store_true",
                   help="Enable the thickness/intensity node-insertion step. "
                        "OFF by default: on the simple-shapes benchmark this step "
                        "adds spurious nodes that cause over/under-segmentation, "
                        "so the default reproduces the original GraFT (VW-only) "
                        "node placement. Enable it for noisier real images where "
                        "the extra nodes may help.")

    # DFS constraints
    p.add_argument("--angle",   type=float, default=140.0,
                   help="Minimum angle (degrees) allowed between consecutive edges.")
    p.add_argument("--overlap", type=int,   default=4,
                   help="Maximum permitted edge overlap between traced filaments.")

    # DFS scoring weights. An edge is kept when its total score >= score-threshold.
    p.add_argument("--intensity-coeff", type=int, default=3,
                   help="Score reward when an edge's intensity stays within "
                        "tolerance of the previous edge. Higher => intensity "
                        "continuity matters more (helps trace through corners).")
    p.add_argument("--thickness-coeff", type=int, default=3,
                   help="Score reward when an edge's thickness stays within "
                        "tolerance of the previous edge. Higher => thickness "
                        "continuity matters more (helps trace through corners).")
    p.add_argument("--angle-coeff", type=int, default=5,
                   help="Score reward for a straight continuation and penalty "
                        "for a sharp turn. Lower => sharp corners are more "
                        "likely to be traced through; higher => the tracer "
                        "prefers straight continuations (e.g. through crossings).")
    p.add_argument("--score-threshold", type=int, default=8,
                   help="Minimum total DFS score required to keep an edge.")
    p.add_argument("--disable-angle-penalty", action="store_true",
                   help="Drop the score penalty applied when an angle is badly "
                        "violated (angle x 1.4 < gate). With the penalty off, a "
                        "sharp corner whose intensity/thickness stay within "
                        "tolerance can still be kept (e.g. score 4+4+0 >= 8), so "
                        "filaments trace through sharp turns. OFF by default "
                        "(penalty active), preserving original behaviour.")
    p.add_argument("--either-constraint-coeff", type=int, default=0,
                   help="Demonstration override. When >0, add this (large) bonus "
                        "to an edge's score whenever EITHER the intensity OR the "
                        "thickness constraint is satisfied, forcing the edge to "
                        "be kept regardless of the angle term (e.g. 15 >> "
                        "score-threshold). Shows that the intensity/thickness "
                        "constraints alone can drive tracing through sharp "
                        "angles. 0 = off (default).")

    # Standalone "arbitrary" trace (fixed tolerances, run alongside the sweep)
    p.add_argument("--arb-intensity-tol", type=float, default=0.17,
                   help="Fixed intensity tolerance for the standalone 'arbitrary' "
                        "trace. Unlike the sweep (which derives this dynamically "
                        "from a percentile), this value is used as-is.")
    p.add_argument("--arb-thickness-tol", type=float, default=0.27,
                   help="Fixed thickness tolerance for the standalone 'arbitrary' "
                        "trace.")
    p.add_argument("--best-metric",
                   choices=["overall_f1", "overall_iou", "mcc_f1", "mcc"],
                   default="overall_f1",
                   help="Metric used to pick the best sweep combination "
                        "(requires --gt).")
    p.add_argument("--arb-only", action="store_true",
                   help="Skip the 121-combination percentile sweep and only run "
                        "the standalone 'arbitrary' trace. Useful for fast "
                        "parameter experiments (e.g. node-count tuning).")

    return p.parse_args()


def build_params(args: argparse.Namespace) -> dict:
    """Collect all tunable algorithm parameters into one dictionary."""
    return {
        # Segmentation
        "preprocess": args.preprocess,
        "sigma":      args.sigma,
        "small":      args.small,
        "thresh_top": args.thresh_top,
        # Graph construction
        "size": args.size,
        "eps":  args.eps,
        # DFS constraint coefficients
        "angleA":          args.angle,
        "overlap":         args.overlap,
        "intensity_coeff": args.intensity_coeff,
        "thickness_coeff": args.thickness_coeff,
        "angle_coeff":     args.angle_coeff,
        "score_threshold": args.score_threshold,
        "apply_angle_penalty": not args.disable_angle_penalty,
        "either_constraint_coeff": args.either_constraint_coeff,
        # Set to None so each connected component gets its own dynamic threshold.
        "intensity_tolerance": None,
        "thickness_tolerance": None,
        # Node-extension parameters (now CLI-tunable; see notes in parse_args).
        "min_edge_pixels":  args.min_edge_pixels,
        "merge_radius":     args.merge_radius,
        "thickness_factor": args.thickness_factor,
        "intensity_factor": args.intensity_factor,
        # NOTE: insert_nodes_by_thickness_intensity_dynamic recomputes these
        # internally (factor * per-component std), so the values below are
        # passed only for signature compatibility and have no effect.
        "thickness_thresh": 2.0,
        "intensity_thresh": 0.25,
        # Standalone arbitrary-parameter trace + best-combo selection
        "arb_intensity_tol": args.arb_intensity_tol,
        "arb_thickness_tol": args.arb_thickness_tol,
        "best_metric":       args.best_metric,
        "node_extension":    args.node_extension,
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def create_output_dirs(base_dir: str) -> None:
    """Create required output sub-directories."""
    # Only the directories the workflow actually writes to. (Earlier versions
    # also created circ_stat/, mov/ and plots/, which were never populated.)
    for name in ("n_graphs", "evaluation_plots", "key_graphs"):
        os.makedirs(os.path.join(base_dir, name), exist_ok=True)


def load_and_pad_image(image_path: str):
    """Load a grayscale image and add a 1-pixel zero-padding on all sides."""
    image = io.imread(image_path)
    return image, np.pad(image, 1, mode="constant")


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Linearly rescale pixel values to [0, 255]."""
    lo, hi = image.min(), image.max()
    return 255.0 * (image - lo) / (hi - lo)


def fit_to_shape(arr: np.ndarray, target_shape) -> np.ndarray:
    """Center-align ``arr`` to ``target_shape`` by padding smaller axes and
    cropping larger ones.

    Used to reconcile small size differences between the ground-truth masks and
    the predicted-label image before a pixel-wise comparison. Each axis is
    handled independently: if the array is shorter than the target on an axis it
    is zero-padded symmetrically, and if it is longer it is centre-cropped. This
    keeps both images on an origin-consistent (centre-aligned) coordinate frame.
    It only makes sense for *small* offsets where the two images are otherwise
    co-registered.
    """
    result = arr
    for axis, (cur, tgt) in enumerate(zip(result.shape, target_shape)):
        if cur < tgt:                       # pad this axis symmetrically
            before = (tgt - cur) // 2
            after = tgt - cur - before
            pad = [(0, 0)] * result.ndim
            pad[axis] = (before, after)
            result = np.pad(result, pad, mode="constant")
        elif cur > tgt:                     # centre-crop this axis
            start = (cur - tgt) // 2
            sl = [slice(None)] * result.ndim
            sl[axis] = slice(start, start + tgt)
            result = result[tuple(sl)]
    return result


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, *path_parts: str) -> None:
    """Save *fig* to the joined path and close it."""
    fig.savefig(os.path.join(*path_parts), bbox_inches="tight")
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, col: str, title: str,
                 out_dir: str, filename: str) -> None:
    """Pivot ``df`` and save a seaborn heatmap; skip silently if column is absent."""
    if col not in df.columns:
        log.warning("Column '%s' not found — skipping heatmap.", col)
        return
    pivot = df.pivot(
        index="percentile_intensity",
        columns="percentile_thickness",
        values=col,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Percentile Thickness")
    ax.set_ylabel("Percentile Intensity")
    save_fig(fig, out_dir, filename)
    log.info("Saved heatmap → %s", filename)


# Maps the user-facing --best-metric choice to the column name in the metrics
# table. "Best" always means the maximum of the chosen column.
BEST_METRIC_COLUMN = {
    "overall_f1":  "overall_overall_f1_score",
    "overall_iou": "overall_overall_iou",
    "mcc_f1":      "MCC_f1_score",
    "mcc":         "MCC_mcc",
}


def evaluate_prediction(graphTagg, posL, imageAnnotated, image_shape,
                        ground_truth_labels, gt_filament_ids,
                        out_subfolder=None, title="Predicted"):
    """Turn one tagged graph into a label image, compare it to the ground truth,
    and (optionally) save the per-prediction evaluation figures.

    This is the single source of truth for the evaluation step — both the
    percentile sweep and the standalone arbitrary/best traces call it, so they
    always score predictions the same way.

    Parameters
    ----------
    graphTagg : nx.Graph
        Tagged graph produced by ``dfs_constrained``.
    posL : dict
        Node-position lookup for drawing/coordinate extraction.
    imageAnnotated : np.ndarray
        Annotated skeleton image (nodes > 1), used to build the node→pixel map.
    image_shape : tuple
        Shape of the predicted-label image (the padded input shape).
    ground_truth_labels : np.ndarray
        Relabelled ground-truth label image, already reconciled to ``image_shape``.
    gt_filament_ids : array-like
        Ground-truth filament ids that go with ``ground_truth_labels``.
    out_subfolder : str or None
        If given, the six evaluation figures are written here. If None, only the
        metrics are computed (no figures).
    title : str
        Title for the predicted panel of the pred-vs-gt comparison figure.

    Returns
    -------
    dict
        Flattened metrics with ``overall_*`` and ``MCC_*`` keys.
    """
    graphTagF_adjusted, _ = utilsGraFT.adjust_filament_tags(graphTagg)

    # Build a node → (row, col) lookup consistent with the graph. np.where returns
    # pixels in row-major order and sp.ndimage.label assigns labels in the same
    # order, so pos_dict[node_id] = (row, col).
    node_coords = np.transpose(np.where(imageAnnotated > 1))
    pos_dict    = {i: (int(node_coords[i, 0]), int(node_coords[i, 1]))
                   for i in range(len(node_coords))}

    filament_coords  = utilsGraFT.extract_filament_coordinates_from_graph(
        graphTagF_adjusted, pos_dict
    )
    predicted_labels = utilsGraFT.create_predicted_label_image(
        filament_coords, image_shape
    )

    predicted_filament_ids = np.unique(predicted_labels)
    predicted_filament_ids = predicted_filament_ids[predicted_filament_ids > 0]

    confusion_matrix = utilsGraFT.compute_confusion_matrix_multi_layer(
        predicted_labels, ground_truth_labels,
        predicted_filament_ids, gt_filament_ids,
    )
    matches = utilsGraFT.match_filaments(
        confusion_matrix, predicted_labels, ground_truth_labels,
        predicted_filament_ids, gt_filament_ids,
    )
    bad_matches = utilsGraFT.bad_good_match(
        matches, predicted_filament_ids, gt_filament_ids,
        F1_THRESHOLD=0.70, IOU_THRESHOLD=0.50,
    )
    _, overall_metrics = utilsGraFT.calculate_metrics(
        predicted_labels, ground_truth_labels,
        matches, confusion_matrix,
        predicted_filament_ids, gt_filament_ids,
    )
    mcc_metrics = utilsGraFT.calculate_metrics_with_mcc1(
        confusion_matrix, matches,
        predicted_filament_ids, gt_filament_ids,
        noise_threshold=10,
    )

    if out_subfolder is not None:
        os.makedirs(out_subfolder, exist_ok=True)

        fig_cmp, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(predicted_labels, cmap="jet")
        axes[0].set_title(title)
        axes[0].axis("off")
        axes[1].imshow(ground_truth_labels, cmap="jet")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        fig_cmp.tight_layout()
        save_fig(fig_cmp, out_subfolder, "pred_vs_gt.png")

        save_fig(
            utilsGraFT.visualize_labels(predicted_labels, ground_truth_labels),
            out_subfolder, "visualize_labels.png",
        )
        save_fig(
            utilsGraFT.visualize_overlaps(predicted_labels, ground_truth_labels),
            out_subfolder, "visualize_overlaps.png",
        )
        save_fig(
            utilsGraFT.visualize_matched_filaments(
                predicted_labels, ground_truth_labels,
                matches, predicted_filament_ids, gt_filament_ids, bad_matches,
            ),
            out_subfolder, "visualize_matched_filaments.png",
        )
        save_fig(
            utilsGraFT.visualize_false_positives_negatives(
                predicted_labels, ground_truth_labels,
                matches, predicted_filament_ids, gt_filament_ids,
            ),
            out_subfolder, "visualize_false_positives_negatives.png",
        )
        save_fig(
            utilsGraFT.visualize_bad_matches(
                predicted_labels, ground_truth_labels,
                matches, predicted_filament_ids, gt_filament_ids, bad_matches,
            ),
            out_subfolder, "visualize_bad_matches.png",
        )

    metrics = {}
    metrics.update({f"overall_{k}": v for k, v in overall_metrics.items()})
    metrics.update({f"MCC_{k}": v for k, v in mcc_metrics.items()})
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    params = build_params(args)

    # -----------------------------------------------------------------------
    # 1. Output directories
    # -----------------------------------------------------------------------
    create_output_dirs(args.output)
    eval_dir = os.path.join(args.output, "evaluation_plots")

    # -----------------------------------------------------------------------
    # 2. Load and preprocess image
    # -----------------------------------------------------------------------
    log.info("Loading image: %s", args.image)
    original_image, image_padded = load_and_pad_image(args.image)
    image_normalized = normalize_image(image_padded)

    # -----------------------------------------------------------------------
    # 3. Segmentation and skeletonization
    # -----------------------------------------------------------------------
    if params["preprocess"] == "full":
        log.info("Segmenting (full pipeline: Gaussian + Frangi + CLAHE + "
                 "Otsu hysteresis + skeletonize) …")
        seg_fn = utilsGraFT.segmentation_skeleton
    else:
        log.info("Segmenting (binary pipeline: skeletonize + prune only) …")
        seg_fn = utilsGraFT.segmentation_skeleton_short

    (imageTubeness,
     imageCleaned,
     imageHysteresis,
     imageHysteresisCleaned) = seg_fn(
        image_normalized,
        params["sigma"],
        params["small"],
        params["thresh_top"],
    )

    # Add a second padding layer so all downstream arrays share the same shape.
    # imageHysteresisCleaned is intentionally left at the current size;
    # make_graph_mask pads it internally.
    image_normalized = np.pad(image_normalized, 1, mode="constant")
    imageTubeness    = np.pad(imageTubeness,    1, mode="constant")
    imageCleaned     = np.pad(imageCleaned,     1, mode="constant")
    imageHysteresis  = np.pad(imageHysteresis,  1, mode="constant")
    imageCleaned     = (imageCleaned * 1) > 0
    imA = imageCleaned  # keep a clean copy of the skeleton for drawing

    # -----------------------------------------------------------------------
    # 4. Node detection and VW-based node placement
    # -----------------------------------------------------------------------
    log.info("Detecting junction / endpoint nodes …")
    imageNodes   = utilsGraFT.node_find(imageCleaned)
    node_initial = imageCleaned + imageNodes
    log.info("  nodes after junction/endpoint detection: %d",
             int(np.count_nonzero(node_initial > 1)))

    log.info("Placing additional nodes via Visvalingam–Whyatt (eps=%s) …",
             params["eps"])
    imF, imgBl = utilsGraFT.project_edges(
        node_initial, params["eps"], params["size"]
    )
    log.info("  nodes after VW placement: %d",
             int(np.count_nonzero(imF > 1)))

    # -----------------------------------------------------------------------
    # 5. Node extension: insert nodes at thickness / intensity discontinuities
    # -----------------------------------------------------------------------
    if not params["node_extension"]:
        log.info("Skipping thickness/intensity node insertion "
                 "(default; enable with --node-extension).")
    else:
        log.info("Inserting extra nodes at local thickness / intensity changes …")
        imageHysteresisCleaned_padded = np.pad(imageHysteresisCleaned, 1, mode="constant")
        distance_map              = ndimage.distance_transform_edt(imageHysteresisCleaned_padded)
        pixel_widths              = distance_map * 2
        pixel_widths_skeletonized = skimage.morphology.skeletonize(pixel_widths > 0)
        thickness_map             = pixel_widths * pixel_widths_skeletonized

        imF = utilsGraFT.insert_nodes_by_thickness_intensity_dynamic(
            imF,
            thickness_map,
            image_normalized,
            min_edge_pixels=params["min_edge_pixels"],
            merge_radius=params["merge_radius"],
            thickness_factor=params["thickness_factor"],
            intensity_factor=params["intensity_factor"],
            thickness_thresh=params["thickness_thresh"],
            intensity_thresh=params["intensity_thresh"],
        )
        log.info("  nodes after thickness/intensity insertion: %d",
                 int(np.count_nonzero(imF > 1)))

    # -----------------------------------------------------------------------
    # 6. Build the graph
    # -----------------------------------------------------------------------
    log.info("Building the graph …")
    mask, index_list = utilsGraFT.project_mask(imF)

    ones             = np.ones((3, 3))
    imageNodeCondense = utilsGraFT.node_condense(
        imF - imageCleaned,
        imageCleaned,
        np.ones((params["size"], params["size"])),
    )
    imgInt = dilation((node_initial > 1).astype(int), square(params["size"]))
    imgBlR = (((imgBl > 0).astype(int) - imgInt) > 0).astype(int)
    df_pos = utilsGraFT.condense_mask(index_list, imageNodeCondense, mask, params["size"])

    imgReLab, _ = sp.ndimage.label(imageNodeCondense, structure=ones)
    imageAnnotated = imgReLab + imageCleaned

    if len(df_pos) == 0:
        log.error(
            "No filamentous structures detected. "
            "Try adjusting --small, --sigma, or --thresh_top."
        )
        sys.exit(1)

    gBo, posL = utilsGraFT.make_graph_mask(
        imageAnnotated, image_normalized, mask, df_pos, imageHysteresisCleaned
    )
    gBu   = utilsGraFT.unify_graph(gBo)
    graph_s = utilsGraFT.test_connectivity(gBu)

    fig_untagged = utilsGraFT.draw_graph(imA, graph_s, posL, "Untagged graph")
    save_fig(fig_untagged, args.output, "n_graphs", "untagged_graph.png")

    # -----------------------------------------------------------------------
    # 7. Load ground truth (optional)
    # -----------------------------------------------------------------------
    HAS_GROUND_TRUTH    = False
    ground_truth_layers = None
    if args.gt:
        try:
            gt_list             = utilsGraFT.prepare_ground_truth_layers(args.gt)
            ground_truth_layers = np.stack(gt_list, axis=0)
            log.info("Ground truth loaded — shape %s", ground_truth_layers.shape)
            HAS_GROUND_TRUTH = True
        except Exception as exc:
            log.warning("Could not load ground truth: %s", exc)

    # -----------------------------------------------------------------------
    # 8. Build the line-graph once (reused across all traces)
    # -----------------------------------------------------------------------
    log.info("Building the line-graph …")
    graphD = utilsGraFT.dangling_edges(graph_s.copy())
    lgG    = nx.line_graph(graph_s.copy())
    lgG_V  = utilsGraFT.lG_edgeVal(lgG.copy(), graphD, posL)

    # One DFS-tracing call, parameterised. The line-graph and node positions are
    # fixed for this image, so every trace only varies the tolerances/percentiles.
    def run_dfs(intensity_tol, thickness_tol, pi, pt):
        return utilsGraFT.dfs_constrained(
            graph_s.copy(), lgG_V.copy(), imgBlR, posL,
            params["angleA"], params["overlap"],
            intensity_tol, thickness_tol,
            intensity_coeff=params["intensity_coeff"],
            thickness_coeff=params["thickness_coeff"],
            angle_coeff=params["angle_coeff"],
            score_threshold=params["score_threshold"],
            percentile_intensity=pi, percentile_thickness=pt,
            apply_angle_penalty=params["apply_angle_penalty"],
            either_constraint_coeff=params["either_constraint_coeff"],
        )

    # Relabel + size-reconcile the ground truth ONCE (it is identical for every
    # combination, so there is no reason to redo it inside the sweep loop).
    image_shape         = image_padded.shape
    ground_truth_labels = None
    gt_filament_ids     = None
    if HAS_GROUND_TRUTH:
        ground_truth_labels, gt_filament_ids = utilsGraFT.relabel_ground_truth_layers(
            ground_truth_layers
        )
        # The GT masks may be a few pixels larger *or* smaller than the padded
        # input; centre-align by padding axes that are too small and cropping
        # axes that are too large. (The previous code only padded, so it silently
        # failed when the GT was larger than the prediction.)
        if ground_truth_labels.shape != image_shape:
            log.warning(
                "Ground-truth shape %s != predicted shape %s — centre-aligning "
                "(pad/crop) to match. Metrics assume the two images are "
                "co-registered; a large offset would make them unreliable.",
                ground_truth_labels.shape, image_shape,
            )
            ground_truth_labels = fit_to_shape(ground_truth_labels, image_shape)

    # -----------------------------------------------------------------------
    # 9. Percentile sweep: trace filaments and evaluate every combination
    # -----------------------------------------------------------------------
    results_list = []

    # Track the best-scoring combination (only meaningful with ground truth).
    best_metric_col = BEST_METRIC_COLUMN[params["best_metric"]]
    best_score      = float("-inf")
    best_pi = best_pt = None
    best_graph        = None
    best_metrics      = None

    if args.arb_only:
        log.info("--arb-only set: skipping the percentile sweep.")
    else:
        log.info("Starting percentile sweep (11 × 11 = 121 combinations) …")
        for pi in range(0, 101, 10):
            for pt in range(0, 101, 10):
                log.info("  DFS  percentile_intensity=%3d  percentile_thickness=%3d", pi, pt)

                # Sweep uses dynamic tolerances (None -> derived from percentile).
                graphTagg = run_dfs(
                    params["intensity_tolerance"], params["thickness_tolerance"],
                    pi, pt
                )

                fig_dfs = utilsGraFT.draw_graph_filament_nocolor(
                    image_padded, graphTagg, posL, "", "filament"
                )
                save_fig(fig_dfs, args.output, "n_graphs",
                         f"graph_pi{pi:03d}_pt{pt:03d}.png")

                if not HAS_GROUND_TRUTH:
                    continue

                subfolder = os.path.join(eval_dir, f"pi{pi:03d}_pt{pt:03d}")
                metrics   = evaluate_prediction(
                    graphTagg, posL, imageAnnotated, image_shape,
                    ground_truth_labels, gt_filament_ids,
                    out_subfolder=subfolder, title=f"Predicted (pi={pi}, pt={pt})",
                )
                log.info("    Saved evaluation plots → %s", subfolder)

                row = {"percentile_intensity": pi, "percentile_thickness": pt}
                row.update(metrics)
                results_list.append(row)

                # Remember the best combination by the chosen metric. NaN never
                # beats -inf, so degenerate combos are skipped automatically.
                score = metrics.get(best_metric_col, float("-inf"))
                if score > best_score:
                    best_score, best_pi, best_pt = score, pi, pt
                    best_graph, best_metrics     = graphTagg, metrics

    # -----------------------------------------------------------------------
    # 10. Save aggregate metrics and summary heatmaps
    # -----------------------------------------------------------------------
    if results_list:
        df_results  = pd.DataFrame(results_list)
        excel_path  = os.path.join(args.output, "evaluation_metrics.xlsx")
        df_results.to_excel(excel_path, index=False)
        log.info("Evaluation metrics saved → %s", excel_path)

        save_heatmap(df_results, "overall_overall_f1_score",
                     "Overall F1-Score vs. Percentile Thresholds",
                     args.output, "overall_f1_heatmap.png")
        save_heatmap(df_results, "MCC_f1_score",
                     "MCC F1-Score vs. Percentile Thresholds",
                     args.output, "MCC_f1_heatmap.png")
        save_heatmap(df_results, "MCC_mcc",
                     "MCC vs. Percentile Thresholds",
                     args.output, "MCC_mcc_heatmap.png")

    # -----------------------------------------------------------------------
    # 11. Key graphs: the two "relevant" traces, saved apart from n_graphs/
    #     so they are easy to find:
    #       (a) the best sweep combination (by --best-metric), and
    #       (b) a standalone trace at fixed "arbitrary" tolerances.
    # -----------------------------------------------------------------------
    key_dir  = os.path.join(args.output, "key_graphs")
    key_rows = []

    # (a) Best sweep combination — only available when we evaluated against GT.
    if best_graph is not None:
        title = (f"Best sweep (pi={best_pi}, pt={best_pt}, "
                 f"{params['best_metric']}={best_score:.3f})")
        save_fig(
            utilsGraFT.draw_graph_filament_nocolor(
                image_padded, best_graph, posL, title, "filament"
            ),
            key_dir, f"best_sweep_pi{best_pi:03d}_pt{best_pt:03d}.png",
        )
        log.info("Best sweep combo: pi=%d pt=%d (%s=%.3f) → %s",
                 best_pi, best_pt, params["best_metric"], best_score, key_dir)
        key_rows.append({
            "which": "best_sweep",
            "percentile_intensity": best_pi, "percentile_thickness": best_pt,
            "intensity_tol": None, "thickness_tol": None,
            **best_metrics,
        })

    # (b) Standalone arbitrary trace at fixed tolerances (always produced).
    arb_it, arb_tt = params["arb_intensity_tol"], params["arb_thickness_tol"]
    log.info("Tracing standalone 'arbitrary' filaments "
             "(intensity_tol=%.3f, thickness_tol=%.3f) …", arb_it, arb_tt)
    graph_arb = run_dfs(arb_it, arb_tt, 0, 0)   # fixed tols -> percentiles unused
    arb_title = f"Arbitrary (it={arb_it}, tt={arb_tt})"
    save_fig(
        utilsGraFT.draw_graph_filament_nocolor(
            image_padded, graph_arb, posL, arb_title, "filament",
        ),
        key_dir, "arbitrary_graph.png",
    )
    if HAS_GROUND_TRUTH:
        m_arb = evaluate_prediction(
            graph_arb, posL, imageAnnotated, image_shape,
            ground_truth_labels, gt_filament_ids,
            out_subfolder=os.path.join(key_dir, "arbitrary_eval"),
            title=arb_title,
        )
        log.info("  Arbitrary  F1=%.3f  IoU=%.3f  MCC=%.3f  over-seg=%.2f (%d GT split)",
                 m_arb["overall_overall_f1_score"],
                 m_arb["overall_overall_iou"], m_arb["MCC_mcc"],
                 m_arb["MCC_oversegmentation_ratio"], int(m_arb["MCC_gt_oversegmented"]))
        key_rows.append({
            "which": "arbitrary",
            "percentile_intensity": None, "percentile_thickness": None,
            "intensity_tol": arb_it, "thickness_tol": arb_tt,
            **m_arb,
        })

    if key_rows:
        key_csv = os.path.join(key_dir, "key_metrics.csv")
        pd.DataFrame(key_rows).to_csv(key_csv, index=False)
        log.info("Key-graph metrics saved → %s", key_csv)

    log.info("Processing complete.")


if __name__ == "__main__":
    main()
