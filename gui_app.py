#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraFT — interactive filament tracer (Streamlit GUI).

A lightweight front-end over the same pipeline as the CLI: set the parameters
with the sliders on the left, hit **Run**, and see the traced image (and, if a
ground-truth folder is given, the evaluation) on the right.

It reuses ``graft_core`` (which in turn only calls ``utilsGraFT``); the CLI and
the algorithm code are untouched. The expensive segmentation + graph build is
cached, so changing a *tracing* parameter and re-running only re-runs the fast
DFS step.

Run with:
    streamlit run gui_app.py
"""

from __future__ import annotations

import hashlib
import io
import time

import streamlit as st

import graft_core as gc

st.set_page_config(page_title="GraFT — Filament Tracer", layout="wide")


# ---------------------------------------------------------------------------
# Cached Stage A. Keyed on (image hash, build-param items); the image array
# itself is passed with a leading underscore so Streamlit does not try to hash
# it. Changing only tracing params -> same key -> cache hit -> no rebuild.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cached_build(img_key: str, build_items: tuple, _image):
    return gc.build_graph(_image, dict(build_items))


def _image_key(raw: bytes, extra: str = "") -> str:
    return hashlib.md5(raw).hexdigest() + extra


# ---------------------------------------------------------------------------
# Sidebar — inputs and parameters
# ---------------------------------------------------------------------------
st.sidebar.title("GraFT controls")

# --- Image + ground truth -------------------------------------------------
st.sidebar.header("Input")
source = st.sidebar.radio("Image source", ["Sample / path", "Upload"], horizontal=True)

uploaded = None
image_path = ""
if source == "Upload":
    uploaded = st.sidebar.file_uploader("Image (PNG / TIFF / JPG)",
                                        type=["png", "tif", "tiff", "jpg", "jpeg"])
else:
    image_path = st.sidebar.text_input(
        "Image path", "data_samples/test_data/simple_shapes.jpg"
    )

gt_folder = st.sidebar.text_input(
    "Ground-truth folder (optional)",
    "data_samples/labeled_ground_truth/simple_shapes",
    help="Folder of label1.png, label2.png … masks. Leave blank to skip evaluation.",
).strip()

# --- Segmentation & graph (Stage A — rebuild on change) -------------------
with st.sidebar.expander("Segmentation & graph", expanded=True):
    preprocess = st.radio(
        "Preprocessing", ["full", "binary"],
        help="full = Frangi tubeness (thin fluorescence). "
             "binary = skeletonize only (thick / synthetic).",
        horizontal=True,
    )
    sigma = st.slider("sigma (Frangi scale)", 0.5, 5.0, 1.0, 0.1)
    small = st.slider("small (min component px)", 0, 500, 50, 5)
    thresh_top = st.slider("thresh_top (hysteresis ratio)", 0.0, 1.0, 0.5, 0.05)
    eps = st.slider("eps (VW node placement)", 0, 1000, 200, 10)
    size = st.slider("size (node condensation kernel)", 1, 15, 6, 1)
    node_extension = st.checkbox("Node extension (thickness/intensity inserts)", value=False)
    if node_extension:
        min_edge_pixels = st.slider("min_edge_pixels", 1, 50, 5, 1)
        merge_radius = st.slider("merge_radius", 1, 50, 10, 1)
        thickness_factor = st.slider("thickness_factor", 0.5, 5.0, 1.5, 0.1)
        intensity_factor = st.slider("intensity_factor", 0.5, 5.0, 1.5, 0.1)
    else:
        min_edge_pixels, merge_radius = 5, 10
        thickness_factor, intensity_factor = 1.5, 1.5

# --- Tracing (Stage B — fast re-trace) ------------------------------------
with st.sidebar.expander("Tracing (DFS scoring)", expanded=True):
    angleA = st.slider("angle gate (deg)", 90.0, 180.0, 140.0, 1.0)
    overlap = st.slider("overlap (max shared edges)", 0, 20, 4, 1)
    intensity_coeff = st.slider("intensity-coeff", 0, 15, 3, 1)
    thickness_coeff = st.slider("thickness-coeff", 0, 15, 3, 1)
    angle_coeff = st.slider("angle-coeff", 0, 15, 5, 1)
    score_threshold = st.slider("score-threshold", 0, 20, 8, 1)
    apply_angle_penalty = st.checkbox("Apply angle penalty", value=True,
                                      help="Off => sharp corners can survive on intensity/thickness alone.")
    either_constraint_coeff = st.slider(
        "either-constraint-coeff (demo)", 0, 20, 0, 1,
        help=">0 forces an edge if EITHER intensity or thickness is satisfied, regardless of angle.")
    arb_intensity_tol = st.slider("intensity tolerance", 0.0, 1.0, 0.17, 0.01)
    arb_thickness_tol = st.slider("thickness tolerance", 0.0, 1.0, 0.27, 0.01)

run = st.sidebar.button("▶ Run", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Assemble parameter dict (keys mirror graft_core.DEFAULT_PARAMS / build_params)
# ---------------------------------------------------------------------------
params = {
    "preprocess": preprocess, "sigma": sigma, "small": float(small),
    "thresh_top": thresh_top, "eps": int(eps), "size": int(size),
    "node_extension": node_extension, "min_edge_pixels": int(min_edge_pixels),
    "merge_radius": int(merge_radius), "thickness_factor": thickness_factor,
    "intensity_factor": intensity_factor,
    "angleA": angleA, "overlap": int(overlap),
    "intensity_coeff": int(intensity_coeff), "thickness_coeff": int(thickness_coeff),
    "angle_coeff": int(angle_coeff), "score_threshold": int(score_threshold),
    "apply_angle_penalty": apply_angle_penalty,
    "either_constraint_coeff": int(either_constraint_coeff),
    "arb_intensity_tol": arb_intensity_tol, "arb_thickness_tol": arb_thickness_tol,
}


# ---------------------------------------------------------------------------
# Run the pipeline (on button press) and stash results in session_state so they
# persist across the reruns Streamlit triggers on any widget change.
# ---------------------------------------------------------------------------
st.title("GraFT — Interactive Filament Tracer")
st.caption("Set parameters on the left, then click **Run**. "
           "Tracing-only changes re-trace in ~1–2 s (the segmentation/graph build is cached).")

if run:
    try:
        # Resolve the image bytes (for caching) + array.
        if source == "Upload":
            if uploaded is None:
                st.error("Please upload an image, or switch to 'Sample / path'.")
                st.stop()
            raw = uploaded.getvalue()
            img = gc.load_grayscale(io.BytesIO(raw))
            key = _image_key(raw)
        else:
            with open(image_path, "rb") as fh:
                raw = fh.read()
            img = gc.load_grayscale(image_path)
            key = _image_key(raw)

        build_items = tuple(sorted((k, params[k]) for k in gc.BUILD_PARAM_KEYS))

        t0 = time.time()
        with st.spinner("Building segmentation + graph …"):
            built = cached_build(key, build_items, img)
        t_build = time.time() - t0

        t0 = time.time()
        with st.spinner("Tracing …"):
            graph = gc.trace(built, params)
            fig_trace = gc.render_trace(built, graph)
        t_trace = time.time() - t0

        metrics, fig_eval = gc.evaluate(built, graph, gt_folder or None)

        st.session_state["result"] = {
            "fig_trace": fig_trace, "fig_eval": fig_eval, "metrics": metrics,
            "nodes": gc.count_nodes(built), "t_build": t_build, "t_trace": t_trace,
            "image_shape": built.image_shape,
        }
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
    except Exception as exc:  # surface pipeline errors (e.g. no filaments) in the UI
        st.error(f"Run failed: {exc}")


# ---------------------------------------------------------------------------
# Display the most recent result
# ---------------------------------------------------------------------------
res = st.session_state.get("result")
if res is None:
    st.info("No run yet — configure parameters and click **Run**.")
else:
    st.success(
        f"Graph: {res['nodes']} nodes · image {res['image_shape']} · "
        f"build {res['t_build']:.1f}s · trace {res['t_trace']:.1f}s"
    )
    col_img, col_eval = st.columns([3, 2])
    with col_img:
        st.subheader("Traced filaments")
        st.pyplot(res["fig_trace"])
    with col_eval:
        st.subheader("Evaluation")
        m = res["metrics"]
        if m is None:
            st.info("No ground-truth folder provided — evaluation skipped.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("F1 (strict)", f"{m['overall_overall_f1_score']:.3f}")
            c2.metric("IoU (strict)", f"{m['overall_overall_iou']:.3f}")
            c3.metric("MCC", f"{m['MCC_mcc']:.3f}")
            c1.metric("F1 (overlap)", f"{m['overall_overall_f1_score_overlap']:.3f}")
            c2.metric("over-seg ratio", f"{m['MCC_oversegmentation_ratio']:.2f}")
            c3.metric("GT split", f"{int(m['MCC_gt_oversegmented'])}")
            st.caption("Strict = standard per-filament Dice/IoU (primary). "
                       "Overlap = lenient (overlap-only). See README → Evaluation.")
            if res["fig_eval"] is not None:
                st.pyplot(res["fig_eval"])
