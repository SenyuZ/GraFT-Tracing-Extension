# Single-layer colour attempt (not usable for evaluation)

`single_layer_colored_attempt.png` is an early ground-truth attempt: every
filament was traced on one layer, each in a different colour, rather than one
filament per layer.

This turned out not to be usable for quantitative evaluation. Where filaments
overlap, their colours cannot be separated reliably (at a crossing it is
impossible to tell which pixels belong to which filament), so the per-filament
masks the evaluation needs cannot be recovered cleanly. (The
`tools/prepare_ground_truth.py composite` mode attempts this kind of colour
separation, but it is explicitly best-effort for this reason.)

The lesson, and the approach used for the usable examples in `../slice1/` and
`../slice2/`, is to trace one filament per layer so that overlaps are never
ambiguous. Several more such single-layer composites were produced during the
project; only this one is kept here, purely to document the pitfall.
