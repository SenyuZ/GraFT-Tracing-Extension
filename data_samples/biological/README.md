# Biological example

Two slices from the bottom region of an *Arabidopsis thaliana* seedling (actin
tagged with the mNeonGreen-FABD2 marker, confocal microscopy), each with a
manually-traced ground truth. Included to show how the pipeline behaves on real
microscopy.

Source of the raw images: Isabella Østerlund's public dataset on Zenodo,
<https://zenodo.org/records/10476058>. The ground truth was traced and labelled
manually for this project.

## Layout

```
biological/
├── slice1/
│   ├── image.png            # input microscopy frame (1024×1024, 8-bit grayscale)
│   ├── raw_layers/          # the 5 hand-traced layers (white bg), utility input
│   │   └── layer1.png … layer5.png
│   └── ground_truth/        # 5 binary masks produced by tools/prepare_ground_truth.py
│       └── label1.png … label5.png
├── slice2/                  # same structure, a second slice (5 filaments)
└── unusable_single_layer/   # an early all-in-one-layer attempt (see its README)
```

The `label*.png` masks are what the workflow consumes via `--gt`; the
`raw_layers/` are kept for provenance (they are the input to the conversion
utility). To regenerate the ground truth from the raw layers:

```bash
python tools/prepare_ground_truth.py layers \
    --input  data_samples/biological/slice1/raw_layers \
    --output data_samples/biological/slice1/ground_truth \
    --threshold 15 --min-pixels 30 --preview
```

## Run the pipeline on a slice

```bash
python GraFT_workflow_still_improved_iterations1.py \
    --image data_samples/biological/slice1/image.png \
    --gt    data_samples/biological/slice1/ground_truth \
    --output results_biological_slice1 \
    --preprocess full
```

> Status: preliminary / qualitative. The preprocessing parameters and the new
> intensity/thickness constraints need to be tuned *jointly* before results on these
> dense real images can be quantified, which was beyond the project's scope. The
> manual labelling alone took about 50 hours; the masks are released as a small
> annotated-actin resource regardless. See the report's discussion and the repo's
> "Limitations & future work" section.
