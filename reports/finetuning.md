# Fine-tuning a logo detector on F1 footage

Zero-shot detection (YOLO-World with text prompts such as "Aramco logo") found no real logos in F1 broadcast footage, even large and clear trackside boards. This report covers the next step: fine-tuning a small YOLO model on a handful of labelled frames from the same footage.

## Data

- **Footage:** one minute of an F1 broadcast, 3024×1964 at 60 fps. Used locally only, never committed.
- **Labelling:** 90 frames labelled in makesense.ai in two batches: 30 from the first 10 s (3 frames/s) and 60 from the rest (1.2 frames/s). 39 frames contain logos (131 boxes); the other 51 are kept as background.
- **Brands:** Aramco, Shell, HP, Salesforce, Mastercard, Intel, Microsoft.
- **Split:** the video is cut into 2-second blocks and every 5th block goes to validation, so near-identical neighbouring frames stay on the same side.

| | Frames | Boxes per brand |
|---|---|---|
| Train | 69 | Aramco 54, Microsoft 21, Salesforce 15, Shell 3, HP 2, Intel 1 |
| Validation | 21 | Aramco 17, Salesforce 5, Intel 5, Mastercard 5, Microsoft 3 |

Shell and HP only appear in training, so they aren't validated. Mastercard only appears in validation, so the model can't learn it.

## Training

`yolo11s` (pretrained on COCO), 1280 px input, 150 epochs, on a free Colab T4: about 12 minutes. The best epoch (74) is kept. Training saves a checkpoint every epoch and resumes after a disconnect (`sports-analyzer train`).

## Results

On the validation split (21 frames, 35 boxes):

| Detector | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Zero-shot YOLO-World s, 1280 px | 0 | 0 | 0 | 0 |
| Fine-tuned YOLO11s, 1280 px | 0.59 | 0.60 | 0.61 | 0.38 |

Per brand (fine-tuned):

| Brand | Val boxes | Precision | Recall | mAP50 |
|---|---|---|---|---|
| Aramco | 17 | 0.49 | 0.88 | 0.66 |
| Salesforce | 5 | 0.59 | 0.31 | 0.42 |
| Intel | 5 | 1.00 | 0.79 | 0.96 |
| Microsoft | 3 | 0.87 | 1.00 | 0.995 |
| Mastercard | 5 | 0 | 0 | 0 |

The confidence threshold matters. At 0.25 (the default when training), Salesforce's correct detections are all filtered out:

| Threshold | Precision | Recall |
|---|---|---|
| 0.10 | 0.57 | 0.62 |
| 0.15 | 0.60 | 0.54 |
| 0.25 | 0.52 | 0.42 |

0.10 gives the best balance for this model.

On the full minute (every 2nd frame, threshold 0.25), the model drew Aramco 2450 boxes, Salesforce 725, Microsoft 601, HP 74, Intel 35, Shell 27 and Mastercard 0. These are box counts, not seconds on screen; turning them into exposure time is the next phase. The annotated video looks good overall, with occasional false positives.

Inference takes 60–80 ms per frame at 1280 px on an M1-series Mac (MPS).

## What the predictions show

- Large, close trackside boards (Aramco, Salesforce) and the Microsoft logo on the cockpit are found reliably.
- Some "false positives" are real Aramco boards that weren't labelled, so Aramco's precision is underestimated.
- A Heineken board was labelled Aramco. Heineken isn't in the brand list, so the model has never seen it as a separate class.
- Small, distant logos and the logos on car liveries are the main misses.

## Caveats

- The validation set is tiny (35 boxes; Microsoft has 3), so the numbers are indicative, not precise.
- Intel's 0.96 is inflated: its validation frames (4.0–5.6 s) and its one training frame (6.0 s) come from the same camera shot, a third of a second apart.
- Everything comes from one race and one circuit. Trackside boards change by circuit and liveries change every season.

## Next steps

- Label more data: a whole race sampled at a steady rate, in CVAT with box interpolation, so every brand appears in both train and validation.
- Add the other sponsors visible on track (Heineken, Pirelli, Louis Vuitton) so they stop being confused with listed brands.
- Lower the default threshold for the fine-tuned model to around 0.10.
- Tiling (SAHI) or higher resolution for small, distant logos.
- Tracking (next phase) to smooth out single-frame misses and false positives.
