# Sports Analyzer: Sponsor Exposure Tracker

> **Status: work in progress.** See [ROADMAP.md](ROADMAP.md) for the plan and progress.

## What it produces

Given a video clip:

- an annotated video with bounding boxes, brand labels and track IDs
- a per-brand exposure report (`report.csv`, `report.json`) and the raw detections (`detections.csv`)
- a timeline chart showing when each brand was visible
- a Gradio web demo for uploading a clip and viewing the results

## Quick start

Requires Python 3.11+.

```bash
# with uv (recommended)
uv sync
uv run sports-analyzer --help

# or with venv + pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
sports-analyzer --help
```

## Usage

### Analyse a clip

```bash
uv run sports-analyzer analyze data/videos/clip.mp4 --sport f1                        # zero-shot (YOLO-World)
uv run sports-analyzer analyze data/videos/clip.mp4 --sport f1 --detector finetuned   # fine-tuned YOLO
```

The annotated video is saved to `outputs/<clip>_<detector>.mp4`, and throughput and detections per brand are printed at the end. Useful options:

- `--every N` runs detection on every Nth frame
- `--conf 0.1` sets the confidence threshold
- `--weights path/to/best.pt` uses a different model
- `--device cpu` forces the CPU (MPS is used when available)

### Sports configs

Each sport has its own config in `configs/<sport>.yaml` with its brands, the text prompts for zero-shot detection and the settings for each detector. To add a sport, add a new file; it becomes available as `--sport <name>`.

### Fine-tuning

1. Extract frames to label. This saves them to `data/frames/<video>/`, along with a `labels.txt` listing the sport's brands:

   ```bash
   uv run python scripts/extract_frames.py data/videos/f1.mp4 --sport f1 --start 10 --count 60
   ```

2. Label the frames in YOLO format, loading the class list from `labels.txt` so the class order matches the config. Unzip the labels into `data/labels/<video>/`. Frames without a label file are treated as having no logos.

3. Hold out a test set. Move a later stretch of frames (and their labels) into separate `_test` folders, so the model is never trained on them. Neighbouring frames are near-duplicates, so split by time rather than at random.

4. Build the training dataset (the last 20% of the training frames, by time, are used for validation) and the test dataset:

   ```bash
   uv run python scripts/prepare_dataset.py data/frames/f1_train data/labels/f1_train --sport f1
   uv run python scripts/prepare_dataset.py data/frames/f1_test data/labels/f1_test --sport f1 \
     --out data/datasets/f1_test --val-fraction 1.0
   ```

5. Train:

   ```bash
   uv run yolo detect train data=data/datasets/f1/data.yaml model=yolo11n.pt \
     epochs=50 imgsz=960 batch=8 device=mps cache=ram name=f1 exist_ok=True
   ```

   The weights are saved to `runs/detect/f1/weights/best.pt`, which is where `configs/f1.yaml` looks for them.

6. Evaluate on the test set (precision, recall and mAP per brand), and compare both detectors on the test clip:

   ```bash
   uv run yolo detect val model=runs/detect/f1/weights/best.pt data=data/datasets/f1_test/data.yaml \
     imgsz=960 device=mps name=f1_test exist_ok=True
   uv run sports-analyzer analyze data/videos/f1_test.mp4 --sport f1 --detector finetuned
   uv run sports-analyzer analyze data/videos/f1_test.mp4 --sport f1 --detector zeroshot
   ```

## Data

Broadcast footage is copyrighted. Clips are used locally only and are never committed; `data/` and `outputs/` are gitignored. Put your own clips in `data/videos/`.

## License

[AGPL-3.0](LICENSE), the same licence as [Ultralytics](https://github.com/ultralytics/ultralytics), which this project builds on.
