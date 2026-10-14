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
   uv run python scripts/extract_frames.py data/videos/f1.mp4 --sport f1 --count 90
   ```

2. Label the frames in YOLO format, loading the class list from `labels.txt` so the class order matches the config. Unzip the labels into `data/labels/<video>/`. Frames without a label file are treated as having no logos. If you labelled frames in several batches, copy the frames and labels into one folder each (here `data/frames/f1_full` and `data/labels/f1_full`); the file names keep the frame number, which the next step uses to split by time.

3. Build the dataset. The video is cut into 2-second blocks and every 5th block goes to validation, so neighbouring near-identical frames stay on the same side while validation still covers the whole video. The script warns about brands that end up only in train or only in val:

   ```bash
   uv run python scripts/prepare_dataset.py data/frames/f1_full data/labels/f1_full --sport f1
   ```

4. Train locally:

   ```bash
   uv run sports-analyzer train data/datasets/f1/data.yaml --name f1 --epochs 50 --imgsz 960
   ```

   Weights are saved to `runs/detect/<name>/weights/best.pt` (`configs/f1.yaml` looks in `runs/detect/f1`). Ultralytics saves a checkpoint after every epoch, so if training stops for any reason, run the same command again and it resumes after the last completed epoch. A finished run is never overwritten; use a new `--name` to train again.

   Or train on a free Colab GPU with [notebooks/finetune_colab.ipynb](notebooks/finetune_colab.ipynb) ([open in Colab](https://colab.research.google.com/github/AndrewKM210/sports-analyser/blob/main/notebooks/finetune_colab.ipynb)), which installs this package from GitHub and saves runs to Google Drive. Zip the dataset and upload it to `MyDrive/sports-analyzer/datasets/`:

   ```bash
   cd data/datasets && zip -r f1.zip f1 -x '*.cache'
   ```

   If Colab disconnects, run all cells again to resume.

5. Evaluate on the validation split (precision, recall and mAP per brand), and compare both detectors on the whole video:

   ```bash
   uv run yolo detect val model=runs/detect/f1/weights/best.pt data=data/datasets/f1/data.yaml \
     imgsz=960 name=f1_val exist_ok=True
   uv run sports-analyzer analyze data/videos/f1.mp4 --sport f1 --detector finetuned
   uv run sports-analyzer analyze data/videos/f1.mp4 --sport f1 --detector zeroshot
   ```

   The video includes the frames the model was trained on, so use it to see the detector working and, later, to measure exposure; take accuracy numbers from the validation split.

## Data

Broadcast footage is copyrighted. Clips are used locally only and are never committed; `data/` and `outputs/` are gitignored. Put your own clips in `data/videos/`.

## License

[AGPL-3.0](LICENSE), the same licence as [Ultralytics](https://github.com/ultralytics/ultralytics), which this project builds on.
