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

## Data

Broadcast footage is copyrighted. Clips are used locally only and are never committed; `data/` and `outputs/` are gitignored. Put your own clips in `data/videos/`.

## License

[AGPL-3.0](LICENSE), the same licence as [Ultralytics](https://github.com/ultralytics/ultralytics), which this project builds on.
