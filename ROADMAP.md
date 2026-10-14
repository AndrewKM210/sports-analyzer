# Roadmap

## Phase 0: Setup

- [x] Set up project with uv
- [x] .gitignore
- [x] README stub
- [x] Test YOLO prediction on Mac

## Phase 1: Zero-shot detection

- [x] YOLO-World detector with brand prompts from a config file
- [x] Run on a short clip with frame sampling
- [x] Save annotated video
- [x] Measure throughput and note quality
- [x] Per-sport configs

## Phase 2: Fine-tuned YOLO

- [ ] Pick 3-5 brands and get a dataset
- [ ] Label a larger dataset in CVAT
- [x] Colab training notebook
- [x] Fine-tuned detector option
- [x] Compare with zero-shot

## Phase 3: Tracking and metrics

- [ ] Track IDs with ByteTrack
- [ ] Merge tracks into appearances
- [ ] Exposure metrics
- [ ] Tests for metrics
- [ ] Export report (CSV, JSON) and raw detections
- [ ] Timeline chart

## Phase 4: Demo and README

- [ ] Gradio app
- [ ] README write-up
- [ ] Demo GIF
