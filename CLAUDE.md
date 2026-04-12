# 3LC Multi-Vehicle Detection Challenge

## Competition Summary

**Goal**: Detect 4 vehicle classes (truck=0, car=1, van=2, bus=3) in traffic-camera images.
**Metric**: mAP@0.5 via pycocotools (COCO-style, bbox task, IoU threshold 0.5).
**Key constraint**: YOLOv8n only, trained from scratch, 640px input, no ensembles/TTA/pseudo-labels.
**This is a data-centric competition** — the model is fixed. The winner is whoever curates the best training data.

- Kaggle: https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge
- Deadline: June 9, 2026
- 3 submissions/day, select up to 2 for final judging
- Public leaderboard: ~40% of test set. Private: remaining ~60% (final ranking).

## Our Advantages

### 1. Rust Tooling — Faster Iteration
Everyone else manually clicks through the 3LC Dashboard to find label errors. We build fast CLI tools in Rust that do the analysis in milliseconds:

- **`label-ranker`**: Compare model predictions against YOLO labels on the training set. Rank images by "label error likelihood" using signals like:
  - High-confidence prediction with no matching label (IoU < 0.3) = missing annotation
  - Matched prediction but IoU 0.3–0.5 = inaccurate box
  - Two same-class labels with IoU > 0.7 = duplicate box
  - Box area < 0.1% of image = garbage tiny box
  - Prediction class != label class at same location = wrong class
- **`map-eval`**: Local mAP@0.5 evaluator matching pycocotools exactly — unlimited local evaluation instead of burning 3 Kaggle submissions/day.

### 2. Claude Code Agent — Automated Label Fixing
We have access to Claude Code source and can build custom agents/skills:

- **Custom agent** (`.claude/agents/label-fixer.md`): Uses vision to review flagged images and decide corrections.
- **Custom skill** (`/fix-labels`): One-command full iteration cycle.
- **Hooks**: `PostToolUse` triggers to chain training → analysis → fixing automatically.

The pipeline no other competitor will have:
```
1. Rust label-ranker flags top-N suspicious images
2. Claude agent reads each image (vision) + current labels + model predictions
3. Claude decides: missing box? wrong class? bad coordinates? fine?
4. Claude rewrites the YOLO .txt label file
5. Retrain → local eval → submit best
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Claude Code Agent (orchestrator)                 │
│                                                   │
│  /fix-labels skill                                │
│    1. Bash: python train.py                       │
│    2. Bash: python predict_on_train.py            │
│    3. Bash: ./label-ranker predictions/ labels/   │
│    4. For top-N flagged images:                   │
│       - Read image (vision)                       │
│       - Read current .txt label                   │
│       - Decide correction                         │
│       - Write corrected .txt                      │
│    5. Bash: python train.py (retrain)             │
│    6. Bash: python predict.py (test set)          │
│    7. Bash: ./map-eval (local score)              │
│    8. Report results                              │
└──────────────────────────────────────────────────┘
```

## Build Priority

1. **Rust `label-ranker`** — prediction-vs-label comparator (highest ROI)
2. **Rust `map-eval`** — local mAP@0.5 matching pycocotools
3. **Python `predict_on_train.py`** — inference on training set for the ranker
4. **Claude agent** `.claude/agents/label-fixer.md` — vision-based label review
5. **Skill** `/fix-labels` — one-command full iteration

## Competition Rules (Key Constraints)

- YOLOv8n architecture only, trained from scratch (no pretrained weights)
- 640px input resolution
- No ensembles, TTA, or pseudo-labels
- Labels are intentionally imperfect — finding and fixing them IS the competition
- Submission: CSV with `id`, `image_id`, `prediction_string` columns
- `prediction_string`: space-separated `class_id confidence x_center y_center width height` (YOLO-normalized)
- No detections: literal `no box`

## Submission Format

```csv
id,image_id,prediction_string
0,MVI_20011__img00001,1 0.92 0.55 0.40 0.12 0.18 3 0.71 0.20 0.30 0.08 0.10
1,MVI_20011__img00002,no box
```

---

## Core Philosophy

**Ultra fast ruthless iteration.** The faster we can iterate, the better we will do. Every tool, every script, every decision should optimize for cycle time. One more iteration beats one more hour of analysis. Ship fast, measure, fix, repeat.

---

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately -- don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes -- don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -- then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections
