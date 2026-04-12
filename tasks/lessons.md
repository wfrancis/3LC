# Lessons Learned

## 2026-04-12: Focus on minority classes for max mAP lift
- COCO mAP@0.5 weights all 4 classes equally
- Cars = 82% of labels but only 25% of mAP weight
- Truck = 1.7% of labels but 25% of mAP weight
- **Rule**: Prioritize truck > van > bus > car when fixing labels — each minority fix has 4x+ more impact

## 2026-04-12: Don't wait for full training to start analysis
- Label-ranker can detect duplicate labels, tiny boxes, and other label-only issues with ZERO predictions
- Found 1,285 tiny boxes across 638 images before training even finished
- **Rule**: Always run label-only analysis first while training runs in parallel

## 2026-04-12: First baseline doesn't need 10 epochs
- For label-ranking purposes, even 2-3 epoch weights produce useful predictions
- Don't wait for convergence on the first pass — iterate fast
- **Rule**: Use early weights for first label-ranking pass, full training for submission

## 2026-04-12: Scene-based prioritization is key
- 99 unique scenes in the dataset, minority classes cluster in specific scenes
- MVI_40863 = #1 priority (643 buses + 104 trucks in 111 images)
- Top 10 truck scenes hold 72% of all trucks
- **Rule**: Review minority-class-heavy scenes first for maximum mAP lift per fix

## 2026-04-12: Clean both train AND val labels
- Val set had 817 tiny boxes across 302 images — same issue as train
- Model learns from train but is evaluated on val; dirty val labels hurt measured mAP
- **Rule**: Always clean both splits, not just train

## 2026-04-12: Epoch 2 model too weak for missing-label detection
- Label-fixer agent reviewed 25+ images for missing labels — found ZERO genuine missing vehicles
- All "missing_label" flags were false positives: sub-detections within existing labels, edge-clipped vehicles, background noise
- Wrong-class detection worked better: found 10 real fixes even with weak model
- **Rule**: Need epoch 5+ model to reliably detect missing labels. Use early weights only for wrong-class analysis.

## 2026-04-12: Don't short-change epoch count
- 5 epochs from scratch is not enough — mAP barely above 0.01 at epoch 2
- YOLOv8n from scratch needs 15-20 epochs to converge
- Killed 2 training runs for being too short/dirty — wasted time
- **Rule**: Start with 20 epochs. Can always stop early if converged, but can't get time back from too-short runs.

## 2026-04-12: Use subagents for parallel analysis
- While GPU is busy training, CPU-bound analysis can run in parallel
- Data distribution analysis, label statistics, issue detection — all parallelizable
- **Rule**: Always have subagents doing useful work while waiting on GPU
