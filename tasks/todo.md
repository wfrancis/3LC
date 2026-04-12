# Competition TODO

## Setup
- [ ] Download starter kit from Kaggle Data tab
- [ ] Set up Python environment (see CLAUDE.md)
- [ ] `3lc login` + `3lc service`
- [ ] `python verify_setup.py`

## First Iteration
- [ ] `python register_tables.py`
- [ ] `python train.py` (baseline)
- [ ] `python scripts/predict_on_train.py` (predictions on train set)
- [ ] Run label-ranker, review top issues
- [ ] Fix labels (agent or manual)
- [ ] Retrain and evaluate

## Tooling
- [x] Rust label-ranker built and tested
- [x] Rust map-eval built and tested
- [x] predict_on_train.py script
- [x] Claude label-fixer agent
- [x] /fix-labels skill
- [x] iterate.sh one-command script
