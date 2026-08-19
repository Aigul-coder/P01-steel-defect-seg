# GitHub publish checklist — P01

## Safe to commit

| Path | Why |
|------|-----|
| `src/`, `scripts/`, `configs/`, `tests/` | Code |
| `docker/`, `.github/workflows/` | MLOps |
| `docs/` incl. `docs/assets/demo/*.png` | Docs + small QA overlays (~3 MB) |
| `data/holdout_ids.txt` | ImageId list only (no pixels) |
| `data/README.md` | Download instructions |
| `artifacts/results_severstal.json` | Honest metrics summary |
| `artifacts/metrics.example.json` | Schema example |
| `artifacts/.gitkeep` | Placeholder |
| `README.md`, `pyproject.toml`, `LICENSE` | Repo meta |

## Never commit

| Path | Why |
|------|-----|
| `data/raw/`, `data/subset/`, `data/holdout_preview/` | Severstal images (Kaggle terms) |
| `*.pt`, `*.pth`, `*.onnx` | Weights (~93 MB each) |
| `.venv/`, `.wheels/` | Local env |
| `.env`, `kaggle.json` | Secrets |
| `artifacts/overlays/` | Generated locally (161+ PNG) |
| `artifacts/model*.pt`, `metrics*.json` (except example/results) | Local train outputs |

## Weights for reviewers

Option A — **GitHub Release** (recommended):

1. Create release `v0.1.0-severstal`
2. Attach `model_baseline.pt`, `model.pt` (optional: `model.onnx`)
3. Link from README

Option B — train from scratch (documented in README; ~30 min on RTX 2070).

## First push (standalone repo)

```powershell
cd D:\Aigul\aigul\projects\P01-steel-defect-seg
git init
git add .
git status   # verify no data/raw, no .pt, no .venv
git commit -m "P01: steel defect segmentation demo (baseline vs improved)"
git branch -M main
git remote add origin https://github.com/<user>/P01-steel-defect-seg.git
git push -u origin main
```

## Pre-push verify

```powershell
git status
git ls-files | Select-String -Pattern '\.pt$|data/raw|holdout_preview|\.venv'
# should return nothing
pytest -q
```

## CI note

`.github/workflows/ci.yml` is configured for **standalone repo** (root = project).  
If you keep P01 inside the monorepo factory, change `working-directory` back to `projects/P01-steel-defect-seg`.
