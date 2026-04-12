# Open-Source Split Plan

Date: 2026-04-10

This note captures the safest path for publishing the new autoresearch work as open source without over-claiming rights over the inherited `SmellNet` codebase.

## 1. Recommendation

Do **not** treat this fork itself as the clean public open-source deliverable.

Instead:

1. Create a **new repo** for the original work only.
2. License that new repo under a license you control, such as `MIT` or `Apache-2.0`.
3. Move only the files that are clearly your own.
4. Keep this fork as the integration/reproduction repo.

Reason:

- This repo does not contain a `LICENSE` file.
- The upstream `MIT-MI/SmellNet` repo also appears to have no explicit license file in the root snapshot inspected during this work.
- Without an explicit license, the code is publicly visible but not cleanly relicensable.

## 2. Suggested new repo names

Any of these would be reasonable:

- `smellnet-autoresearch`
- `smellnet-repro-tools`
- `smellnet-edge-research`
- `smellnet-benchmark-tools`

Best default: `smellnet-autoresearch`

That name matches the main original contribution and avoids implying ownership over the full upstream project.

## 3. Files to move into the new repo

These are the files that should move first.

### 3.1 Core autoresearch files

- `autoresearch_smellnet/README.md`
- `autoresearch_smellnet/__init__.py`
- `autoresearch_smellnet/prepare.py`
- `autoresearch_smellnet/train.py`
- `autoresearch_smellnet/search.py`
- `autoresearch_smellnet/search_phase2.py`
- `autoresearch_smellnet/search_contrastive.py`
- `autoresearch_smellnet/program.md`

### 3.2 Design / process docs

- `openspec/project.md`
- `openspec/config.yaml`
- `openspec/changes/add-smellnet-autoresearch-pipeline/proposal.md`
- `openspec/changes/add-smellnet-autoresearch-pipeline/design.md`
- `openspec/changes/add-smellnet-autoresearch-pipeline/tasks.md`
- `openspec/changes/add-smellnet-autoresearch-pipeline/specs/autoresearch-harness/spec.md`
- `openspec/changes/add-smellnet-autoresearch-pipeline/specs/inference-artifacts/spec.md`

### 3.3 Findings docs

- `analysis/autoresearch_findings_2026-04-07.md`
- this file: `analysis/open_source_split_plan_2026-04-10.md`

## 4. Current upstream dependencies inside the autoresearch code

Right now the new autoresearch files still import a few things from the inherited SmellNet tree.

### 4.1 Imports that still point into `models/`

From `autoresearch_smellnet/train.py`:

- `models.models.GCMSMLPEncoder`
- `models.models.Transformer`

From `autoresearch_smellnet/prepare.py`:

- `models.dataset.PairedDataset`
- `models.dataset.UniqueGCMSampler`

From `autoresearch_smellnet/search_contrastive.py`:

- `models.evaluate.evaluate_contrastive`
- `models.loss.cross_modal_contrastive_loss`
- `models.utils.ingredient_to_category`

### 4.2 What to do about those dependencies

For the new public repo, create a small self-contained package and copy only the minimal pieces needed for the autoresearch harness to run:

- transformer model
- GC-MS encoder
- paired dataset helpers
- contrastive loss
- contrastive evaluation helper
- ingredient-to-category mapping

The goal is to avoid depending on the whole inherited `models/` subtree.

## 5. Proposed package layout for the new repo

Suggested shape:

```text
smellnet_autoresearch/
  pyproject.toml
  README.md
  LICENSE
  smellnet_autoresearch/
    __init__.py
    prepare.py
    train.py
    search.py
    search_phase2.py
    search_contrastive.py
    model_zoo.py
    datasets.py
    losses.py
    evaluation.py
    taxonomy.py
  docs/
    findings.md
    artifact_contract.md
    publishing_notes.md
  openspec/
    ...
```

Where:

- `model_zoo.py` contains the minimal transformer + GC-MS encoder code
- `datasets.py` contains `PairedDataset` and `UniqueGCMSampler`
- `losses.py` contains the contrastive loss
- `evaluation.py` contains the contrastive retrieval evaluation
- `taxonomy.py` contains the category mapping

## 6. What should stay out of the new repo

Do **not** move these by default:

- `data/`
- `offline_training/` / `offline_testing/` snapshots
- `chi_paper_data/`
- `saved_models/`
- upstream notebooks and analysis scripts unrelated to the autoresearch work
- large generated `autoresearch_runs/` directories
- the current `smell-pi` artifact bundle unless you explicitly decide to publish trained weights

## 7. Checkpoints and artifacts

Be careful with pretrained artifacts.

Recommended default:

- publish the **artifact format**
- publish the **training code**
- publish the **preprocessing contract**
- do **not** publish large trained checkpoints until redistribution rights are clear

If you later decide to publish checkpoints, do it intentionally and document:

- exact data source
- exact preprocessing
- exact metric
- whether the weights were trained from public data only
- expected license / redistribution basis

## 8. Minimal publication sequence

This is the safest concrete order.

1. Create a new repo, for example `smartinelle/smellnet-autoresearch`.
2. Add:
   - `README.md`
   - `LICENSE`
   - `pyproject.toml`
   - package skeleton
3. Copy the autoresearch code and docs listed above.
4. Replace imports from `models.*` with local package modules.
5. Add one small smoke test.
6. Make the package operate on a user-supplied SmellNet checkout or data root.
7. Publish only after the new repo runs independently from this fork.

## 9. Short README positioning for the new repo

Suggested wording:

> This repo contains autoresearch and evaluation tooling built around the public SmellNet benchmark setup. It is not the original SmellNet codebase and does not redistribute the upstream dataset or claim ownership over inherited upstream code.

That framing is accurate and avoids over-claiming.

## 10. Practical next step

When ready, the next implementation step is:

- scaffold the new standalone repo
- copy the autoresearch files
- vendor the minimal dependencies listed in section 4
- make the harness run against a user-provided SmellNet data root

That is the cleanest path to a genuinely publishable open-source project.
