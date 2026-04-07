## ADDED Requirements
### Requirement: Provide a paper-faithful SmellNet autoresearch baseline
The system SHALL provide a SmellNet-specific autoresearch baseline that reproduces the six-channel sensor preprocessing used by the paper-matching transformer runs.

#### Scenario: Baseline data preparation
- **GIVEN** a SmellNet training directory and testing directory
- **WHEN** the baseline preparation pipeline runs
- **THEN** it normalizes raw CSV columns into the canonical SmellNet sensor schema
- **AND** it keeps the six model-input channels in this exact order: `NO2`, `C2H5OH`, `VOC`, `CO`, `Alcohol`, `LPG`
- **AND** it subtracts the first row of each recording
- **AND** it applies temporal differencing with `periods=25`
- **AND** it builds sliding windows with `window_size=100` and `stride=50`

#### Scenario: Train-only normalization
- **GIVEN** prepared training and testing windows
- **WHEN** normalization runs
- **THEN** the scaler is fit only on flattened training windows
- **AND** the same scaler is applied to testing windows without refitting

### Requirement: Provide an executable baseline training entrypoint
The system SHALL expose a training entrypoint that can train a baseline SmellNet classifier without relying on hidden in-memory preprocessing state.

#### Scenario: Baseline training run
- **GIVEN** valid SmellNet train and test directories
- **WHEN** the baseline training entrypoint is executed
- **THEN** it trains a classifier on the prepared six-channel windows
- **AND** the default tracked experiment is the benchmark-faithful single-model transformer baseline with contrastive learning disabled
- **AND** it evaluates at least top-1 and top-5 accuracy on the held-out test split
- **AND** it writes reproducible artifacts to a run directory

#### Scenario: Baseline run metadata
- **GIVEN** a completed baseline training run
- **WHEN** the run artifacts are written
- **THEN** the saved metadata identifies the experiment track as `exact-upstream`
- **AND** it records that the baseline model family is `transformer`
- **AND** it records that contrastive learning is disabled for the tracked baseline
- **AND** it records that the primary score is the held-out window-level classification result

### Requirement: Provide a validation-locked budgeted search loop
The system SHALL provide a time-budgeted search loop for the exact-upstream transformer track that uses grouped validation carved only from the official training split.

#### Scenario: Grouped validation search split
- **GIVEN** the official SmellNet training split
- **WHEN** the budgeted search loop prepares data
- **THEN** it reserves whole CSV files for validation rather than splitting windows from the same file across train and validation
- **AND** it fits normalization only on the resulting training subset
- **AND** it leaves the official test split untouched during candidate selection

#### Scenario: Validation-locked model selection
- **GIVEN** a fixed time budget and multiple candidate runs
- **WHEN** the budgeted search loop executes
- **THEN** it ranks candidates using validation metrics only
- **AND** it evaluates the official test split only once, after the best validation candidate has been selected

### Requirement: Support a promoted second-stage search
The system SHALL support a focused second-stage search for the exact-upstream transformer track that reduces validation noise and expands optimization-policy search without widening to new model families.

#### Scenario: Two-stage promotion
- **GIVEN** a total search budget
- **WHEN** the phase-2 loop executes
- **THEN** it runs a cheaper first stage to explore many candidates
- **AND** it promotes the strongest validation candidates into a longer second stage

#### Scenario: Rotated grouped validation
- **GIVEN** the grouped validation protocol
- **WHEN** the phase-2 loop scores promoted candidates
- **THEN** it supports rotating the held-out validation file selection across multiple grouped folds
- **AND** it aggregates validation scores across those folds before choosing the final candidate

#### Scenario: Expanded optimization policy search
- **GIVEN** the exact-upstream transformer family
- **WHEN** the phase-2 loop samples candidates
- **THEN** it may vary learning rate, weight decay, label smoothing, warmup ratio, width, depth, head count, and dropout
- **AND** it preserves the exact-upstream preprocessing contract
