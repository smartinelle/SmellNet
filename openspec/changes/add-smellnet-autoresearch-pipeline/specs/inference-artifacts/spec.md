## ADDED Requirements
### Requirement: Save inference-ready preprocessing artifacts
The system SHALL persist all preprocessing state required to reproduce offline normalization and Raspberry Pi inference for a trained model.

#### Scenario: Export preprocessing contract
- **GIVEN** a completed baseline training run
- **WHEN** artifacts are written
- **THEN** the run directory contains machine-readable preprocessing metadata
- **AND** that metadata includes channel ordering, dropped columns, differencing period, windowing parameters, label ordering, and scaler statistics

### Requirement: Save a reloadable model checkpoint
The system SHALL persist a checkpoint that can be reloaded for offline evaluation or device-side inference conversion.

#### Scenario: Export checkpoint and configuration
- **GIVEN** a completed baseline training run
- **WHEN** artifacts are written
- **THEN** the run directory contains a checkpoint file
- **AND** the run directory contains the model architecture parameters used to create the network
