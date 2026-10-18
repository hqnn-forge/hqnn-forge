# Experiment records

::: hqnn_forge.experiment

Format-1 records add PCA, channel, retuned-threshold, chunking, selected-fold
and validation-extension settings. Each fold stores the full per-level noise
scores, distinct training/evaluation noise and devices, preprocessing parameters
and attempt diagnostics. Replay supplies backward-compatible defaults when an
older record lacks these fields. Observers and private checkpoint paths are
not serialized or required for numerical replay.
