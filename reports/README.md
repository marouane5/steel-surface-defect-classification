# Experiment reports

Snapshot of `neu_cls_seed42_v1` (seed 42). Checkpoint selection used validation macro-F1 and loss. Test predictions belong only to the selected fine-tuned ResNet18.

- `validation_comparison.csv`: best validation metrics and original training times for all three phases.
- `*_history.csv` and `*_learning_curves.png`: epoch-level training/validation metrics.
- `inventory.csv`, `exact_duplicate_images.csv`, `splits.csv`, `split_metadata.json`: image inventory and split provenance.
- `test_metrics.json`, `test_classification_report.csv`, `test_confusion_matrix.csv`, `test_predictions.csv`: held-out results.
- `test_confusion_matrices.png`, `test_example_predictions.png`, `gradcam_examples.png`: evaluation visuals.
- `environment.json`: hardware, versions, settings, and total original training duration.
- `execution_summary.json`: final successful notebook verification, which reused trained checkpoints.

Raw images and trained weights are excluded from this folder. Checkpoints are available in the project release.
