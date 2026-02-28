# Layer 4: Model

Sub-proposal of the [documentation index](README.md). Overview: [15 Layers overview](15_layers_overview.md). Prev: [18 Layer Fixture](18_layer_fixture.md). Next: [20 Layer Recommendation](20_layer_recommendation.md).

---

## What it does

Forward pass (Inference), loss (Training), and training loop (AxonTrain). Params come from CheckpointLoader. Same forward/loss used for training and inference.

## Public surface

RecGPT.Inference.forward/4, RecGPT.Inference.forward_full_sequence/4, RecGPT.Training.build_train_batch/4, RecGPT.Training.loss_shifted_ce/2, RecGPT.AxonTrain.stream_batches/4, RecGPT.AxonTrain.run/3.

## How to test

inference_test.exs, training_test.exs, axon_train_test.exs. Stub checkpoint params for Inference.

---

## See also

- [15 Layers overview](15_layers_overview.md) - Diagram and table.
- [16 Layer Artifacts](16_layer_artifacts.md) - CheckpointLoader.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.