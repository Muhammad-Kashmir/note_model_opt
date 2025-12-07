# A collection of notes on model deployment

This repo is **not** a polished library or a finished survey.  
It’s simply a place where I collect working notes, small experiments, and mental maps around:

- [MLSoC.md](./notes/MLSoC.md) - tour of the non-CUDA SoC landscape: Vulkan/Kompute, OpenCL, MLX, ARM Compute Library/ARM NN, oneDNN, TVM, IREE, ncnn, ExecuTorch/LiteRT, plus vendor stacks (Qualcomm QNN, TI TIDL, NXP eIQ, NNAPI).
- [OptimizingModels.md](./notes/OptimizingModels.md) - model and edge-level optimization mainstream pathways: quantization (PTQ/QAT, FP8/INT4), pruning/sparsity, distillation, low-rank adapters, and deployment toolchains (TensorRT ModelOpt, Intel INC/OpenVINO, TorchAO, ONNX Runtime, TFLite, TVM, ncnn).

If you’re okay with half-baked ideas, TODOs, and rough edges, you might find something useful here.
If you see something obviously wrong or missing, PRs are all very welcome.
