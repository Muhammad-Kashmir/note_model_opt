# ML Training & Inference on SoCs: the Other-than-CUDA Landscape

**An T. Le, Hanoi, Dec 2025.**

On NVIDIA, life is simple: you live in **CUDA-land**, and everything important eventually compiles down to it.

On ARM-based SoCs, at the time of writing, there is no single “CUDA for everything”-but there *is* an increasingly coherent stack of tools that, together, fill the same role:

* low-level GPU/NPU APIs
* ARM-optimized kernel libraries
* ML compilers and runtimes
* vendor-specific SoC SDKs

This note is a quick tour of that landscape, with links to codebases and docs for easy access. Note that this blog is nowhere near exhaustive as I highly likely miss some emergent pathways.

---

## 1. Low-level compute APIs: the “CUDA-ish” layer

### Vulkan compute & Kompute

On many ARM SoCs (Adreno, Mali, etc.), the modern low-level compute API is **Vulkan** with compute shaders (SPIR-V). Instead of writing raw Vulkan boilerplate, you can use:

* **[Kompute](https://github.com/KomputeProject/kompute)** - a general-purpose GPU compute framework on top of Vulkan, targeting “1000s of cross vendor graphics cards (AMD, Qualcomm, NVIDIA & friends)” and explicitly supporting mobile.

Kompute gives you tensor-like abstractions, dispatch helpers, and cross-platform support, which makes it feel much closer to CUDA while still working across desktop + mobile GPUs.

If you want to write custom kernels, physics/vision ops, or exotic inference backends on Adreno/Mali, **Vulkan + Kompute** is the closest thing to "CUDA on ARM" you’ll get today.

---

### OpenCL (legacy but still around)

On some devices, especially older Mali GPUs and certain embedded boards, **OpenCL** is still supported and is used under the hood by things like the **[ARM Compute Library’s](https://software-dl.ti.com/processor-sdk-linux/esd/AM62PX/11_01_16_13/exports/docs/linux/Foundational_Components/Machine_Learning/arm_compute_library.html)** GPU path.

For *new* designs, Vulkan is the better long-term bet; OpenCL is mainly relevant if you’re stuck with legacy drivers.

---

### Metal + MPS (Apple SoCs)

Not generic ARM, but worth mentioning: on M-series and A-series SoCs, **Metal** plus **Metal Performance Shaders (MPS)** is essentially Apple’s CUDA equivalent, and it’s now wrapped very nicely by:

* **[MLX](https://github.com/ml-explore/mlx)** - Apple’s NumPy-like array and neural-net framework optimized for Apple silicon.

If you treat an M2/M4/M-whatever laptop as a “fat SoC”, MLX is currently the state-of-the-art training+inference stack.

---

## 2. ARM-optimized kernel libraries: the cuBLAS/cuDNN analogues

### ARM Compute Library (ACL) & ARM NN

The **[ARM Compute Library](https://www.arm.com/products/development-tools/embedded-and-software/compute-library)** is a collection of hand-tuned primitives (conv, GEMM, activations, etc.) for **ARM Cortex-A/Neoverse CPUs and Mali GPUs**, using NEON/SVE and OpenCL.

On top of it sits **[ARM NN](https://github.com/ARM-software/armnn)**, a higher-level inference engine that can take TF, [TFLite](https://arm-software.github.io/armnn/23.11/delegate.html), [ONNX](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html) and map them onto CPU/GPU/NPUs (Ethos-U/N) using those kernels and NNAPI where appropriate.

You’ll bump into ACL / ARM NN indirectly in:

* NXP’s **eIQ** stack for i.MX
* Some Android NNAPI drivers
* Certain ARM-optimized builds of PyTorch / TF

---

### oneDNN on AArch64

Intel’s **[oneDNN](https://github.com/uxlfoundation/oneDNN)** (part of oneAPI) is not just for x86: it explicitly supports **ARM 64-bit (AArch64)** and is used as a high-performance CPU backend in many frameworks.

On ARM servers or SoCs where CPU is your main workhorse (e.g., LLM inference on Neoverse or Graviton), [oneDNN](https://uxlfoundation.github.io/oneDNN/) is the “good math library” underneath a lot of stacks (PyTorch, ONNX Runtime, vLLM CPU backends, etc.).

---

## 3. ML compilers & runtimes: the “XLA/TensorRT” of SoCs

### Apache TVM

**[Apache TVM](https://tvm.apache.org/)** is an open ML compiler that takes pre-trained models (PyTorch, TF, ONNX, etc.) and generates optimized code for CPUs, GPUs and specialized accelerators.

Why it matters for SoCs:

* Targets **ARM CPUs**, **Mali GPUs**, and various NPUs via vendor integrations.
* Can be the **single compiler** feeding multiple SoCs (RB8, AM68A, i.MX93…) with different backends.
* Powers or integrates with vendor stacks (e.g., TI’s Edge AI, parts of NXP eIQ).

TVM is one of the few serious options if you want a *unified* compiler that still lets you squeeze each SoC properly.

---

### IREE (MLIR-based compiler + runtime)

**[IREE](https://iree.dev/)** is an MLIR-based end-to-end compiler and runtime that lowers models to a unified IR and then to hardware-specific backends like CPU, **Vulkan (Mali/Adreno)**, Metal, and CUDA.

For SoCs, the interesting path is:

> Framework (PyTorch/TF/JAX) -> MLIR -> IREE -> **Vulkan SPIR-V on mobile GPU** or **ARM CPU**

This lets you share most of your toolchain between datacenter and edge, with only the final backend differing.

---

### ncnn (Vulkan-first mobile runtime)

**[ncnn](https://github.com/Tencent/ncnn)** is a lightweight C++ NN inference framework designed from day one for mobile and embedded. It has:

* CPU kernels optimized for mobile
* A strong **Vulkan backend** for GPU acceleration on Android/ARM.

If you want a **minimal dependency**, high-performance inference engine for Android phones, SBCs, or simple robots, ncnn is a very practical option.

---

### ExecuTorch & LiteRT (ex-TensorFlow Lite)

For graph-level runtimes on-device:

* **[ExecuTorch](https://pytorch.org/projects/executorch/)** is [PyTorch’s unified on-device runtime](https://docs.pytorch.org/executorch/index.html), with backends for CPUs, GPU delegates and SoC-specific accelerators like Qualcomm QNN.
* **[LiteRT](https://ai.google.dev/edge/litert)** (formerly TensorFlow Lite) remains the [standard TFLite runtime](https://ai.google.dev/edge/litert) for Android/embedded, with delegates for NNAPI, GPU, and vendor NPUs.

These don’t replace low-level APIs; they sit *above* them and route ops to NNAPI, QNN, GPU delegates, etc.

---

### LLM-focused: vLLM on ARM

If your main concern is LLM inference:

* **[vLLM](https://github.com/vllm-project/vllm)** is a high-throughput library for LLM serving that now supports non-x86 architectures, including ARM, when paired with appropriate BLAS / oneDNN backends.

On ARM servers or beefy SoCs where CPU is dominant, vLLM + oneDNN/ACL is a realistic way to run sizeable models.

---

## 4. Vendor SoC stacks: “CUDA for X”

This is where things get very SoC-specific. Each vendor now ships something that *behaves* like “CUDA + cuDNN” for their particular mix of CPU/GPU/NPU.

### Qualcomm Snapdragon / QRB / RB8

For RB8-class robotics boards and Snapdragon phones, the key piece is:

* **[Qualcomm AI Engine Direct / QNN SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)** - a set of low-level APIs and tools to target CPU, Adreno GPU, and Hexagon Tensor Processor from one abstraction.

It plugs into:

* **ExecuTorch Qualcomm backend** (tutorial [here](https://docs.pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html))
* **ONNX Runtime QNN Execution Provider** (see [here](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) and [Qualcomm docs](https://docs.qualcomm.com/nav/home/index_QNN.html?product=1601111740009302]))

In practice: **you train on GPUs elsewhere, convert to ONNX or ExecuTorch, then deploy via QNN** for extremely efficient mixed-precision inference on RB8.

---

### Texas Instruments Jacinto / AM68A / AM69A

On TI’s AM6xA boards (Jacinto for edge AI / robotics), the relevant stack is:

* **[Edge AI SDK](https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-am68a/latest/exports/docs/linux/index_Edge_AI.html)** - system-level SDK with GStreamer, vision components, and AI tooling.
* **[TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools)** (TI Deep Learning) - tools and runtimes to compile TFLite / ONNX / TVM models for the C7xMMA AI accelerator + ARM cores.

This gives you a pretty complete **capture -> preprocess -> infer -> postprocess** pipeline for vision and perception on AM68A/AM69A with good power/perf trade-offs.

---

### NXP i.MX (incl. i.MX93)

NXP’s answer is **[eIQ](https://www.nxp.com/design/design-center/software/eiq-ai-development-environment%3AEIQ)** - a full AI software environment for i.MX. (see [docs](https://www.nxp.com/docs/en/user-guide/UG10166.pdf))

According to the recent **i.MX Machine Learning User’s Guide**, eIQ supports multiple inference engines (TFLite, ONNX Runtime, PyTorch, OpenCV) and maps them onto Cortex-A, GPUs, and NPUs across i.MX 8/9 families, including **i.MX93**.

Under the hood it blends:

* ARM NN / ACL
* CMSIS-NN for MCUs
* Vendor NPU runtimes and compilers

So on i.MX, **eIQ is your entry point**; TVM can also be integrated via Yocto/meta-ml when you need more control.

---

### Android NNAPI (cross-vendor abstraction)

Adding for completeness, across many Android SoCs you also have **[NNAPI](https://developer.android.com/ndk/guides/neuralnetworks)** - a C API that lets higher-level frameworks target hardware accelerators without knowing the vendor details.

**ExecuTorch**, **LiteRT/TFLite**, and **ONNX Runtime** can all delegate to NNAPI when a suitable driver exists. For pure portability across mixed devices, NNAPI is still very valuable (even though it’s being phased out in Android 15 in favor of newer APIs).

---

### Apple silicon: MLX + Metal + ANE

We already touched on this, but it’s worth grouping as a “SoC stack”:

* **[MLX](https://github.com/ml-explore/mlx)** gives you NumPy-like arrays, NN layers, optimizers, and JAX-style transforms, all tuned for **Apple’s unified memory + GPU + ANE**.

For laptop-class SoC development, MLX is arguably the most coherent “CUDA-like” experience outside NVIDIA right now: you can do serious training and inference on-device.

---

## 5. Training vs. inference on SoCs: what’s realistic?

### Inference: where SoCs shine today

If you’re deploying **vision, audio, or moderate-size language models** onto SoCs, the current “sane” stack looks roughly like:

* **RB8 / Snapdragon / QRB**
  -> Train on GPU -> export -> deploy via **QNN SDK** (often through ExecuTorch or ONNX Runtime).

* **TI AM68A / AM69A**
  -> Train -> export ONNX/TFLite -> compile via **TIDL / Edge AI SDK** -> integrate in GStreamer/OpenVX pipelines.

* **NXP i.MX (e.g., i.MX93)**
  -> Use **eIQ** with TFLite/ONNX/PyTorch backends; optionally TVM for extra optimization.

* **Generic Android + Mali/Adreno**
  -> **LiteRT/TFLite or ExecuTorch** with delegates (NNAPI, GPU) for mainstream flows.
  -> **ncnn** or **TVM/IREE+Vulkan** when you want a custom C++ runtime.

* **Apple silicon**
  -> **MLX** or Core ML for both training and deployment.

---

### Training: still mostly off-device (with some exceptions)

Full-scale ML training remains firmly in GPU/datacenter land. For SoCs, the realistic training-ish activities are:

* **Lightweight fine-tuning / adapters / LoRA** on ARM CPUs using frameworks backed by **oneDNN** or ACL.
* Small RL loops or online updates where the heavy network stays fixed and you only adjust a head.
* On Apple silicon, *real* training with **[MLX](https://ml-explore.github.io/mlx/build/html/index.html)** is now very usable for research-scale models.

NPUs and mobile GPUs are still primarily exposed for inference; training on them is mostly a research topic, not a production pathway.

---

## 6. Putting it together: a practical “SoC ML stack” mental model

If you’re trying to architect something that spans multiple SoCs (RB8 + AM69A + i.MX93, for example), a good mental model is:

1. **Pick a compiler/runtime axis**

   * Use **TVM** or **IREE** where you want a programmable compiler.
   * Use **ExecuTorch** / **LiteRT** where you want a graph runtime with sane tooling.

2. **Map each deployment to its vendor SDK**

   * Qualcomm -> **QNN SDK**
   * TI -> **TIDL / Edge AI SDK**
   * NXP -> **eIQ**
   * Android phones -> **NNAPI** + GPU / NNAPI / ncnn
   * Apple -> **MLX / Core ML**

3. **Use low-level APIs only when necessary**

   * Reach for **Vulkan + Kompute** (or raw Vulkan/OpenCL) when you truly need custom kernels or non-standard compute.

4. **Keep CPU as a universal fallback**

   * Backed by **oneDNN** or ACL for dense math.
   * Plug LLM use-cases into **vLLM** on ARM servers if you can.

That’s about as close as we currently get to “CUDA, but for heterogeneous ARM SoCs”: not one API, but a layered ecosystem you can standardize around.
