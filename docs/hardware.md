# Hardware & Performance Optimization Guide

Running LLMs and transcription models locally requires optimizing hardware resource allocations. This guide explains how to select the right GGUF models, allocate VRAM, and tune parameters to get the fastest output (tokens/sec).

---

## 1. Choosing GGUF Quantization Levels

Large models are compressed (quantized) from floating-point weights (`F16` or `F32`) to lower bit representations (e.g. 4-bit, 8-bit) to save RAM and run faster.

| Quantization Type | Bits | Memory Requirement | Perplexity (Loss of Quality) | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **`IQ4_XS` / `Q4_0`** | ~4.0 | Extremely Low | Minor / Low | Great for low-spec CPUs (e.g. laptops, Raspberry Pi). |
| **`Q4_K_M`** | ~4.5 | Low | Very Low | **Sweet spot** for most users. Balanced speed and quality. |
| **`Q5_K_M`** | ~5.5 | Medium | Negligible | Excellent quality, slightly slower generation. |
| **`Q8_0`** | 8.0 | High | Near Zero | Very close to unquantized. Best for high-RAM/VRAM machines. |
| **`F16`** | 16.0 | Extremely High | Zero (Original) | Requires massive memory. Only use on professional workstations. |

---

## 2. Memory (RAM / VRAM) Allocation Calculations

Before running a model, estimate if your system can load it entirely to prevent system thrashing (swapping to disk):

$$\text{Required RAM} \approx (\text{Model Parameters in Billions} \times \text{Bits per weight}) + \text{Context Buffer (KV Cache)}$$

*   *Example*: A **7B** model at **Q4_K_M** (~4.5 bits) requires:
    $$(7 \times 4.5) / 8 \approx 3.9 \text{ GB}$$
    Add $\sim 1\text{ to }2\text{ GB}$ of context buffer space. You will need at least **6 GB** of available RAM/VRAM.

### CPU vs GPU (VRAM Offloading)
*   **CPU (System RAM)**: Slowest. If running on CPU, ensure you use faster DDR4/DDR5 system RAM.
*   **GPU (VRAM)**: Up to 10x-20x faster. Offloading the entire model to the GPU provides the fastest response speeds.
*   **Partial Offloading**: If your GPU VRAM is smaller than the model size, you can offload a subset of layers (e.g. 20 layers on GPU, remaining 12 on CPU). Performance will scale linearly based on the proportion of layers running on the GPU.

---

## 3. CPU Thread Allocation Optimization

By default, the backend `llama-server` tries to use multiple CPU threads. However, allocating too many threads can cause CPU thrashing, actually degrading inference speeds.

### Best Practices:
*   **Physical Cores vs Logical Cores**: Only allocate threads equal to your CPU's **physical cores**, NOT logical cores (hyperthreads). Hyperthreading adds processing overhead during matrix operations.
    *   *Example*: If you have an 8-core CPU with hyperthreading (16 logical processors), set your threads to **8**.
*   **NUMA Node Isolation**: On multi-socket server CPUs, configure thread counts to align with a single NUMA node socket to avoid memory travel latency between sockets.
*   **GPU Offload Multi-threading**: If you are offloading 100% of the model to a GPU, CPU thread settings are negligible since the GPU performs all compute operations.

---

## 4. Troubleshooting Low Performance

*   **Inference Speeds Dropping to < 2 tokens/sec**: The model size is likely exceeding your hardware's available memory, causing it to fall back to swap memory (pagefile) on your hard drive. Choose a lower parameters model (e.g. 3B instead of 8B) or a tighter quantization level.
*   **System Freezing**: Ensure the gateway's CPU usage is moderated by setting thread limits on low-spec devices, and check that no other intensive applications are sharing VRAM.
