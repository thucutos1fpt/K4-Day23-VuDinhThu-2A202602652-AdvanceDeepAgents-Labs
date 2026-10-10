# Survey: Efficient Inference and Small Language Models

## TL;DR

Efficient inference for large language models (LLMs) uses complementary approaches at algorithmic, model, and system levels. Post-training quantization (GPTQ and variants) and 4-bit weight+FP16 activation recipes are dominant practical techniques for large memory and latency gains with modest quality loss [1][2][3][4]. Hybrid methods that combine quantized bases with low-rank adapters or explicit outlier handling (e.g., W ≈ Q + LR, OSAQ) improve accuracy when pushing to lower bitwidths [5][6]. Speculative decoding, attention-aware quantization objectives, and kernel-level optimizations (FlashAttention) reduce generation latency beyond precision reductions [7][8][9]. Methods for producing small models include knowledge distillation (DistilBERT), structured dropout (LayerDrop), parameter-efficient fine-tuning (LoRA/QLoRA), and retrieval-augmented approaches; QLoRA enables finetuning of very large models in constrained memory [10][11][12][4][13]. Benchmarks and runtime ecosystems (lm-eval-harness, Open LLM Leaderboard, vLLM, TGI, DeepSpeed ZeRO, FSDP) provide reproducible evaluation and deployment recipes; practitioner guides emphasise Pareto analysis of throughput/latency/cost and measurement-first workflows [14][15][16][17][18][19][20][21][22].

## Background

Research and practice split efficient LM inference into at least three layers:
- Model-level techniques that change parameters or architecture (distillation, LoRA, LayerDrop) to reduce compute or tunable parameters [10][12][11].
- Numeric/representation techniques (post-training quantization, quantization-aware training, and low-rank adapters combined with quantized bases) which reduce memory and memory-bandwidth pressure [1][2][4][5][6][3].
- System/runtime techniques (optimized attention kernels like FlashAttention, inference engines such as vLLM/TGI, and sharding/offload systems like DeepSpeed ZeRO and FSDP) that reduce wall-clock latency and support large or sharded deployments [9][17][18][19][20].

Effective deployment combines these layers and measures tradeoffs (throughput, latency, quality, cost) using benchmarking suites and measurement workflows before settling on configuration [21][22].

## Algorithmic and numeric approaches compared

Post-Training Quantization (PTQ) and GPTQ
- GPTQ-style PTQ is widely used to quantize weights to 4-bit (and sometimes 3/2-bit) while keeping activations at FP16/BF16; tools like AutoGPTQ and llm-compressor provide practical recipes and calibration guidance [1][2].
- Practical outcomes: HF blog and llm-compressor examples show large model-size reductions (e.g., ~64% reduction in an example W4A16 run) with small perplexity increases (~+4.4% in the documented example) when using W4A16 [2].
- Limitations: extreme low-bit quantization exposes outlier weights and functional degradation; targeted fixes include outlier-aware techniques and low-rank corrections [6][5].

Hybrid quantization + low-rank adapters
- Recent arXiv work formalizes combining a quantized base Q with a low-rank correction LR (W ≈ Q + LR) and proposes near-optimal algorithms for this joint problem, arguing improved recovery of accuracy without reverting to full precision [5].
- Practical LoRA-style adapters stacked on top of a quantized base (QLoRA style) enable memory-efficient finetuning and preserve much of fine-tuning performance on instruction datasets [4].

Outlier handling and attention-aware objectives
- Papers identify that outliers (large-magnitude weight elements) and attention-sensitivity drive PTQ failures; methods like OSAQ and attention-output reconstruction objectives (joint QKV losses) target these failure modes and claim improved accuracy for aggressive quantization regimes [6][8].

Decoding-level acceleration and speculative decoding
- Speculative decoding techniques (folding speculative branches, multi-branch verification) reduce per-sample computation by reusing computations across draft branches; HF daily summaries highlight SpecFold-style approaches for folding multi-branch redundancy in speculative decoding to speed generation [7].
- These approaches are complementary to quantization: they reduce compute per token independently of weight precision.

Kernel and operator optimizations
- FlashAttention is a high-impact kernel-level improvement that reduces IO and memory traffic in attention computation, yielding wall-clock speedups and lower memory use; inference engines adopt such kernels to push throughput and accommodate longer contexts [9].

## Model-level compression and adaptation methods compared

Knowledge distillation and LayerDrop
- Distillation (DistilBERT) reduces parameter count during pre-training and retains much of the original model's capability with significant speed/size gains [10].
- LayerDrop trains models to be robust to layer removal, enabling depth reduction at inference without retraining small models from scratch; sub-networks extracted from LayerDrop-trained models often outperform small-from-scratch baselines [11].

Parameter-efficient fine-tuning (LoRA, QLoRA)
- LoRA injects low-rank adapters and freezes the base model, greatly reducing trainable parameter count; QLoRA applies low-bit quantization plus LoRA to allow finetuning very large models with modest GPU memory [12][4].
- Outcomes: QLoRA enabled finetuning of 65B-class models on a single 48GB GPU and produced models competitive on instruction-following benchmarks per Hugging Face reports [4].

RAG and retrieval as a model-size substitute
- Retrieval-augmented generation is a deployment pattern (not in the gathered notes as detailed experiments) that reduces the need for large parametric knowledge by bringing external context at inference time; it is frequently discussed alongside compression methods in practitioner guides but specific sources in this survey were limited to more general mentions [15][21].

## Benchmarks, runtime systems, and deployment tradeoffs

Benchmarks and evaluation frameworks
- lm-eval-harness (EleutherAI) underlies reproducible evaluation across many benchmarks (MMLU, BigBench), and Hugging Face's Open LLM Leaderboard uses it for cross-model comparisons [15][14][23][16].
- tinyBenchmarks proposes small, reliable subsets to speed evaluation and reduce cost when comparing many model/configurations [16].

Inference engines, kernels, and sharding
- vLLM targets high-throughput, low-latency serving with attention optimizations, KV-cache paging, and integration with parallelism strategies; TGI is a Rust/Python inference stack used in production but now in maintenance mode [17][18].
- DeepSpeed ZeRO and PyTorch FSDP provide sharding and offload patterns for training and inform similar strategies for serving very large or sharded models [19][20].

Practical measurement and cost tradeoffs
- NVIDIA GenAI-Perf guidance and DigitalOcean practitioner guides recommend Pareto analysis of throughput/latency/cost and measuring the saturation knee (B_sat) to select precision, batching, and hardware configurations that meet SLOs at minimal TCO [21][22].

## Trends and open problems

Trends
- Convergence on 4-bit weight quantization (with FP16 activations) as a practical sweet spot for many production deployments, supported by tooling (AutoGPTQ, llm-compressor) and optimized kernels [1][2][3][4].
- Growing interest in hybrid approaches that combine quantized weights with small learned corrections (low-rank adapters, outlier-absorbing transforms) to push lower bitwidths while retaining accuracy [5][6].
- Increasing attention to decoding-level optimizations (speculative decoding, folding multi-branch redundancy) and system-level kernels (FlashAttention) to reduce latency without changing model precision [7][9].

Open problems
- Predicting end-to-end quality vs latency tradeoffs: many papers propose improvements at one level (quantization, adapter correction, attention objective) but do not provide comprehensive wall-clock latency and real-world throughput numbers across engines and hardware platforms; cross-system benchmarking remains limited [8][5][6].
- Understanding knowledge retention in small/ distilled/finetuned models: on-policy distillation selectively transfers skills vs factual knowledge per recent analysis, raising questions about how to preserve parametric knowledge in small models or during distillation pipelines [13].
- Automated toolchains for joint optimization: end-to-end pipelines that jointly optimize precision, adapter rank, decoding strategy, and kernel selection for a target latency/throughput SLO are still nascent — most work focuses on one or two knobs at a time [5][7][17].
- Extreme low-bit inference with guaranteed accuracy: methods to make 2–3-bit quantization reliably usable across architectures and workloads need better outlier handling and principled calibration; hybrid W ≈ Q + LR methods are promising but require more practical evaluation and tooling [6][5].

## References
[1] Making LLMs lighter with AutoGPTQ and transformers (Hugging Face blog). web. https://huggingface.co/blog/gptq-integration (2023-08-23)
[2] Understanding Post-Training Quantization with LLM Compressor (Hugging Face blog). web. https://huggingface.co/blog/rishiraj/llm-compressor (n.d.)
[3] Introduction to Quantization (Hugging Face blog). web. https://huggingface.co/blog/merve/quantization (2023-08-25)
[4] QLoRA: Efficient Finetuning of Quantized LLMs. hf-search. https://huggingface.co/papers/2305.14314 (2023-05-23)
[5] GPTQ-intrinsic LoRA: A Near-optimal Algorithm for Low-precision Quantization with Low-rank Adaptation. arxiv. https://arxiv.org/abs/2606.01412 (2026-06-01)
[6] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization. arxiv. https://arxiv.org/abs/2605.04738 (2026-05-06)
[7] SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models. hf-daily. https://huggingface.co/papers/2610.04875 (2026-10-06)
[8] From Attention Sensitivity to Layer Role: Revisiting Mixed-Precision Quantization of Transformers. arxiv. https://arxiv.org/abs/2609.34866 (2026-09-28)
[9] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arxiv. https://arxiv.org/abs/2205.14135 (2022-06-23)
[10] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arxiv. https://arxiv.org/abs/1910.01108 (2019-10-02)
[11] Reducing Transformer Depth on Demand with Structured Dropout (LayerDrop). arxiv. https://arxiv.org/abs/1909.11556 (2019-09-25)
[12] LoRA: Low-Rank Adaptation of Large Language Models. arxiv. https://arxiv.org/abs/2106.09685 (2021-06-17)
[13] On-Policy Distillation Teaches New Skills but Not New Knowledge. hf-daily. https://huggingface.co/papers/2610.09639 (2026-10-07)
[14] Open LLM Leaderboard - About. web. https://huggingface.co/docs/leaderboards/main/open_llm_leaderboard/about (n.d.)
[15] EleutherAI / lm-evaluation-harness. web. https://github.com/EleutherAI/lm-evaluation-harness (n.d.)
[16] tinyBenchmarks: evaluating LLMs with fewer examples. hf-search. https://huggingface.co/papers/2402.14992 (2024-02-22)
[17] vLLM documentation (usage, parallelism & backends). web. https://docs.vllm.ai/en/stable/usage/ (n.d.)
[18] Text Generation Inference (TGI) — Hugging Face. web. https://huggingface.co/docs/text-generation-inference/en/index (n.d.)
[19] DeepSpeed ZeRO docs (ZeRO and ZeRO-Infinity tutorial). web. https://github.com/deepspeedai/DeepSpeed/blob/ad026a1f/docs/_tutorials/zero.md (n.d.)
[20] PyTorch Fully Sharded Data Parallel (FSDP) tutorial. web. https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html (n.d.)
[21] How Much Does Your LLM Inference Cost? (NVIDIA blog, GenAI-Perf). web. https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/ (2025-06-18)
[22] The LLM Inference Trilemma: Throughput, Latency, Cost (practitioner guide). web. https://www.digitalocean.com/blog/llm-inference-tradeoffs (2026-04-22)
[23] Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models (BIG-bench). arxiv. https://arxiv.org/abs/2206.04615 (2022-06-12)
