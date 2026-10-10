# Survey: Video and Multimodal Generation

## TL;DR

Recent progress in video generation has shifted the state of the art from early GAN/VAE-based pipelines to diffusion-dominated and transformer-hybrid approaches. Latent diffusion (often combined with video VAEs and cascaded super-resolution/interpolation stages) and transformer-hybrid designs are widely used for high-fidelity text-to-video and multimodal generation. Conditioning uses frozen text encoders (T5/CLIP), cross-attention, and modular priors to combine text, images, and audio. Evaluation relies on FVD/FID/CLIPScore and human studies, but important gaps remain in temporal and multimodal evaluation and dataset provenance [1][2][3][4][5][6][7].

## Background

Video generation methods evolved from early GANs and VAEs to autoregressive token models, diffusion models, and transformer hybrids. Diffusion-based models (DDPM, LDM) adapted to video with spatiotemporal UNets, latent VAEs, and transformer denoisers; autoregressive token-based systems compress video into discrete tokens for efficient long-range modeling [6][7][3][5]. Staged pretraining (image → video), latent compression, and cascaded architectures (base generation + temporal/spatial super-resolution) are prevalent engineering patterns [3][4][8].

## Architectures and temporal modeling

- Diffusion & Latent Diffusion: Modern high-quality T2V systems commonly use latent diffusion: a video VAE compresses frames into latents, and a spatiotemporal UNet or transformer denoiser performs conditional denoising in latent space. Latent diffusion reduces compute and enables higher-resolution generation [3][8][6].

- Transformer hybrids and autoregressive tokenization: Some approaches use discrete tokenizers and transformer decoders to model long sequences (Phenaki) or hybrid diffusion-transformer designs (CogVideoX) to gain sequence modeling benefits for long-range coherence [5][9][7].

- Temporal coherence techniques: Temporal self-attention, factorized space-time blocks (2D spatial + 1D temporal), hierarchical frame-rate tokens, and interpolation / super-resolution stages are common strategies to balance spatial fidelity and temporal dynamics; staged finetuning from image models provides strong spatial priors that help visual quality [3][4][6].

Evidence: surveys and model pages summarize these design choices and trade-offs [6][7][8].

## Conditioning and multimodal interfaces

- Text conditioning: Many systems use frozen text encoders (T5 or CLIP) with classifier-free guidance or cross-attention in denoisers. Cascaded designs or priors map text to visual latents (Make-A-Video) or tokens (Phenaki) to handle missing paired text-video data [3][4][5].

- Image and video conditioning: Image-conditioned generation encodes reference images into latents and uses cross-attention or concatenation. Motion adapters (AnimateDiff) inject motion priors into frozen T2I models to synthesize temporally coherent frames while preserving image-domain personalization [10].

- Audio and multimodal fusion: Audio datasets (AudioSet) are used to pretrain audio encoders; audio conditioning is typically handled via learned audio encoders and cross-attention or multimodal embedding alignment [11]. Sora and other product pages describe multi-shot conditioning and recaptioning to improve alignment [12].

Representative systems and their conditioning mechanisms: Imagen Video (frozen T5, cascaded diffusion, temporal attention) [3]; Make-A-Video (text-image priors + motion learned from unlabeled video) [4]; Phenaki (discrete video tokens + transformer generation) [5]; AnimateDiff (motion adapters on frozen T2I) [10]; Sora (spacetime transformer attention, recaptioning) [12].

## Evaluation, datasets, applications, and risks

- Datasets: Large-scale web-mined datasets (HowTo100M, WebVid) power pretraining but contain noisy alignments and provenance concerns (watermarks, copyright). Supervised benchmarks like Kinetics and UCF101 remain useful for controlled evaluation [1][2][13][14][15][16].

- Metrics: Image metrics (FID), video metrics (FVD), and cross-modal metrics (CLIPScore) are commonly used. However, FID/FVD inherit biases from pretrained feature extractors and can miss temporal or semantic errors; newer metrics (STREAM) aim to separate spatial and temporal evaluation [17][18][19][20]. Human studies remain essential for assessing perceptual quality and misuse potential [21].

- Applications: film/VFX, virtual avatars/telepresence, creative content generation, accessibility (captioning). 

- Risks: Deepfakes, copyright violations from training on copyrighted materials (e.g., WebVid provenance issues), privacy harms (training on people without consent), and representational bias are documented concerns [15][16][21].

## Trends and open problems

- Trend: Dominance of diffusion-based pipelines and increasing use of latent compression and transformer modules for long-range temporal modeling. Cascaded pipelines and staged pretraining remain engineering norms to reuse image models while adding temporal capacity [6][7][3][8].

- Trend: Modular conditioning strategies that reuse frozen encoders (T5/CLIP) and inject modality-specific priors (motion adapters, audio encoders) to enable multimodal control without retraining massive models end-to-end [3][4][10][12][11].

- Open problem: Evaluation gap — existing metrics poorly capture temporal coherence, multimodal alignment, and physical plausibility. There is a need for standardised human evaluation protocols and metrics with demonstrated correlation to human judgments across domains [17][20][19].

- Open problem: Dataset provenance and ethics — web-mined datasets introduce copyright and privacy risks; curation and documentation (model cards) are necessary but unevenly adopted [15][16][21].

- Open problem: Efficient long-duration generation — sampling latency and compute costs for diffusion-based models motivate research in distillation, few-step samplers, and better latent designs (e.g., VDOT++, world-attention architectures) [6].

- Open problem: Controllable dynamics and physical plausibility — ensuring generated motion obeys physics, preserves identity, and supports precise editing/conditioning remains challenging; motion-customization and adapter techniques help but are limited by base-model priors [10][22].

## References
[1] HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. arxiv. https://arxiv.org/abs/1906.03327 (2019)
[2] Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval (WebVid). arxiv. https://arxiv.org/abs/2104.00650 (2021)
[3] Imagen Video: High Definition Video Generation with Diffusion Models. arxiv. https://arxiv.org/abs/2210.02303 (2022-10-05)
[4] Make-A-Video: Text-to-Video Generation without Text-Video Data. arxiv. https://arxiv.org/abs/2209.14792 (2022-09-29)
[5] Phenaki: Variable Length Video Generation From Open Domain Textual Description. arxiv. https://arxiv.org/abs/2210.02399 (2022-10-05)
[6] Video Diffusion Models: A Survey. arxiv. https://arxiv.org/abs/2405.03150 (2024)
[7] From Sora What We Can See: A Survey of Text-to-Video Generation. arxiv. https://arxiv.org/abs/2405.10674v1 (2024)
[8] Stable Video Diffusion (Hugging Face model card). hf-search. https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt (2023)
[9] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arxiv. https://arxiv.org/abs/2408.06072 (2024-08-12)
[10] AnimateDiff project page. web. https://animatediff.github.io/ (2023)
[11] AudioSet (dataset). web. https://research.google/pubs/audio-set-an-ontology-and-human-labeled-dataset-for-audio-events/ (2017)
[12] OpenAI Sora product/technical pages. web. https://openai.com/index/sora/ (2024)
[13] Kinetics Human Action Video Dataset. web. https://deepmind.com/research/open-source/kinetics (2017)
[14] UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild. arxiv. https://arxiv.org/abs/1212.0402 (2012)
[15] WebVid dataset landing / GitHub. web. https://m-bain.github.io/webvid-dataset/ (n.d.)
[16] WebVid provenance investigation (Simon Willison blog). web. https://simonwillison.net/2022/Sep/29/webvid/ (2022)
[17] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (introduces FID). web. https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium (2017)
[18] Fréchet Video Distance (FVD) implementations and references. web. https://github.com/google-research/fvd (n.d.)
[19] CLIPScore: A Reference-free Evaluation Metric for Image Captioning. arxiv. https://arxiv.org/abs/2104.08718 (2021)
[20] STREAM: Spatio-Temporal Evaluation and Analysis Metric for Video Generative Models. hf-search. https://huggingface.co/papers/2403.09669 (2024)
[21] Model Cards for Model Reporting. arxiv. https://arxiv.org/abs/1810.03993 (2019)
[22] Pumpire: Unified Benchmark for Metric Distance Estimation (hf-daily). hf-daily. https://huggingface.co/papers/2610.12423 (2026-10-08)
