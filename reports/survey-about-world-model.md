# Survey of World Models in AI

## TL;DR

Learned "world models" are internal predictive or generative models used for planning, imagination, and representation learning. Research spans recurrent latent-dynamics models, latent planning methods (PlaNet, Dreamer), model-based planning with learned predictors (MuZero), neural scene representations (GQN, NeRF), graph/relational models, and sequence-modeling approaches (Decision Transformer). Empirical work shows model-based methods can improve sample efficiency but face recurring failure modes (compounding model error, harmful synthetic rollouts, benchmark sensitivity); evaluation practices increasingly emphasize closed-loop task performance, generalization splits, and uncertainty-aware rollouts [1][2][3][4][5][6][7][8].

## Background

World models appear across AI and neuroscience traditions: Schmidhuber framed predictors as intrinsic "world models" and linked curiosity-driven exploration to model improvement objectives [2]. The free-energy / predictive-processing framework in neuroscience formalizes perception as minimizing prediction error using internal generative models [9]. Modern ML instantiations include unsupervised generative recurrent models trained to compress spatio-temporal observations and enable policy learning inside imagined rollouts (World Models) [1].

## Learned latent dynamics and imagination-based agents

Latent-dynamics approaches learn compact state representations and transition models from pixels or other observations, then use those models for planning or to imagine trajectories for policy optimization. World Models trained a VAE + RNN to compress and generate environment sequences, then trained small policies on compressed features and demonstrated policy transfer from hallucinated "dream" environments to the real environment [1].

PlaNet and Dreamer learn latent transition models for planning or actor-critic updates in latent space; PlaNet performs planning in latent space, while Dreamer learns and evaluates behavior entirely via imagined trajectories inside its latent model [3][10]. These methods report strong sample-efficiency on control from pixels benchmarks but require careful treatment of representation learning and rollout design [3][10][8].

## Planning with learned models and hybrid search

MuZero learns a compact predictive model that outputs values and policies (and rewards) for internal search without reconstructing raw observations; combining the learned model with MCTS enabled strong performance on board games and Atari [4]. Analyses suggest planning often helps most during learning (improving targets and data generation), and shallow planning can suffice in many domains; however, planning depth and rollout horizon must be tuned to avoid compounding model error [4][11].

## Neural scene representations and grounding

Neural scene representations like GQN and NeRF model 3D scenes and rendering processes: GQN learns scene representations from multiple views and can render novel viewpoints, supporting downstream policy learning with fewer environment samples in some settings [5]. NeRF represents scenes as continuous radiance fields and yields high-fidelity novel-view synthesis, useful for visual grounding in embodied agents [6].

## Graph-based, relational, and causal world models

Graph networks and relational inductive biases support object-centric and interaction-aware world models suitable for physical reasoning and combinatorial generalization [12]. Structural causal models provide a formalism for encoding mechanisms and answering counterfactuals, aligning with goals for explainable and intervention-capable world models [13].

## Sequence modeling and offline trajectory methods

Sequence-modeling approaches reframe decision-making as conditional generation: Decision Transformer conditions on desired returns and autoregressively predicts actions using Transformer architectures trained on offline trajectories [14]. This line connects large-scale sequence models with implicit world-modeling capacity when trained on diverse trajectory data.

## Evaluation practices, benchmarks, and failure modes

Benchmarking studies and surveys recommend evaluating world models by downstream closed-loop performance (task success, robustness, safety), sample-efficiency (env steps and wall-clock), and generalization (procedural hold-outs) rather than only prediction loss [7][8]. Empirical critiques show synthetic rollouts can harm learning in some suites (e.g., Dyna-style methods underperforming on DMC) and advocate cross-benchmark validation and short/uncertainty-aware rollouts [15][8][11][16].

## Trends and open problems

Key trends: growing emphasis on latent-space planning (less expensive than pixel-space planning), uncertainty-aware and short rollouts to mitigate compounding errors, object-centric and causal representations for generalization, and integration of large sequence models for decision-making [3][10][6][12][14].

Open problems: reliable long-horizon planning under model error, grounding learned models to real-world sensors and effectors (sim-to-real), principled evaluation of hallucination and counterfactual reasoning, and scalable methods for continual world-model revision and uncertainty bookkeeping [17][7][16].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-27)
[2] Adaptive confidence and adaptive curiosity (A possibility for implementing curiosity and boredom in model-building neural controllers). web. https://people.idsia.ch/~juergen/FKI-149-91ocr.pdf (1991)
[3] Learning Latent Dynamics for Planning from Pixels (PlaNet). hf-search. https://huggingface.co/papers/1811.04551 (2019-06-04)
[4] MuZero. hf-search. https://huggingface.co/papers/1911.08265 (2019-11-19)
[5] Neural scene representation and rendering (Generative Query Network). web. https://www.science.org/doi/10.1126/science.aar6170 (2018-06-15)
[6] NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. hf-search. https://huggingface.co/papers/2003.08934 (2020-03-19)
[7] World-Action Models for Robot Learning and Control: A Survey. arxiv. https://arxiv.org/abs/2609.16074 (2026-09-13)
[8] Benchmarking Model-Based Reinforcement Learning. web. https://www.cs.toronto.edu/~tingwuwang/mbrl/mbrl.pdf (n.d.)
[9] The free-energy principle: a unified brain theory?. web. https://www.nature.com/articles/nrn2787 (2010)
[10] DreamerV2 (Dreamer family). hf-search. https://huggingface.co/papers/2010.02193 (2022-02-12)
[11] Fixing That Free Lunch: When, Where, and Why Synthetic Data Fails in Model-Based Policy Optimization. web. https://www.scholarfeed.org/paper/2510.01457v2 (2025-10-01)
[12] Relational inductive biases, deep learning, and graph networks. hf-search. https://huggingface.co/papers/1806.01261 (2018-06-04)
[13] Counterfactuals and Policy Analysis in Structural Models. hf-search. https://huggingface.co/papers/1302.4929 (2013-02-20)
[14] Decision Transformer. hf-search. https://huggingface.co/papers/2106.01345 (2021-06-02)
[15] Stealing That Free Lunch: Exposing the Limits of Dyna-Style Reinforcement Learning. web. https://exa.ai/library/publication/bt4lbldlrtc (2024-12-18)
[16] Rollout Efficiency in Reinforcement Learning for Reasoning Large Language Models: A Taxonomy and Future Directions. arxiv. https://arxiv.org/abs/2609.25463 (2026-09-21)
[17] Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks. hf-daily. https://huggingface.co/papers/2610.11794 (2026-10-08)
