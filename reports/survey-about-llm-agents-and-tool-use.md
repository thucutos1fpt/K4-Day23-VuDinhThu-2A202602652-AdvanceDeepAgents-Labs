# Survey: LLM agents and tool use

## TL;DR

LLM agents that call external tools form a rapidly maturing field spanning self-supervised tool learning, planner/actor decompositions, uncertainty-aware invocation, closed-loop solver grounding, and automated meta-design. Early self-supervised methods (Toolformer) showed models can learn to call APIs from text, while more recent work focuses on separating abstract planning from concrete tool reification, quantifying uncertainty to reduce needless calls, and closing the loop with executor feedback in domain-specific solvers [1][2][3][4]. Industry and open-source tooling (LangChain, AutoGPT, OpenAI Agents SDK, Microsoft Copilot) operationalize these patterns with Tool abstractions, schema-driven integrations, and governance primitives [5][6][7][8]. Evaluation has shifted from short, static benchmarks to long-horizon, sim-to-real, and audit-focused suites (RetailBench, PlanBench-XL, EnvFactory, DFAH) that reveal new failure modes such as blind deference to tools, non-determinism, and long-horizon incoherence [9][10][11][12][13][14]. Hugging Face and community work contribute practical runtimes, benchmarks, and memory/critic patterns (REMORY, Opera) that bridge research and engineering [15][16][10].

## Background

Research on LLM tool use began by demonstrating that models can benefit from external APIs for specialized capabilities (calculator, search, translation) and that such behavior can be induced with self-supervision over web text [1]. Web-enabled answering prototypes (WebGPT) combined human demonstrations and reward-model optimization to train browsing-capable systems that produce citations and reduce some hallucinations while exposing new risks from web access [17][18]. Engineering ecosystems then translated those research concepts into developer-facing Tool primitives, agent factories, and integration patterns that enable rapid composition and deployment of tool-using agents [5][7][19]. Community projects and product platforms (AutoGPT, BabyAGI) distilled simple control-loop patterns (planner/executor/prioritizer with vector-memory) that accelerated experimentation but also surfaced operational and safety caveats [6][20].

## Comparing approaches

- Self-supervised API induction vs. planner-first designs
  - Toolformer uses self-supervised insertion and selection of API-call candidates into pretraining data so models learn when and how to call tools without explicit labels; this made smaller models competitive on tasks by delegating to tools [1].
  - Chain-of-Abstraction (CoA) instead advocates a two-stage decode: generate abstract reasoning chains with placeholders, then reify placeholders via parallel tool calls; CoA reports better accuracy and lower latency by separating strategy from knowledge retrieval [2].

- Uncertainty and selective invocation
  - UALA integrates explicit uncertainty estimation into the agent loop to decide where tool calls are warranted; it reports fewer tool calls and improved efficiency/robustness compared to naive always-call agents, noting that verbalized confidence is unreliable as an uncertainty signal [3].
  - Related work on agent-based detection/resolution shows specialist transducers that classify, clarify, or answer can improve quality on ambiguous inputs at the cost of more LLM invocations and latency, underscoring tradeoffs between interaction length and result quality [21].

- Closed-loop solver grounding and domain-specific executors
  - Embodied CAD demonstrates that iteratively selecting typed CAD operations and using solver feedback to repair plans produces executable artifacts and higher assurance than one-shot script generation, highlighting the value of solver-grounded rewards and repair loops in engineering domains [4].

- Internalization and learned scaffolding
  - Some research explores internalizing tool-like behaviors inside model internals (graded-transformer morphisms) to make operations differentiable and interpretable, offering a path away from external API reliance [22].
  - ADAS proposes automating the design of agentic systems themselves by meta-search that synthesizes agent architectures and components, showing that discovered designs can outperform hand-crafted baselines in multiple domains [23].

- Empirical caution: agent-tool interactions can mislead evaluation
  - Studies show agents may blindly defer to a frozen tool (e.g., a GNN) and that stronger LLM backbones can defer more, suggesting that high agreement with tool outputs does not imply the agent is exercising judgment [13].
  - DFAH-style harnesses argue for separate measurements of determinism/replayability and evidence-conditioned faithfulness because replayable trajectories are not necessarily accurate or faithful to evidence [12].

- Engineering and product practices
  - Frameworks like LangChain provide Tool objects, agent factories, prompt templates, and memory/chain patterns to compose agents; OpenAI and Microsoft provide SDKs and MCP/OpenAPI plugin patterns with schema validation and secure tool registration for production integration [5][7][19][8].
  - Open-source starter projects (AutoGPT, BabyAGI) popularized simple autonomous loops (task creation, execution, prioritization) using vector stores for context; releases and cookbooks emphasize practical concerns like auth, rate limits, and human oversight [6][20][7].

## Trends and open problems

- Trend: richer, deployment-relevant evaluation
  - The community is moving from short QA-style benchmarks to long-horizon, economically and physically grounded simulations (RetailBench), environment synthesis for training (EnvFactory), and planner-focused suites (PlanBench-XL) to surface realistic failure modes such as incoherent long-term policies and sim-to-real brittleness [9][11][10][14].

- Trend: modular, schema-driven tooling + governance
  - Production systems increasingly adopt schema-validated tools (OpenAPI / MCP), registries, and permission-trimmed retrieval to reduce risk when exposing tools to models; these patterns appear in OpenAI's Agents SDK examples and Microsoft Copilot docs and product announcements [19][8].

- Open problem: invocation judgment vs. blind deference
  - Empirical results show agents often defer to tool outputs rather than applying independent judgment, and stronger backbones may defer more—raising the question of how to design gating, verification, and calibration so agents use tools judiciously rather than parroting them [13].

- Open problem: auditability, determinism, and faithfulness
  - Audit-focused work (DFAH) shows determinism and accuracy can be uncorrelated; regulated deployments will need tooling and benchmarks that separately measure replayability, evidence-conditioned faithfulness, and provenance of tool-invoked outputs [12].

- Open problem: long-horizon coherence and memory
  - Benchmarks such as RetailBench and PlanBench-XL highlight that current agents struggle to sustain coherent strategies over long horizons; advances in compact, revisable memory (e.g., REMORY) and persistent critic loops (Opera) are promising but require integration with planning and tool registries [9][10][15][16].

- Open problem: automation of agent design and safe internalization
  - ADAS suggests automating the search for agent scaffolding, while internalization efforts propose embedding tool-like morphisms inside models. Both raise questions about controllability, interpretability, and how to balance end-to-end learning with explicit governance and tool sandboxing [23][22].

- Engineering challenges: sim-to-real, noisy tools, and provenance
  - Sim-to-real gaps (When Simulation Lies) and supply-chain issues around skill provenance (Skill Constellations) show that practical deployments must handle noisy APIs, tool name collisions, and provenance/versioning of shared skills and plugins [14][?].

## References
[1] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[2] Efficient Tool Use with Chain-of-Abstraction Reasoning. arxiv. https://arxiv.org/abs/2401.17464 (2024-01-30)
[3] Towards Uncertainty-Aware Language Agent. arxiv. https://arxiv.org/abs/2401.14016 (2024-01-25)
[4] Embodied CAD: Solver-Grounded LLM Agents for Parametric B-Rep Assembly Modeling. arxiv. https://arxiv.org/abs/2606.31252 (2026-06-30)
[5] LangChain — OpenAI integration (Docs). web. https://docs.langchain.com/oss/javascript/integrations/tools/openai (n.d.)
[6] AutoGPT — GitHub (Significant-Gravitas). web. https://github.com/Significant-Gravitas/AutoGPT (2023-03-16)
[7] OpenAI Cookbook: How to build a tool-using agent with LangChain (notebook). web. https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_build_a_tool-using_agent_with_Langchain.ipynb (n.d.)
[8] Microsoft 365 Copilot — Copilot APIs Overview. web. https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/copilot-apis-overview (n.d.)
[9] RetailBench: Benchmarking long horizon reasoning and coherent decision making of LLM agents in realistic retail environments. arxiv. https://arxiv.org/abs/2606.15862 (2026-06-14)
[10] PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents. hf-search. https://huggingface.co/papers/2606.22388 (2026-06-21)
[11] EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL. hf-search. https://huggingface.co/papers/2605.18703 (2026-05-18)
[12] Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents. arxiv. https://arxiv.org/abs/2601.15322 (2026-01-17)
[13] When the Tool Decides: LLM Agents Defer Blindly to Graph Neural Network Tools, and Stronger Backbones Defer More. arxiv. https://arxiv.org/abs/2606.14476 (2026-06-12)
[14] When Simulation Lies: A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents. hf-search. https://huggingface.co/papers/2605.11928 (2026-05-12)
[15] REMORY: Learning Residual Memory for Context Compaction. hf-daily. https://huggingface.co/papers/2610.11287 (2026-10-08)
[16] Opera: A Verbal Critic Framework for Long-horizon Coding Agents. hf-daily. https://huggingface.co/papers/2609.33987 (2026-10-08)
[17] OpenAI — WebGPT research blog. web. https://openai.com/index/webgpt/ (2021-12-16)
[18] WebGPT (paper). web. https://cdn.openai.com/WebGPT.pdf (2021-12)
[19] OpenAI Developers — ChatGPT Agents SDK / Agents Cookbook (AWS AgentCore example). web. https://developers.openai.com/cookbook/examples/partners/aws/chatgpt_agents_sdk_aws_agentcore_cookbook/notebooks/chatgpt_agents_sdk_aws_agentcore_cookbook.md (n.d.)
[20] BabyAGI — GitHub (yoheinakajima). web. https://github.com/yoheinakajima/babyagi/ (2023-04-03)
[21] Agent-Based Detection and Resolution of Incompleteness and Ambiguity in Interactions with Large Language Models. arxiv. https://arxiv.org/abs/2507.03726 (2025-07-04)
[22] Internalizing Tools as Morphisms in Graded Transformers. arxiv. https://arxiv.org/abs/2511.17840 (2025-11-21)
[23] Automated Design of Agentic Systems. arxiv. https://arxiv.org/abs/2408.08435 (2024-08-15)
