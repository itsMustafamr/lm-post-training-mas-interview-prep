# Bella Yang Interview Prep: LLM Post-Training & Multi-Agent Systems
## Argonne National Laboratory — Aurora Supercomputer Internship

---

# PART 1: Know Your Interviewer & The Project

## Dr. Chih-Hsuan (Bella) Yang
- **Institution:** Argonne National Laboratory, Lemont, IL
- **Division:** Mathematics and Computer Science (MCS) Division
- **Focus:** Large-scale LLM training, multi-agent systems, credit assignment

## Her Key Paper: "Who Gets the Reward & Who Gets the Blame?"
**arXiv: 2511.10687**

Core problem: When a multi-agent system solves a problem collaboratively, how do we assign credit/blame to individual agents to enable per-agent post-training?

**Framework:**
1. System runs → produces outcome reward (correct/incorrect)
2. Credit assignment module distributes reward to agents
3. Each agent receives a training signal proportional to its contribution
4. Agents are fine-tuned with their individual signals (AT-GRPO)

**Key innovations:**
- Shapley value attribution for agent credit
- Error localization via trace walking
- AT-GRPO (Agent- and Turn-wise GRPO)

## The Internship Project
**Title:** Multi-agent LLM pipeline training and evaluation on Aurora

**Day-to-day work:**
- Build and orchestrate multi-agent pipelines (solver, critic, reviser, verifier)
- Execute pipelines on math/reasoning benchmarks (GSM8K, MATH)
- Implement logging and trace analysis systems
- Run large-scale LLM inference on Aurora's HPC infrastructure
- Evaluate pipeline performance with controlled experiments
- Implement credit assignment and per-agent fine-tuning

**Infrastructure:** Aurora supercomputer (Argonne, 1+ exaFLOP)

---

# PART 2: LLM Fundamentals Refresher

## Transformer Architecture

```
Input Tokens → Embedding + Positional Encoding
    ↓
[× N layers]
    Multi-Head Self-Attention
        Q = XW_Q,  K = XW_K,  V = XW_V
        Attention(Q,K,V) = softmax(QK^T / √d_k) · V
    Add & Norm
    Feed-Forward Network (2 linear layers + GELU)
    Add & Norm
    ↓
Output → Language Model Head → Logits → Softmax → Probabilities
```

**Self-attention formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Multi-head attention:** Run H attention heads in parallel, concatenate:
$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

## Autoregressive Generation

At each step t, the model predicts the next token:
$$p(x_t \mid x_1, \ldots, x_{t-1}) = \text{softmax}(W_{\text{lm}} \cdot h_t)$$

**Sampling strategies:**
- **Greedy:** argmax — deterministic, fast, repetitive
- **Temperature:** divide logits by T before softmax (T>1 = more random)
- **Top-p (nucleus):** sample from smallest set with cumulative prob ≥ p

## Key Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Perplexity | $\exp(-\frac{1}{T}\sum \log p(x_t))$ | Language model quality |
| BLEU | n-gram precision with brevity penalty | Translation/summarization |
| ROUGE | n-gram recall | Summarization |
| Pass@k | $1 - \binom{n-c}{k}/\binom{n}{k}$ | Code/math generation |
| Win Rate | Fraction of head-to-head wins | RLHF evaluation |

---

# PART 3: Post-Training Methods — Deep Dive

## 3.1 SFT (Supervised Fine-Tuning)

**What:** Fine-tune a pre-trained LLM on (instruction, response) pairs to teach it to follow instructions.

**Loss function:**
$$\mathcal{L}_{\text{SFT}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x)$$

Where $T$ = number of response tokens, $x$ = instruction, $y$ = response.

**Critical detail:** Only compute loss on *response* tokens. Instruction tokens are masked (label = -100 in HuggingFace).

**When to use SFT:**
- Teaching a base model to follow instructions
- Domain adaptation (medical, legal, math)
- First stage of RLHF/DPO pipeline

**Data creation:**
- Human annotators write (instruction, response) pairs
- Synthetic data: GPT-4 generates diverse instructions
- Self-play: model generates responses, filtered by quality

---

## 3.2 RLHF (Reinforcement Learning from Human Feedback)

**Three-stage pipeline:**

```
Stage 1: SFT
  Pre-trained LM → Fine-tune on demonstrations → π_SFT

Stage 2: Reward Model
  Collect human preferences (y_w ≻ y_l)
  Train R_φ with Bradley-Terry loss

Stage 3: PPO
  Optimize π_θ to maximize R_φ(x,y) - β·KL(π_θ||π_SFT)
```

**PPO Objective:**
$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(r_t A_t,\ \text{clip}(r_t, 1{-}\epsilon, 1{+}\epsilon) A_t\right)\right] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

where $r_t = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$

**KL penalty:** Prevents the policy from drifting too far from $\pi_\text{SFT}$.

**Reward hacking:** The model learns to exploit the reward model — high scores without actual quality improvement.
- Mitigation: KL penalty, ERM, ensemble reward models

**Pros/Cons:**
| Pros | Cons |
|------|------|
| Flexible reward signal | Complex 3-stage pipeline |
| Human preferences directly encoded | Reward hacking risk |
| Works with non-differentiable rewards | PPO is unstable |

---

## 3.3 DPO (Direct Preference Optimization)

**Core idea:** Solve RLHF without training a reward model. DPO shows that the optimal RLHF policy satisfies:

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

**DPO Loss (substitute above into preference objective):**

$$\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

**Intuition:** Increase probability of $y_w$ and decrease $y_l$, but stay close to $\pi_\text{ref}$ (controlled by $\beta$).

**Implementation:**
```python
def dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1):
    chosen_rewards = beta * (pol_w - ref_w)
    rejected_rewards = beta * (pol_l - ref_l)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

**Variants:**
| Variant | Change from DPO | When to use |
|---------|-----------------|-------------|
| **IPO** | Replace sigmoid with identity | Avoids overfit to extreme pairs |
| **KTO** | No paired data needed | When you have unpaired feedback |
| **SimPO** | Reference-free (length-normalized) | Simpler, no ref model needed |
| **ORPO** | Combines SFT + DPO in one stage | More efficient training |

---

## 3.4 GRPO (Group Relative Policy Optimization)

**Used in:** DeepSeek-Math, DeepSeek-R1

**Key insight:** Replace PPO's value function with group normalization. For each prompt, sample G completions and normalize rewards within the group.

**Algorithm:**
1. For prompt $x$, sample $G$ completions $\{y_1, \ldots, y_G\}$ from $\pi_\text{old}$
2. Score each: $r_i = \text{Reward}(x, y_i)$ (binary for math)
3. Group-normalize: $\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$
4. Update with clipped objective + KL penalty

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\sum_{i=1}^G \min\!\left(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1{-}\epsilon, 1{+}\epsilon) \hat{A}_i\right) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

**Advantage over PPO:** No value network → 50% fewer parameters to update; more stable for math.

**DeepSeek-R1 connection:** DeepSeek-R1 uses GRPO with binary (correct/incorrect) rewards on math problems. The model learns to generate long chain-of-thought reasoning because only correct final answers get rewarded.

---

## 3.5 LoRA & PEFT

**LoRA math:**
$$W = W_0 + \Delta W = W_0 + B \cdot A$$
where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$

**Forward pass:**
$$h = W_0 x + \frac{\alpha}{r} B A x$$

**Parameter reduction:**
- Full: $d \times k$ params (e.g., 768×768 = 589,824)
- LoRA: $r(d + k)$ params (r=8: 8×1536 = 12,288) → **98% reduction**

**Multi-agent LoRA:** Load ONE base model on GPU. Each agent gets its own (A, B) adapter. Switch adapters between agent calls — memory efficient!

```
GPU Memory: [Base Model W_0] + [Adapter_solver: A_s, B_s]
                             + [Adapter_critic: A_c, B_c]
                             + [Adapter_reviser: A_r, B_r]
```

---

# PART 4: Multi-Agent Systems — Deep Dive

## Why Multi-Agent?

- **Cognitive diversity:** Different agents catch different errors
- **Specialization:** Solver optimized for generation, critic for verification
- **Error correction:** Iterative refinement loops
- **Ensemble effect:** Aggregating outputs reduces variance
- **Parallelism:** Independent agents run simultaneously

## Communication Topologies

```
Sequential:          Star (Hub):          Fully Connected:
A → B → C → D       A ─── Hub ─── C       A ─── B
                     B ───┘         ─── D   │ × │
                                            C ─── D

Ring:                Hierarchical:
A → B → C           Orchestrator
↑         ↓            ├── Team A → [a1, a2, a3]
D ← E ← F             └── Team B → [b1, b2, b3]
```

## Agent Roles for Math Reasoning

| Role | Responsibility | Key Output |
|------|---------------|------------|
| **Solver** | Generate step-by-step solution | Full reasoning chain + answer |
| **Critic** | Find logical/arithmetic errors | VERDICT: CORRECT/INCORRECT + explanation |
| **Reviser** | Fix errors based on critique | Corrected solution |
| **Verifier** | Extract and validate final answer | ANSWER: [NUMBER] |

## MAS Frameworks Comparison

| Framework | Best For | Topology | Notes |
|-----------|----------|----------|-------|
| **CrewAI** | Role-based teams | Hierarchical | Easy to use, high-level |
| **LangGraph** | Complex state machines | Graph | Fine-grained control |
| **AutoGen** | Conversational MAS | Flexible | Microsoft Research |
| **OpenAI Swarm** | Handoff patterns | Dynamic | Lightweight |

## Design Patterns

1. **Sequential Pipeline:** Solver→Critic→Reviser→Verifier (most interpretable)
2. **Multi-Agent Debate:** N agents argue independently, converge via majority vote
3. **Mixture-of-Agents:** N proposers → 1 aggregator (best quality)
4. **Solver-Verifier:** Generate multiple solutions, verify each, pick best
5. **Hierarchical Planning:** Orchestrator decomposes task, sub-agents solve parts

---

# PART 5: Credit Assignment — Deep Dive

## The Problem

A 4-agent pipeline (Solver→Critic→Reviser→Verifier) produces answer: **wrong**.

Which agent caused the failure?
- Was the Solver's initial solution wrong?
- Did the Critic miss the error?
- Did the Reviser fail to fix it?
- Did the Verifier misextract the answer?

Without credit assignment, we can only train all agents with the same negative signal — inefficient and potentially harmful.

## Bella Yang's Framework

```
System Run → Outcome Reward (0/1)
                ↓
         Credit Assignment Module
         ├── Shapley Values (blame/credit by contribution)
         ├── Error Localization (trace walking)
         └── Process Rewards (step-level signals)
                ↓
         Per-Agent Training Signals {φ_solver, φ_critic, φ_reviser, φ_verifier}
                ↓
         AT-GRPO Fine-tuning (agent-specific updates)
```

## Shapley Value Attribution

The Shapley value gives agent $i$ their average marginal contribution:

$$\varphi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

**Worked Example (3 agents: solver, critic, reviser):**

Coalition values:
- $v(\emptyset) = 0$, $v(\{s\}) = 0.5$, $v(\{c\}) = 0$, $v(\{r\}) = 0$
- $v(\{s,c\}) = 0.7$, $v(\{s,r\}) = 0.6$, $v(\{c,r\}) = 0$
- $v(\{s,c,r\}) = 1.0$

Solver's Shapley:
$$\varphi_s = \frac{1}{3!}\left[v(\{s\}) + (v(\{s,c\})-v(\{c\})) + (v(\{s,r\})-v(\{r\})) + v(\{s,c,r\})-v(\{c,r\})\right] \times \text{weights}$$

**Properties:**
- **Efficiency:** $\sum_i \varphi_i = v(N)$ (all credit sums to outcome)
- **Symmetry:** Equal agents get equal credit
- **Dummy:** Agents contributing nothing get $\varphi = 0$
- **Linearity:** $\varphi_i(v+w) = \varphi_i(v) + \varphi_i(w)$

## Error Localization

**Algorithm:**
1. Walk trace chronologically
2. At each solver/reviser output, check if ground truth number appears
3. First step where ground truth disappears → error introduction point
4. Assign blame signal (-1) to that agent
5. Assign credit (+0.5) to any agent that subsequently corrects it

## Process Reward Models (PRM) vs ORM

| | ORM | PRM |
|-|-----|-----|
| Signal level | Outcome (end of sequence) | Step (each reasoning step) |
| Data needed | Final answer label | Step-level correctness labels |
| Advantages | Easy to collect | More precise, catches early errors |
| Disadvantages | Sparse signal | Expensive to annotate |
| Used in | Most RLHF systems | Let's Verify Step by Step (Lightman) |

## AT-GRPO (Agent- and Turn-wise GRPO)

**Two-level advantage normalization:**

1. **Agent-level:** Normalize across agents on same problem
$$A_i^{\text{agent}} = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

2. **Turn-level:** Normalize across turns within one agent
$$A_{i,t}^{\text{turn}} = \frac{r_{i,t} - \mu_{r_i}}{\sigma_{r_i} + \epsilon}$$

3. **Combined:**
$$A_{i,t} = \alpha \cdot A_i^{\text{agent}} + (1-\alpha) \cdot A_{i,t}^{\text{turn}}$$

---

# PART 6: Aurora Supercomputer & HPC-Scale LLM Inference

## Aurora Specs

| Spec | Value |
|------|-------|
| **Nodes** | 10,624 |
| **GPUs** | 63,744 Intel Ponte Vecchio (GPU Max 1550) |
| **Peak performance** | 1.012 ExaFLOPS (FP64) |
| **Memory per node** | 128 GB HBM2e GPU + 512 GB DDR5 CPU |
| **Storage** | 220 PB Lustre filesystem |
| **Network** | HPE Slingshot-11 (800 Gb/s) |
| **OS** | Cray Operating System |

## Distributed LLM Inference Strategies

### Tensor Parallelism (TP)
Split weight matrices across GPUs. Each GPU holds a slice of the model.
```
W (d×k) → [W_1 (d×k/N), W_2 (d×k/N), ..., W_N (d×k/N)]
```
- Communication: AllReduce after each layer
- Best for: Large models that don't fit on one GPU

### Pipeline Parallelism (PP)
Split layers across nodes. Each node processes one stage.
```
Node 1: Layers 0-7 → Node 2: Layers 8-15 → Node 3: Layers 16-23
```
- Communication: Activations passed between nodes
- Bubble overhead: efficiency = 1 - (p-1)/(m+p-1)

### Data Parallelism (DP)
Same model, different data batches on each node.
```
Node 1: Batch 1 → Node 2: Batch 2 → Node 3: Batch 3
```
- Communication: Gradient AllReduce
- Best for: Training; inference = different prompts to different nodes

### Expert Parallelism (EP) for MoE
Different experts on different GPUs. Tokens routed to expert GPUs.

## Multi-Agent on Aurora Strategy

```
Aurora (10,624 nodes)
    ├── Inference Cluster (e.g., 1,000 nodes)
    │   ├── Agent Group 1 (Solver) — 250 nodes, TP=8, PP=4
    │   ├── Agent Group 2 (Critics, 3×) — 250 nodes
    │   ├── Agent Group 3 (Reviser) — 250 nodes
    │   └── Agent Group 4 (Verifier) — 250 nodes
    └── Pipeline Orchestrator (CPU nodes)
```

## Key Optimization Techniques

| Technique | Description | Speedup |
|-----------|-------------|---------|
| **KV-Cache** | Cache attention keys/values to avoid recomputation | 2-10× |
| **Continuous Batching** | Process requests in a rolling batch, no padding waste | 2-5× throughput |
| **Speculative Decoding** | Small draft model generates tokens, large model verifies | 2-4× |
| **Quantization** | INT8/INT4 weights — half the memory, similar quality | 2× memory |
| **Flash Attention** | Memory-efficient attention computation | 3× memory, 2× speed |

---

# PART 7: Math & Reasoning Benchmarks

## GSM8K
- **Size:** 8,500 grade-school math problems
- **Difficulty:** 2-8 step reasoning, single correct numerical answer
- **Format:** Natural language question → step-by-step solution + answer
- **MAS approach:** Solver generates chain-of-thought, Critic checks each step

## MATH Dataset
- **Size:** 12,500 competition math problems
- **Difficulty:** 5 levels (1=easy, 5=competition), 7 subjects
- **Subjects:** Algebra, Number Theory, Geometry, Calculus, Combinatorics, Probability, Prealgebra
- **MAS approach:** Debate works well for hard problems (3-4 agents, 3 rounds)

## Evaluation Metrics for MAS

| Metric | Formula | What it measures |
|--------|---------|------------------|
| **Accuracy** | correct/total | Final answer quality |
| **Convergence Rate** | problems converged / total | Pipeline efficiency |
| **Error Correction Rate** | (wrong→right) / initially_wrong | Critic+Reviser value |
| **Collaboration Efficiency** | (MAS_acc - SA_acc) / (N_agents - 1) | Value per agent |
| **Pass@k** | $1 - \binom{n-c}{k}/\binom{n}{k}$ | Sample efficiency |

---

# PART 8: Case Studies

## DeepSeek-R1 and GRPO
DeepSeek trained a reasoning model using GRPO with:
- **Reward:** Binary (correct/incorrect) on math problems
- **Group size:** G=8 completions per problem
- **Key finding:** GRPO alone (without SFT warmup) produces "aha moment" behavior
- **Chain-of-thought:** Model learns to reason longer when rewarded only on correctness
- **Performance:** Matches OpenAI o1 on math benchmarks

## Multi-Agent Debate (Du et al., 2023)
- 3 GPT-3.5 agents debate for 3 rounds
- +10% accuracy on MMLU, +15% on math over single agent
- Key insight: Disagreement forces agents to justify and correct themselves

## Mixture-of-Agents (Wang et al., 2024)
- L1: 3 independent proposers (LLaMA-3, Mistral, Qwen)
- L2: 1 GPT-4 aggregator synthesizes best elements
- Outperforms any single model, even on models not in the proposer set
- Reason: Aggregation reduces variance; reference answers help smaller models

## AuroraGPT
- Large-scale LLM pre-trained on Aurora
- Uses scientific literature + code
- Demonstrates Aurora's capability for LLM training at exascale

---

# PART 9: 50+ Interview Questions & Ideal Answers

## Post-Training Fundamentals

**Q1: What is RLHF and why is it used?**
A: RLHF (Reinforcement Learning from Human Feedback) is a 3-stage process for aligning LLMs with human preferences. Stage 1: SFT on demonstrations. Stage 2: Train reward model on preference pairs using Bradley-Terry loss. Stage 3: Fine-tune with PPO to maximize reward while penalizing KL divergence from the SFT model. Used because human preferences are hard to specify as a loss function — RLHF lets the model learn what humans actually want.

**Q2: How does DPO differ from RLHF?**
A: DPO eliminates the reward model entirely. It derives a closed-form relationship between the optimal RLHF policy and the reference policy: r*(x,y) = β log π*(y|x)/π_ref(y|x) + constant. Substituting this into the preference objective gives the DPO loss directly. Advantages: simpler (one stage vs three), no reward hacking, more stable training. Disadvantage: requires paired preferences; can't incorporate online feedback.

**Q3: What is reward hacking?**
A: The policy learns to exploit weaknesses in the reward model to get high scores without actually improving quality. Example: generating very long outputs (if reward correlates with length), repeating tokens the reward model likes, or generating plausible-sounding but incorrect content. Mitigated by: KL penalty (β parameter), ensemble reward models, process rewards instead of outcome rewards.

**Q4: Explain the KL penalty in RLHF/DPO.**
A: The KL divergence term -β·KL(π_θ || π_ref) penalizes the policy for deviating too far from the reference (SFT) model. Without it, the model would exploit the reward to generate degenerate outputs. β controls the tradeoff: high β → stays close to SFT (safe but less improvement); low β → more optimization of reward (risky). In DPO, β appears directly in the loss as the implicit reward scaling factor.

**Q5: What is GRPO and how does it compare to PPO?**
A: GRPO samples G completions per prompt and normalizes advantages within the group (A_i = (r_i - mean)/std), eliminating the need for a value network. Compared to PPO: simpler (no critic network), more stable for math with binary rewards, requires less memory. Compared to DPO: GRPO is on-policy (generates new completions each step), works with any reward function (not just preferences).

**Q6: What is LoRA and why is it important for multi-agent systems?**
A: LoRA reparameterizes weight updates as ΔW = B·A where B∈R^{d×r}, A∈R^{r×k}, r<<d,k. Only A and B are trained. For multi-agent: load one base model on GPU and maintain separate adapters per agent role. During inference, swap adapters between agent calls. This is crucial on Aurora where loading 4 separate copies of a 70B model is infeasible.

**Q7: What is a Process Reward Model and when would you use it?**
A: A PRM assigns a reward to each reasoning step, unlike an ORM which only rewards the final answer. Use PRM when: (1) you want to catch errors early in long reasoning chains, (2) you have labeled step-level data, (3) you're doing credit assignment in multi-agent pipelines (which step went wrong?). PRM trains with step-level correctness labels using cross-entropy per step.

**Q8: What is rejection sampling fine-tuning?**
A: Generate N completions per prompt, keep only those with reward above threshold, fine-tune on those. Simpler than PPO but effective. Used in LLaMA-2, DeepSeek-Math stage 1. Advantage: stable, no RL complexity. Disadvantage: requires many samples, wastes compute on rejected ones.

**Q9: What is Constitutional AI?**
A: Anthropic's method: (1) Model generates response, (2) Model critiques its own response using a list of constitutional principles ("Is this harmful?"), (3) Model revises based on critique. Iteratively apply. Reduces need for human annotation by using the model itself as the critic.

**Q10: Explain iterative DPO.**
A: Train DPO, use the updated model to generate new completions, have the reward model (or humans) label them, create new preference pairs, train DPO again. Repeat. Better than one-shot DPO because on-policy data is more informative. Related to online DPO / self-play fine-tuning.

---

## Multi-Agent Systems

**Q11: Why use multiple agents instead of one large model?**
A: (1) Error correction — critics catch solver errors, improving accuracy by 10-20% on hard math. (2) Specialization — each agent can be fine-tuned for its role. (3) Ensemble effects — aggregating diverse outputs reduces variance. (4) Parallelism — agents can run concurrently on Aurora's distributed nodes. (5) Interpretability — with a trace logger, we know exactly what each agent contributed.

**Q12: How do you design a MAS for math reasoning?**
A: Sequential pipeline: Solver generates step-by-step solution → Critic reviews and outputs VERDICT: CORRECT/INCORRECT → if INCORRECT, Reviser generates improved solution → repeat up to max_rounds → Verifier extracts final answer. Key design choices: (1) System prompts per agent, (2) Convergence criterion (critic says CORRECT), (3) Max rounds to prevent infinite loops, (4) Full trace logging.

**Q13: How do you prevent error cascading in a MAS?**
A: (1) Structured outputs — force agents to use templates (VERDICT: X, ANSWER: Y) for reliable parsing. (2) Confidence thresholds — only accept revisions with high confidence. (3) Majority voting — debate between N solvers, pick majority answer. (4) Verifier as last line of defense — independently extracts and validates answer. (5) Trace logging — detect if an agent is stuck in a loop.

**Q14: How would you evaluate a multi-agent system?**
A: (1) Accuracy on benchmark (GSM8K, MATH). (2) Convergence rate (fraction of problems converging within max_rounds). (3) Error correction rate (wrong→right / total wrong). (4) Collaboration efficiency (accuracy gain per extra agent). (5) Latency (rounds × per-agent latency). (6) Ablation studies: 1 agent vs 2 vs 3, with/without critic, etc.

**Q15: What is the Mixture-of-Agents pattern?**
A: Layer 1 (proposers): N agents independently generate solutions. Layer 2 (aggregator): 1 agent synthesizes all proposals into a final answer. Key insight: even identical agents improve performance because the aggregator sees multiple independent attempts and can recognize the consensus or best approach. Outperforms any single agent in Wang et al. 2024.

---

## Credit Assignment

**Q16: What is the credit assignment problem in MAS?**
A: When a multi-agent pipeline produces an outcome (correct/wrong), the system reward is at the system level, but we need per-agent signals for fine-tuning. How much credit/blame does each agent deserve? Hard because agents interact — the critic's contribution depends on whether the solver made an error; the reviser's value depends on the critique quality.

**Q17: How do Shapley values solve credit assignment?**
A: Shapley values compute each agent's average marginal contribution across all possible coalition orderings. φ_i = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!][v(S∪{i}) - v(S)]. This satisfies fairness axioms: efficiency (Σφ=v(N)), symmetry (equal agents get equal credit), dummy (no-contribution agents get 0). Exponential to compute exactly (2^N coalitions), but Monte Carlo sampling approximates well.

**Q18: What is AT-GRPO?**
A: Agent- and Turn-wise GRPO. Standard GRPO computes advantages by normalizing rewards within a group of G completions from one agent. AT-GRPO extends to multi-agent: (1) Agent-level advantage: normalize across agents on same problem. (2) Turn-level advantage: normalize across turns within one agent. (3) Combine: A_combined = α·A_agent + (1-α)·A_turn. This enables fine-grained credit at both agent and turn resolution.

---

## HPC & Aurora

**Q19: What is tensor parallelism?**
A: Split weight matrices across multiple GPUs. For a linear layer W (d×k), GPU1 holds columns 1 to k/2, GPU2 holds k/2+1 to k. Each GPU computes a partial result; an AllReduce sums them. Suitable for large matrices (attention Q/K/V, FFN). Used in Megatron-LM. Communication: one AllReduce per transformer layer (~2 per attention + 2 per FFN).

**Q20: How would you deploy 100 agents on Aurora?**
A: (1) Determine model size — 7B model needs 14GB (FP16), fits on one GPU with KV-cache. (2) Assign agent groups to node subsets — 10 nodes per agent (10 agents × 10 nodes). (3) Use tensor parallelism within each agent group (TP=8 on one node). (4) Orchestrator runs on CPU nodes, routes problems to agent groups. (5) Use async inference for throughput — pipeline multiple problems simultaneously. (6) Cache base model weights, swap LoRA adapters per agent role.

---

## Practical Design Questions

**Q21: How would you generate training data for a math-solving MAS?**
A: (1) Seed problems from GSM8K/MATH (existing labeled datasets). (2) Run multi-agent pipeline on each problem, log traces. (3) Label outcomes (correct/incorrect by comparing to ground truth). (4) For SFT: use correct solver outputs as demonstrations. (5) For DPO: pair correct vs incorrect revisions as chosen/rejected. (6) For GRPO: use binary reward (correct=1, wrong=0) directly. (7) Augment by rephrasing questions with GPT-4.

**Q22: How would you debug a MAS that has low accuracy?**
A: (1) Log all traces with full agent outputs. (2) Check per-agent accuracy — which agent introduces most errors? (3) Check critic effectiveness — what fraction of incorrect solutions does it catch? (4) Check reviser effectiveness — does revision improve accuracy? (5) Error analysis — categorize failures (arithmetic, logic, misunderstood problem). (6) Ablation — remove agents one at a time to measure their individual contributions. (7) Check for error cascading — is a wrong initial solution propagating through?

---

## Advanced Conceptual

**Q23: How does non-stationarity affect multi-agent training?**
A: In MAS training, agents update simultaneously — the Critic is trained while the Solver is also changing. This creates a non-stationary environment: the optimal critic policy depends on the current solver policy, but both are changing. Solutions: (1) Sequential updates (train Solver for k steps, then Critic). (2) Centralized training with decentralized execution (CTDE). (3) Self-play with periodic opponent updates. (4) Large batch sizes to average out policy changes.

**Q24: What emergent behaviors have been observed in MAS?**
A: (1) Specialization without explicit reward for it — agents develop complementary strategies. (2) Negotiation protocols — agents develop implicit communication patterns. (3) Division of labor — some agents learn to delegate, others to execute. (4) Error detection sophistication — critics learn to check specific error types common in the solver's outputs. These are desirable but hard to guarantee.

---

# PART 10: Discussion Topics & How to Stand Out

## Topics to Proactively Raise
1. **Bella Yang's paper** — "I've read your credit assignment paper. I'm curious how you handle the stochastic nature of Shapley approximation in training..."
2. **Aurora specifics** — "For running inference on Aurora, would you use vLLM with tensor parallelism or a custom inference stack?"
3. **AT-GRPO details** — "When combining agent-level and turn-level advantages, how do you choose α? Is it tuned or fixed?"
4. **Data flywheel** — "Could you create a self-improving loop where each training run generates better data for the next?"

## Questions to Ask
- "What model sizes are you targeting on Aurora — 7B, 70B, or larger?"
- "How many agents does a typical pipeline have in your current experiments?"
- "What's the most surprising finding from your credit assignment experiments?"
- "How do you handle the cold-start problem when all agents have random initial policies?"

## How to Stand Out
- Show you understand the whole stack: theory → implementation → evaluation → scaling
- Reference specific numbers (Aurora specs, benchmark results)
- Propose extensions to her work (e.g., adaptive α in AT-GRPO, curriculum on problem difficulty)

---

# PART 11: Key Papers to Reference

| Paper | Authors | Year | One-line summary |
|-------|---------|------|-----------------|
| InstructGPT | Ouyang et al. | 2022 | RLHF for instruction following |
| DPO | Rafailov et al. | 2023 | Replace RM+PPO with one preference loss |
| DeepSeekMath/GRPO | Shao et al. | 2024 | GRPO with group normalization |
| DeepSeek-R1 | DeepSeek AI | 2025 | GRPO for long chain-of-thought reasoning |
| LoRA | Hu et al. | 2021 | Low-rank adaptation of large LMs |
| MAS Debate | Du et al. | 2023 | Multiagent debate improves factuality |
| Mixture-of-Agents | Wang et al. | 2024 | Aggregating N agents beats any single model |
| Let's Verify Step by Step | Lightman et al. | 2023 | Process reward models for math |
| Who Gets Reward/Blame | Yang et al. | 2024 | Credit assignment for MAS fine-tuning |
| GSM8K | Cobbe et al. | 2021 | Grade-school math benchmark |
| MATH | Hendrycks et al. | 2021 | Competition math benchmark |

---

# PART 12: Cheat Sheet — One-Page Summary

## Post-Training Formulas

| Method | Loss / Key Formula |
|--------|--------------------|
| **SFT** | $\mathcal{L} = -\frac{1}{T}\sum_t \log p_\theta(y_t \mid y_{<t}, x)$ |
| **Reward Model** | $\mathcal{L} = -\log\sigma(R(y_w) - R(y_l))$ |
| **DPO** | $\mathcal{L} = -\log\sigma\!\left(\beta(\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})\right)$ |
| **GRPO Adv.** | $A_i = (r_i - \mu_r)/(\sigma_r + \epsilon)$ |
| **LoRA** | $h = W_0 x + \frac{\alpha}{r}BAx$ |

## MAS Roles

| Role | Input | Output |
|------|-------|--------|
| Solver | Problem | Step-by-step solution + answer |
| Critic | Solution | VERDICT: CORRECT/INCORRECT |
| Reviser | Solution + Critique | Corrected solution |
| Verifier | Final solution | ANSWER: [NUMBER] |

## Shapley Value
$$\varphi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\,[v(S \cup \{i\}) - v(S)]$$

## Aurora Quick Facts
- 10,624 nodes | 63,744 GPUs | 1.012 ExaFLOPS
- Intel Ponte Vecchio GPU | HPE Slingshot-11 network
- Strategies: TP (within node), PP (across nodes), DP (data batches)

## Key DPO Variants

| Variant | β role | Reference-free? |
|---------|--------|----------------|
| DPO | KL penalty | No |
| IPO | Identity loss | No |
| SimPO | Length norm | Yes |
| KTO | Unpaired | No |
