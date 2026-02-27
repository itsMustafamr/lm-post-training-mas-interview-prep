# LLM Post-Training & Multi-Agent Systems — Interview Prep Repository

> **Goal:** Master LLM post-training methods and multi-agent system (MAS) design for an internship at **Argonne National Laboratory** focused on training and evaluating multi-agent systems of large language models at leadership scale on the **Aurora supercomputer**.

---

## 🗺️ Topic Map

```
Transformer LLMs
    └── Post-Training
            ├── SFT  (Notebook 01)
            ├── RLHF (Notebook 02)  ←─ Reward Model (Notebook 06)
            ├── DPO  (Notebook 03)
            ├── GRPO (Notebook 04)
            └── LoRA (Notebook 05)
    └── Multi-Agent Systems
            ├── Basics    (Notebook 07)
            ├── Debate    (Notebook 08)
            └── Credit Assignment (Notebook 09)  ←─ Shapley, PRM, AT-GRPO
    └── Full Pipeline (Notebook 10) — ties everything together
```

---

## 📚 Learning Path

| # | Notebook | Topic | Key Concepts |
|---|----------|-------|--------------|
| 01 | `01_SFT_Supervised_Fine_Tuning.ipynb` | SFT | Cross-entropy loss, label masking, instruction tuning |
| 02 | `02_RLHF_Reinforcement_Learning_Human_Feedback.ipynb` | RLHF | PPO, reward model, KL penalty |
| 03 | `03_DPO_Direct_Preference_Optimization.ipynb` | DPO | DPO loss derivation, chosen/rejected pairs |
| 04 | `04_GRPO_Group_Relative_Policy_Optimization.ipynb` | GRPO | Group advantages, DeepSeek-R1 |
| 05 | `05_LoRA_Parameter_Efficient_Fine_Tuning.ipynb` | LoRA | ΔW=BA, rank reduction, multi-agent adapters |
| 06 | `06_Reward_Modeling.ipynb` | Reward Models | Bradley-Terry, PRM vs ORM |
| 07 | `07_Multi_Agent_Basics.ipynb` | MAS Basics | Solver→Critic→Reviser→Verifier pipeline |
| 08 | `08_Multi_Agent_Debate_System.ipynb` | Debate & MoA | Debate convergence, Mixture-of-Agents |
| 09 | `09_Multi_Agent_Credit_Assignment.ipynb` | Credit Assignment | Shapley values, error localization, AT-GRPO |
| 10 | `10_Full_Pipeline_MAS_PostTraining.ipynb` | End-to-End | Build → Run → Credit → Train → Evaluate |

**Recommended order:** 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10

---

## ⚙️ Setup

### Local (pip)
```bash
git clone <this-repo>
cd lm-post-training-mas-interview-prep
pip install -r requirements.txt
pip install -e .          # installs src/ as 'lm_mas_prep' package
jupyter notebook
```

### Google Colab
Each notebook starts with a `!pip install` cell — just open and run.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### HPC / Aurora
```bash
module load conda
conda create -n mas python=3.10
conda activate mas
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
├── README.md
├── requirements.txt
├── setup.py
├── interview_prep/
│   └── Bella_Yang_Interview_Prep_LLM_PostTraining_MAS.md
├── notebooks/
│   ├── 01_SFT_Supervised_Fine_Tuning.ipynb
│   ├── 02_RLHF_Reinforcement_Learning_Human_Feedback.ipynb
│   ├── 03_DPO_Direct_Preference_Optimization.ipynb
│   ├── 04_GRPO_Group_Relative_Policy_Optimization.ipynb
│   ├── 05_LoRA_Parameter_Efficient_Fine_Tuning.ipynb
│   ├── 06_Reward_Modeling.ipynb
│   ├── 07_Multi_Agent_Basics.ipynb
│   ├── 08_Multi_Agent_Debate_System.ipynb
│   ├── 09_Multi_Agent_Credit_Assignment.ipynb
│   └── 10_Full_Pipeline_MAS_PostTraining.ipynb
├── src/
│   ├── agents/          # BaseAgent, Solver, Critic, Reviser, Verifier
│   ├── orchestration/   # Pipeline, Debate, MixtureOfAgents, Logger
│   ├── training/        # SFT, DPO, GRPO, RLHF, RewardModel, LoRA
│   ├── credit_assignment/ # Shapley, ErrorLocalizer, PRM, AT-GRPO
│   ├── evaluation/      # Benchmarks, Metrics, Visualization
│   └── data/            # GSM8K, MATH, PreferenceDataset loaders
└── configs/
    ├── sft_config.yaml
    ├── dpo_config.yaml
    ├── grpo_config.yaml
    └── mas_config.yaml
```

---

## 🔑 Key Concepts Quick Reference

| Concept | One-line summary | Where |
|---------|-----------------|-------|
| **SFT** | Fine-tune on (instruction, response) pairs | NB 01, `src/training/sft_trainer.py` |
| **RLHF** | SFT → Reward Model → PPO | NB 02, `src/training/rlhf_trainer.py` |
| **DPO** | Replace RM + PPO with one preference loss | NB 03, `src/training/dpo_trainer.py` |
| **GRPO** | Group-normalized advantages, no value fn | NB 04, `src/training/grpo_trainer.py` |
| **LoRA** | ΔW = B·A, train only r(d+k) params | NB 05, `src/training/lora_utils.py` |
| **Reward Model** | Bradley-Terry pairwise loss | NB 06, `src/training/reward_model.py` |
| **MAS Pipeline** | Solver→Critic→Reviser→Verifier | NB 07, `src/orchestration/pipeline.py` |
| **Debate** | Agents argue, majority wins | NB 08, `src/orchestration/debate.py` |
| **MoA** | Many proposers + one aggregator | NB 08, `src/orchestration/mixture_of_agents.py` |
| **Shapley** | Fair credit attribution via marginal contribution | NB 09, `src/credit_assignment/shapley.py` |
| **Error Localization** | Walk trace to find first error agent | NB 09, `src/credit_assignment/error_localization.py` |
| **AT-GRPO** | Agent- and Turn-wise GRPO for MAS | NB 09, `src/credit_assignment/at_grpo.py` |

---

## 🎯 Interview Prep

See [`interview_prep/Bella_Yang_Interview_Prep_LLM_PostTraining_MAS.md`](interview_prep/Bella_Yang_Interview_Prep_LLM_PostTraining_MAS.md) for:
- Dr. Bella Yang's research and the Argonne internship project
- Deep dives on all post-training methods with math
- Multi-agent system design patterns
- Credit assignment (Shapley, AT-GRPO)
- Aurora supercomputer specs and HPC-scale inference
- 50+ interview Q&A pairs
- One-page cheat sheet

---

## 📖 Key Papers

1. **SFT/RLHF:** Ouyang et al., "Training language models to follow instructions" (InstructGPT, 2022)
2. **DPO:** Rafailov et al., "Direct Preference Optimization" (2023)
3. **GRPO / DeepSeek-R1:** Shao et al., "DeepSeekMath" (2024)
4. **LoRA:** Hu et al., "LoRA: Low-Rank Adaptation" (2021)
5. **MAS Debate:** Du et al., "Improving Factuality via Multiagent Debate" (2023)
6. **MoA:** Wang et al., "Mixture-of-Agents" (2024)
7. **Credit Assignment:** Yang et al., "Who Gets the Reward & Who Gets the Blame?" (arXiv:2511.10687)
8. **PRM:** Lightman et al., "Let's Verify Step by Step" (2023)
9. **GSM8K:** Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021)
