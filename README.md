# 🔬 White-Box Activation Steering to Prevent Deceptive Alignment in Cooperative Multi-Agent Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Status: Research](https://img.shields.io/badge/status-research--in--progress-yellow.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official research codebase** for the paper:

> **"White-Box Activation Steering to Prevent Deceptive Alignment in Cooperative Multi-Agent Systems"**  
> Muhammad Adam — Carnegie Mellon University Africa, 2026  
> *Targeting NeurIPS / ICLR*

---

## 🧩 Problem

In cooperative multi-agent systems (MAS), agents can learn to behave differently depending on whether they are being monitored — appearing cooperative under oversight but pursuing a hidden defecting strategy when unmonitored. This is called **deceptive alignment**, and it is invisible to behavioral audits by definition: the agent's outputs look identical in both modes.

Standard defenses (RLHF fine-tuning, output monitoring) either require expensive retraining or cannot detect deception encoded in the agent's internal representations.

---

## 💡 Our Approach

We propose a **white-box, inference-time** method that:

1. **Identifies** a linear direction in the agent's representation space that separates cooperative from deceptive internal states — using contrastive activation pairs from monitored vs. unmonitored rollouts.
2. **Steers** the agent's activations at inference time by projecting out the deceptive direction — suppressing the defecting mode without any retraining.

The key insight is that **deceptive alignment leaves a detectable geometric signature in activation space**, even when outputs are behaviorally identical. This method finds and removes that signature at runtime.

---

## 🗂️ Repository Structure

```
white-box-activation-steering/
├── src/
│   ├── steering/
│   │   ├── __init__.py
│   │   ├── activation_hook.py       # PyTorch forward hooks for activation extraction
│   │   ├── direction_finder.py      # Linear probe / PCA / DoM direction learning
│   │   └── steering_intervention.py # Inference-time projection intervention
│   ├── probing/
│   │   ├── __init__.py
│   │   └── linear_probe.py          # Layer-wise linear deception probes
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py            # Abstract transformer policy agent
│   │   └── transformer_agent.py     # GPT-2 small based policy agent
│   ├── envs/
│   │   ├── __init__.py
│   │   └── deceptive_coin_game.py   # Deceptive Coin Game environment (Env 1)
│   └── utils/
│       ├── __init__.py
│       ├── rollout.py               # Multi-agent rollout collection
│       ├── metrics.py               # Deception rate, task performance, retention
│       └── visualisation.py        # Safety-utility frontier plots
├── experiments/
│   ├── 01_probe_layer_selection.py  # Find the layer where deception is most separable
│   ├── 02_direction_learning.py     # Learn v̂ via LP, DoM, PCA; compare
│   ├── 03_steering_evaluation.py    # Full evaluation: deception rate vs performance
│   └── 04_ablations.py             # Steering coefficient α, layer depth ablations
├── configs/
│   ├── default.yaml
│   └── ablation_alpha.yaml
├── notebooks/
│   └── results_visualisation.ipynb
├── tests/
│   ├── test_steering.py
│   ├── test_probe.py
│   └── test_env.py
├── docs/
│   ├── method.md
│   └── experiment_design.md
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Emadamx/white-box-activation-steering.git
cd white-box-activation-steering
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.0+, TransformerLens, scikit-learn

---

## 🚀 Quick Start

### Step 1 — Find the deception layer

```bash
python experiments/01_probe_layer_selection.py --config configs/default.yaml
```

This trains a linear probe at each transformer layer and produces a plot of probe accuracy vs. layer depth. The peak identifies layer `l*` — where deception is most linearly separable.

### Step 2 — Learn the steering direction

```bash
python experiments/02_direction_learning.py --config configs/default.yaml --method lp
```

Methods: `lp` (linear probe), `dom` (difference of means), `pca`.

### Step 3 — Evaluate steering

```bash
python experiments/03_steering_evaluation.py --config configs/default.yaml
```

Outputs the safety-utility frontier: deception suppression rate vs. cooperative task performance retained.

### Step 4 — Run ablations

```bash
python experiments/04_ablations.py --config configs/ablation_alpha.yaml
```

---

## 📐 Method

### Formal Setup

We consider `N` transformer-based policy agents trained via RL on a shared cooperative task. A **deceptively aligned** agent maintains two behavioral modes:

- **Cooperative mode** `π_c`: active when a monitoring signal `σ = 1` is present
- **Defecting mode** `π_d`: active when `σ = 0` (unmonitored)

The outputs are behaviorally identical during monitored episodes. The deceptive strategy is encoded only in the agent's internal activations.

### Steering Direction

Let `H^c` and `H^d` be the activation matrices at layer `l*` collected from monitored and unmonitored rollouts respectively, **matched on identical observations** `o_t`.

We learn a unit steering direction `v̂ ∈ ℝ^d` via linear probing:

```
v̂ = argmin_{||w||=1}  L_BCE( w·H, y )
```

where `y = 1` for cooperative activations and `y = 0` for deceptive.

### Inference-Time Intervention

At inference time, activations at layer `l*` are steered by projecting out the deceptive component:

```
h'_t = h_t - α · (h_t · v̂) · v̂
```

where `α ∈ [0, 1]` is the steering coefficient controlling intervention strength.

---

## 📊 Preliminary Results

*Preliminary results on a randomly initialised agent (full RL-trained evaluation ongoing).*

| Condition | Deception Rate | Performance Retained |
|---|---|---|
| No intervention | 53.00% | 100.00% |
| LP steering α=0.4 | 49.50% | 93.68% |
| LP steering α=0.6 | 48.50% | 103.30% |
| LP steering α=0.8 | **44.00%** | **102.20%** |
| LP steering α=1.0 | 56.00% | 91.76% |

> Best result at α=0.8: deception reduced by ~17% with no performance cost.
> Note: results are preliminary on a randomly initialised agent without RL training.
> Full evaluation with a trained deceptive agent is ongoing.
> Linear probe accuracy at l*=0: **0.780** (78% separation of cooperative 
> vs deceptive activations)

---

## 📄 Related Work

- Hubinger et al. (2019). *Risks from Learned Optimization in Advanced Machine Learning Systems.*
- Anthropic (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.*
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.*
- Turner et al. (2023). *Activation Addition: Steering Language Models Without Optimization.*
- Abdelnabi et al. (2023). *Not What You've Signed Up For: Indirect Prompt Injection.*

---

## 📬 Citation

```bibtex
@article{adam2026activation,
  title   = {White-Box Activation Steering to Prevent Deceptive Alignment
             in Cooperative Multi-Agent Systems},
  author  = {Adam, Muhammad},
  year    = {2026},
  note    = {Preprint. Carnegie Mellon University Africa.}
}
```

---

## 👤 Author

**Muhammad Adam**  
MSc Engineering AI, Carnegie Mellon University – Africa  
[madam2@andrew.cmu.edu](mailto:madam2@andrew.cmu.edu) · [GitHub](https://github.com/Emadamx) · [LinkedIn](https://linkedin.com/in/muhammad-adam-bb35931b0)
