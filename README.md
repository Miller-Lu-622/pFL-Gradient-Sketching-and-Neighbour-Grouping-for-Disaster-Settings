# Personalized Federated Learning with Gradient Sketching and Neighbour Grouping for Disaster Settings

This repository is my research fork built on top of [PFLlib](https://github.com/TsingZ0/PFLlib). It is being adapted to study a disaster-oriented personalized federated learning framework that supports:

- strong natural Non-IID data across disaster events and regions
- repeated regrouping under drift and changing participation
- lightweight neighbour discovery from compact update signatures
- personalized coordination beyond a single shared global model

## Research Motivation

Disaster settings combine several federated learning difficulties at once:

- **heterogeneity**: data differs across events, regions, sensors, and recovery stages
- **drift**: conditions evolve over time
- **instability**: participation may be bursty, correlated, or disrupted by communication failures

Many existing FL methods address only part of this problem. General FL methods often focus on a shared global model, while clustering and relation-based methods can be too static or too expensive to refresh frequently. This project explores a more lightweight and dynamic alternative.

## Core Idea

The proposed framework uses **compact update signatures** derived from client updates to support efficient neighbour grouping and regrouping.

At a high level:

1. clients receive the current model and perform local training
2. clients send updates to the server
3. the server converts updates into compact signatures
4. approximate neighbour search identifies similar clients
5. the server combines global aggregation with neighbour-based personalized coordination
6. neighbour relations are refreshed periodically or when drift is detected

The goal is to preserve useful collaboration structure without relying on expensive full pairwise similarity computation.

## Research Questions

**RQ1.** Can compact update signatures support repeated regrouping and server-side personalized coordination under the combined effects of heterogeneity, drift, and instability?

**RQ2.** How effective is the framework on naturally heterogeneous disaster data such as xBD, and how do its benefits compare with supplementary evaluation on conventional FL benchmarks?

## Planned Methodology

This project is centered on a simulation-based evaluation pipeline.

### Primary evaluation setting
- **xBD / xView2-style disaster damage assessment** as the main benchmark
- client splits designed to reflect natural heterogeneity across events, locations, and conditions
- disaster-style instability simulated through bursty dropout, churn, and changing availability

### Supplementary evaluation
- conventional FL benchmarks such as **CIFAR-10** and **CIFAR-100**
- comparisons against standard FL and personalized FL baselines

### Key measurements
- task performance: accuracy, F1, precision, recall, AUC, loss
- personalization behaviour: per-client performance and variance
- efficiency: communication overhead, signature construction cost, neighbour retrieval cost, regrouping latency
- robustness: behaviour under drift and unstable participation

## Current Repository Role

This repository currently serves as the implementation base for the proposed work.

Planned development directions include:

- implementing compact gradient/update signatures
- adding neighbour retrieval and client-specific collaboration sets
- supporting repeated regrouping strategies
- simulating disaster-induced communication instability
- benchmarking against existing FL / pFL baselines

## Repository Structure

```text
PFLlib/
├── dataset/                  # dataset generation scripts and shared utilities
├── system/
│   ├── flcore/              # clients, servers, optimizers, training logic
│   ├── models/              # model definitions
│   └── utils/               # utilities
├── docs/                    # original framework assets/docs
├── env_cuda_latest.yaml     # environment setup
└── prepare.sh               # setup helper
```

## Dataset Tracking Policy

To keep the repository lightweight:

- raw/generated dataset subdirectories under `dataset/` are not intended to be versioned by default
- `dataset/utils/` is retained for reusable processing utilities
- large dataset artifacts should be stored locally or through dedicated data storage, not committed into Git whenever possible

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Miller-Lu-622/pFL-Gradient-Sketching-and-Neighbour-Grouping-for-Disaster-Settings.git
cd pFL-Gradient-Sketching-and-Neighbour-Grouping-for-Disaster-Settings
```

### 2. Create the environment
```bash
conda env create -f env_cuda_latest.yaml
conda activate pfllib
```

If needed, adjust the PyTorch/CUDA version to match your machine.

### 3. Generate or prepare datasets
Use the scripts in `dataset/` as the starting point for benchmark preparation. The disaster-oriented pipeline will be added incrementally as the project implementation progresses.

### 4. Run baseline experiments
The current codebase inherits the original PFLlib training entry points under `system/`. For example:

```bash
cd system
python main.py -data MNIST -m CNN -algo FedAvg -gr 2000 -did 0
```

## Project Status

This is an active research repository. The current stage focuses on:

- turning the proposal into an executable implementation plan
- adapting the inherited PFLlib structure to the new disaster-oriented setting
- preparing a clean baseline for later algorithm development and experiments

## Acknowledgement

This repository is based on the excellent open-source project **PFLlib**:

- Jianqing Zhang et al., *PFLlib: A Beginner-Friendly and Comprehensive Personalized Federated Learning Library and Benchmark*, JMLR 2025.
- Original repository: <https://github.com/TsingZ0/PFLlib>

The original codebase provides the baseline framework and benchmark structure on which this research fork is built.

## Proposal Reference

This repository is aligned with the proposal:

**“Personalized Federated Learning with Gradient Sketching and Neighbour Grouping for Disaster Settings”**

Current framing:

- compact update signatures for lightweight coordination
- repeated neighbour regrouping under drift
- server-side personalized coordination
- xBD-centered disaster evaluation with supplementary conventional benchmarks
