# SwarmRL

SwarmRL is a Multi-Agent Reinforcement Learning (MARL) project focused on training a swarm of autonomous drones (up to 50 agents) to cooperatively explore and cover a 3D disaster zone. It utilizes the MAPPO (Multi-Agent Proximal Policy Optimization) algorithm for decentralized execution with a centralized critic during training.

The project features a full-stack implementation, including a custom PettingZoo 3D environment, Ray RLlib integration, a FastAPI backend for inference streaming, and a React + Three.js frontend for real-time 3D visualization.

## 🏗️ Architecture

```text
                    SwarmRL

                  React Frontend
                       |
                    Three.js
                       |
                   WebSocket
                       |
                   FastAPI
                       |
              RLlib Inference Engine
                       |
                    MAPPO
                  /       \
              Actor       Critic
            PyTorch      PyTorch
                 \        /
                PettingZoo
                     |
             3D Swarm Environment
          /          |           \
     Drones       Obstacles    Coverage Map
```

## 🚀 Tech Stack

- **Reinforcement Learning:** PettingZoo, Gymnasium, Ray RLlib, PyTorch (MAPPO)
- **Backend:** Python, FastAPI, WebSockets
- **Frontend:** React, Three.js, Node.js
- **Metrics/Logging:** TensorBoard / Custom Analytics Dashboard

## 🎯 Core MVP Metrics Target

For the final MVP, we are targeting:
- **Number of drones:** 50
- **Area coverage:** >85–90%
- **Collision rate:** <5%
- **Training episodes:** 1,000+
- **Visualization FPS:** ~30–60 FPS
- **RL Algorithm:** MAPPO (Decentralized Actors, Centralized Critic)

## 📅 30-Day Implementation Roadmap

### Week 1: Core Environment (PettingZoo)
- **Day 1-2:** Project setup (Python/Node environments) and initial 3D world dimension/agent design.
- **Day 3-5:** Define observation/action spaces and build PettingZoo env (`reset()`, `step()`, multi-agent handling).
- **Day 6-7:** Implement collision logic (drone-drone, drone-obstacle) and test environment loop.

### Week 2: Reward System & RL Baseline
- **Day 8-10:** Implement voxel/grid coverage tracking and engineer/validate reward functions.
- **Day 11-14:** Register environment with Ray RLlib, train a baseline PPO model, and establish tracking metrics.

### Week 3: MAPPO & Multi-Agent Coordination
- **Day 15-17:** Architect MAPPO (CTDE) and build PyTorch Actor/Critic networks.
- **Day 18-21:** Train initial agents, tune hyperparameters, and scale simulation to 25-50 agents.

### Week 4: 3D Visualization & Product Integration
- **Day 22-24:** Build React dashboard layout, Three.js canvas, and FastAPI WebSocket server.
- **Day 25-28:** Animate drone movement via live coordinate stream, visualize sensor coverage, and build analytics UI.
- **Day 29-30:** End-to-end system testing, performance optimization, and final demo preparation.

## 🛠️ Getting Started

*(Instructions will be updated as the implementation progresses through the 30-day roadmap)*

### Prerequisites
- Python 3.9+
- Node.js 18+

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SwarmRL.git
   cd SwarmRL
   ```
2. *(More installation steps to come...)*

## 🤝 Team Roles & Responsibilities

- **RL Engineer:** PettingZoo, RLlib, MAPPO, reward design
- **ML Engineer:** PyTorch actor-critic networks, training experiments
- **Backend Engineer:** FastAPI, WebSocket, inference service
- **Frontend Engineer:** React, Three.js, dashboard
- **Product Manager:** Requirements, milestones, metrics, testing and demo
- **QA/Testing:** Collision tests, scalability, UI and simulation validation
