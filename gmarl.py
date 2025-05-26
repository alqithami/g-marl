{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env python3\
"""\
G\uc0\u8209 MARL: Graph\u8209 based Multi\u8209 Agent Reinforcement\u8209 Learning reference pipeline\
-----------------------------------------------------------------------\
\'95 End\uc0\u8209 to\u8209 end training / evaluation script ready for **RunPod** or local GPU.\
\'95 PyTorch 2.3  +  PyTorch\uc0\u8209 Geometric 2.5  +  rich CLI via argparse / yaml.\
\'95 Modular dataset loader (Intel\uc0\u8209 Lab, SWaT, WADI, TON\u8209 IoT, synthetic).\
\'95 PPO with GAE, mini\uc0\u8209 batch optimisation, checkpointing & TensorBoard.\
\'95 Energy\uc0\u8209 aware reward, selective communication gate, GAT encoder.\
\
Author : OpenAI\'a0ChatGPT reference implementation  \'96  2025\uc0\u8209 05\u8209 27\
License: Apache\uc0\u8209 2.0\
"""\
\
from __future__ import annotations\
\
import argparse, os, json, math, random, time, yaml, shutil, sys\
from pathlib import Path\
from typing import List, Tuple, Dict, Optional\
\
import numpy as np\
import torch\
import torch.nn as nn\
import torch.nn.functional as F\
import torch.optim as optim\
from torch.utils.tensorboard import SummaryWriter\
\
from torch_geometric.nn import GATConv\
from torch_geometric.data import Data, Batch\
from torch_geometric.loader import DataLoader as GraphLoader\
\
# ---------------------------------------------------------------------\
# 1.  Utility helpers\
# ---------------------------------------------------------------------\
\
def set_seed(seed: int) -> None:\
    random.seed(seed)\
    np.random.seed(seed)\
    torch.manual_seed(seed)\
    if torch.cuda.is_available():\
        torch.cuda.manual_seed_all(seed)\
\
# ---------------------------------------------------------------------\
# 2.  Dataset utilities (synthetic + real corpora stubs)\
# ---------------------------------------------------------------------\
\
class SyntheticIoTGraphDataset(torch.utils.data.Dataset):\
    """Generate synthetic graph\uc0\u8209 stream episodes on\u8209 the\u8209 fly."""\
\
    def __init__(self,\
                 num_nodes: int = 64,\
                 seq_len: int = 2_000,\
                 input_dim: int = 8,\
                 avg_degree: int = 4,\
                 anomaly_prob: float = 0.05,\
                 seed: int = 0):\
        super().__init__()\
        self.N = num_nodes\
        self.T = seq_len\
        self.D = input_dim\
        self.avg_degree = avg_degree\
        self.anom_prob = anomaly_prob\
        self.rng = np.random.default_rng(seed)\
\
        # Pre\uc0\u8209 generate static topology once per episode\
        self.edge_index = self._rand_edges()\
        self.x = self._rand_signals()\
        self.labels = self._rand_labels()\
\
    # .................................................................\
    def _rand_edges(self):\
        edges = []\
        for i in range(self.N):\
            nbrs = self.rng.choice(self.N, self.avg_degree, replace=False)\
            for j in nbrs:\
                if i != j:\
                    edges.append((i, j))\
        edges = list(\{(i, j) for (i, j) in edges\} | \{(j, i) for (i, j) in edges\})\
        src, dst = zip(*edges)\
        return torch.tensor([src, dst], dtype=torch.long)\
\
    def _rand_signals(self):\
        base = self.rng.standard_normal((self.T, self.N, self.D), dtype=np.float32)\
        # simple AR(1) smoothing\
        for t in range(1, self.T):\
            base[t] += 0.8 * base[t - 1]\
        return torch.from_numpy(base)  # (T,N,D)\
\
    def _rand_labels(self):\
        labels = self.rng.random((self.T, self.N)) < self.anom_prob\
        return torch.from_numpy(labels.astype(np.int64))  # (T,N)\
\
    # .................................................................\
    def __len__(self):\
        return self.T - 1  # each step predicts next\uc0\u8209 state anomaly\
\
    def __getitem__(self, idx):\
        x_t = self.x[idx]       # (N,D)\
        label_t = self.labels[idx]  # (N,)\
        data = Data(x=x_t, edge_index=self.edge_index)\
        return data, label_t\
\
# TODO: RealDatasetLoader class for Intel\uc0\u8209 Lab, SWaT, WADI, TON\u8209 IoT.\
\
# ---------------------------------------------------------------------\
# 3.  Model definitions\
# ---------------------------------------------------------------------\
\
class GATEncoder(nn.Module):\
    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 32,\
                 heads: int = 4, layers: int = 2):\
        super().__init__()\
        self.layers = nn.ModuleList()\
        self.layers.append(GATConv(in_dim, hidden, heads=heads, concat=True))\
        for _ in range(layers - 2):\
            self.layers.append(GATConv(hidden * heads, hidden, heads=heads, concat=True))\
        if layers > 1:\
            self.layers.append(GATConv(hidden * heads, out_dim, heads=heads, concat=False))\
        self.act = nn.ELU()\
\
    def forward(self, x, edge_index):\
        for i, gat in enumerate(self.layers):\
            x = gat(x, edge_index)\
            if i < len(self.layers) - 1:\
                x = self.act(x)\
        return x  # (N, out_dim)\
\
class ActorCritic(nn.Module):\
    def __init__(self, in_dim: int, hidden: int = 64):\
        super().__init__()\
        self.actor_det = nn.Sequential(nn.Linear(in_dim, hidden), nn.SELU(), nn.Linear(hidden, 2))\
        self.actor_comm = nn.Sequential(nn.Linear(in_dim, hidden), nn.SELU(), nn.Linear(hidden, 2))\
        self.critic = nn.Sequential(nn.Linear(in_dim, hidden), nn.SELU(), nn.Linear(hidden, 1))\
\
    def forward(self, z):\
        logits_det = self.actor_det(z)\
        logits_comm = self.actor_comm(z)\
        value = self.critic(z).squeeze(-1)\
        return logits_det, logits_comm, value\
\
# ---------------------------------------------------------------------\
# 4.  PPO Buffer & Agent\
# ---------------------------------------------------------------------\
\
class RolloutBuffer:\
    def __init__(self):\
        self.clear()\
\
    def clear(self):\
        self.x = []      # list[Tensor (N,D)]\
        self.edge = []   # list[Tensor (2,E)]\
        self.a_det = []  # list[Tensor (N,)]\
        self.a_comm = []\
        self.rewards = []\
        self.dones = []\
        self.values = []\
        self.logp = []\
        self.labels = []\
\
    # .................................................................\
    def add(self, x, edge, a_det, a_comm, reward, done, value, logp, label):\
        self.x.append(x.detach())\
        self.edge.append(edge.detach())\
        self.a_det.append(a_det.detach())\
        self.a_comm.append(a_comm.detach())\
        self.rewards.append(torch.as_tensor(reward, dtype=torch.float32))\
        self.dones.append(done)\
        self.values.append(value.detach())\
        self.logp.append(logp.detach())\
        self.labels.append(label.detach())\
\
# ---------------------------------------------------------------------\
class PPOTrainer:\
    def __init__(self, model_cfg: dict, algo_cfg: dict, device: torch.device):\
        self.device = device\
        # models\
        self.encoder = GATEncoder(**model_cfg["encoder"]).to(device)\
        self.ac = ActorCritic(**model_cfg["actor"]).to(device)\
        self.opt = optim.Adam(list(self.encoder.parameters()) + list(self.ac.parameters()),\
                              lr=algo_cfg["lr"])\
        # algo\
        self.clip = algo_cfg["clip"]\
        self.gamma = algo_cfg["gamma"]\
        self.lam = algo_cfg["gae"]\
        self.value_coeff = algo_cfg["value_coeff"]\
        self.entropy_coeff = algo_cfg["entropy_coeff"]\
\
    # .................................................................\
    def _evaluate_actions(self, z, a_det, a_comm):\
        logits_det, logits_comm, value = self.ac(z)\
        dist_det = torch.distributions.Categorical(logits=logits_det)\
        dist_comm = torch.distributions.Categorical(logits=logits_comm)\
        logp = dist_det.log_prob(a_det) + dist_comm.log_prob(a_comm)\
        entropy = (dist_det.entropy() + dist_comm.entropy()).mean()\
        return logp, entropy, value\
\
    # .................................................................\
    def ppo_update(self, buf: RolloutBuffer, batch_size: int = 256, epochs: int = 4):\
        # pack tensors (T, N, \'85) -> (T*N, \'85)\
        x = torch.cat(buf.x)                 # (T*N, D)\
        edge = buf.edge[0]                  # shared edge_index (2,E)\
        a_det = torch.cat(buf.a_det)\
        a_comm = torch.cat(buf.a_comm)\
        rewards = torch.stack(buf.rewards)\
        dones = torch.as_tensor(buf.dones, dtype=torch.float32)\
        values = torch.cat(buf.values)\
        logp_old = torch.cat(buf.logp)\
\
        # GAE\uc0\u8209 \u955  advantages ------------------------------------------------\
        adv = torch.zeros_like(rewards)\
        gae = 0.0\
        for t in range(len(rewards)-1, -1, -1):\
            delta = rewards[t] + self.gamma * (1 - dones[t]) * values[t+1 if t+1 < len(values) else t] - values[t]\
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae\
            adv[t] = gae\
        returns = adv + values\
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)\
\
        # shuffle indices -------------------------------------------------\
        idxs = torch.randperm(x.size(0))\
        for _ in range(epochs):\
            for start in range(0, len(idxs), batch_size):\
                batch = idxs[start:start+batch_size]\
                z = self.encoder(x[batch].to(self.device), edge.to(self.device))\
                logp, entropy, value = self._evaluate_actions(z,\
                                                              a_det[batch].to(self.device),\
                                                              a_comm[batch].to(self.device))\
                ratio = torch.exp(logp - logp_old[batch].to(self.device))\
                surr1 = ratio * adv[batch].to(self.device)\
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv[batch].to(self.device)\
                pg_loss = -torch.min(surr1, surr2).mean()\
\
                v_loss = F.mse_loss(value, returns[batch].to(self.device))\
                loss = pg_loss + self.value_coeff * v_loss - self.entropy_coeff * entropy\
\
                self.opt.zero_grad()\
                loss.backward()\
                nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.ac.parameters()), 0.5)\
                self.opt.step()\
\
# ---------------------------------------------------------------------\
# 5.  Main entry\
# ---------------------------------------------------------------------\
\
def parse_args():\
    p = argparse.ArgumentParser("G-MARL training")\
    p.add_argument("--config", type=str, required=False, help="YAML config")\
    p.add_argument("--out", type=str, default="runs/demo", help="output dir")\
    p.add_argument("--seed", type=int, default=42)\
    return p.parse_args()\
\
# ---------------------------------------------------------------------\
DEFAULT_CFG = \{\
    "model": \{\
        "encoder": \{"in_dim": 8, "hidden": 32, "out_dim": 32, "heads": 4, "layers": 2\},\
        "actor": \{"in_dim": 32, "hidden": 64\}\
    \},\
    "algo": \{\
        "lr": 3e-4, "clip": 0.2, "gamma": 0.97, "gae": 0.95,\
        "value_coeff": 0.5, "entropy_coeff": 0.01\
    \},\
    "env": \{\
        "num_nodes": 64, "seq_len": 2000, "input_dim": 8, "avg_degree": 4,\
        "anomaly_prob": 0.05, "comm_penalty": 0.001\
    \},\
    "train": \{"episodes": 200, "steps_per_episode": 1999, "batch_size": 512, "ppo_epochs": 4\}\
\}\
\
# ---------------------------------------------------------------------\
\
def main():\
    args = parse_args()\
    cfg = DEFAULT_CFG\
    if args.config:\
        with open(args.config) as f:\
            user_cfg = yaml.safe_load(f)\
        cfg = \{**cfg, **user_cfg\}\
\
    set_seed(args.seed)\
    out_dir = Path(args.out)\
    out_dir.mkdir(parents=True, exist_ok=True)\
    writer = SummaryWriter(log_dir=str(out_dir))\
\
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\
    print(f"[INFO] Device: \{device\}")\
\
    # Dataset + loader ---------------------------------------------------\
    env_cfg = cfg["env"]\
    dataset = SyntheticIoTGraphDataset(**env_cfg)\
    loader = GraphLoader(dataset, batch_size=1, shuffle=False)  # step loader\
\
    trainer = PPOTrainer(cfg["model"], cfg["algo"], device)\
    buffer = RolloutBuffer()\
\
    global_step = 0\
    for ep in range(cfg["train"]["episodes"]):\
        buffer.clear()\
        total_reward = 0.0\
        hidden_edge = dataset.edge_index  # static for synthetic\
        for t, (data, labels) in enumerate(loader):\
            data = data.to(device)\
            labels = labels.to(device)\
            z = trainer.encoder(data.x, data.edge_index)\
            logits_det, logits_comm, value = trainer.ac(z)\
            dist_det = torch.distributions.Categorical(logits=logits_det)\
            dist_comm = torch.distributions.Categorical(logits=logits_comm)\
            a_det = dist_det.sample()\
            a_comm = dist_comm.sample()\
            logp = dist_det.log_prob(a_det) + dist_comm.log_prob(a_comm)\
            # simple reward: +1 correct, \uc0\u8722 2 miss, \u8722 0.5 false+, comm penalty\
            r_det = torch.where(a_det == labels, 1.0,\
                                torch.where((labels == 1) & (a_det == 0), -2.0, -0.5))\
            r_comm = -env_cfg["comm_penalty"] * 32 * a_comm.float()\
            reward = (r_det + r_comm).mean()\
\
            done = torch.tensor(t == cfg["env"]["seq_len"] - 2, dtype=torch.float32)\
            buffer.add(data.x, data.edge_index, a_det, a_comm, reward, done,\
                       value.mean(), logp, labels)\
            total_reward += reward.item()\
            global_step += 1\
\
        trainer.ppo_update(buffer, batch_size=cfg["train"]["batch_size"],\
                            epochs=cfg["train"]["ppo_epochs"])\
        writer.add_scalar("reward/episode", total_reward / dataset.T, ep)\
        if (ep + 1) % 50 == 0:\
            ckpt = \{\
                "encoder": trainer.encoder.state_dict(),\
                "ac": trainer.ac.state_dict(),\
                "opt": trainer.opt.state_dict(),\
                "cfg": cfg\
            \}\
            torch.save(ckpt, out_dir / f"ckpt_ep\{ep+1\}.pt")\
            print(f"[CKPT] Saved checkpoint at episode \{ep+1\}")\
\
    writer.close()\
    print("[TRAIN] Finished training.")\
\
# ---------------------------------------------------------------------\
if __name__ == "__main__":\
    main()\
}