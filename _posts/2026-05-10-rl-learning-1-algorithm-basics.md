---
title: '强化学习 1：算法基础'
date: 2026-05-10
permalink: /posts/2026/05/rl-learning-1-algorithm-basics/
tags:
  - Reinforcement Learning
---

# RL 算法原理

## 什么是 RL 问题？

> **强化学习（Reinforcement Learning）** 是研究智能体（Agent）如何在与环境的序列交互中，通过试错学习一个最优行为策略，以最大化长期累积奖励的理论框架。

一句话理解：强化学习就是学习做什么能使 agent 累积收益最大化。
![Pasted image 20260508010646](attachments/Pasted%20image%2020260508010646.png)

---
## 数学建模框架

### 马尔科夫决策过程

RL问题的理想数学化形式是马尔科夫决策过程（MDP) ：

$$\mathcal{M} = \langle \mathcal{S},\ \mathcal{A},\ \mathcal{P},\ \mathcal{R},\ \gamma \rangle$$

| 符号            | 名称     | 含义                                                         |
| ------------- | ------ | ---------------------------------------------------------- |
| $\mathcal{S}$ | 状态空间   | 环境所有可能状态的集合                                                |
| $\mathcal{A}$ | 动作空间   | Agent 所有可选动作的集合                                            |
| $\mathcal{P}$ | 状态转移概率 | $\mathcal{P}(s'\mid s,a) = P(s_{t+1}=s'\mid s_t=s, a_t=a)$ |
| $\mathcal{R}$ | 奖励函数   | $\mathcal{R}(s,a)$，执行动作后的即时反馈                              |
| $\gamma$      | 折扣因子   | $\gamma \in [0,1)$，控制未来奖励的权重                               |

MDP 成立的核心假设：**马尔可夫性（Markov Property）**

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

下一状态只取决于**当前状态与当前动作**，与更早的历史无关。

---
### 轨迹数据

把Agent 与环境一次完整交互的记录定义为一条**轨迹（Trajectory）**，简称 $\tau$ ：

$$\tau = (s_0, a_0, r_0,\ s_1, a_1, r_1,\ \ldots,\ s_T, a_T, r_T)$$

由以下过程生成：

$$s_0 \sim \rho_0(\cdot), \quad a_t \sim \pi_\theta(\cdot|s_t), \quad s_{t+1} \sim P(\cdot|s_t, a_t)$$

策略 $\pi_\theta: \mathcal{S} \times \mathcal{A} \to [0,1]$，即 $\pi_\theta(a|s)$ 为在状态 $s$ 下采取动作 $a$ 的条件概率。

---
### 优化目标

**折扣累积回报**（discounted return）定义为：

$$G_t \triangleq \sum_{k=0}^{\infty} \gamma^k \, r_{t+k}$$

强化学习的目标是找到一个最优的策略$\pi_\theta(a_t|s_t)$ 最大化从初始状态出发的期望折扣累积回报

$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \gamma^t r_t \right] $$

其中轨迹的概率分布为：

$$p_\theta(\tau) = \rho_0(s_0) \prod_{t=0}^{\infty} \pi_\theta(a_t|s_t) \, P(s_{t+1}|s_t, a_t)$$

---
## 核心基础概念

### 1. 价值函数

记作 $V^{\pi}(s_t)$。

含义：从状态$s$出发，沿着策略$\pi$执行期望获得的折扣累积回报。

> 价值函数和策略有什么关系？为啥会带有上标$\pi$?  不同策略计算的价值函数不一样，存在最优价值函数

$$V^\pi(s_t) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t\right]$$

### 2. 状态-动作函数

记作 $Q^\pi(s_t,a_t)$。

含义：在状态$s_t$下采取动作$a_t$之后,继续沿着策略$\pi$执行期望获得的折扣累积回报。

$$Q^\pi(s_t, a_t) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t, a_t\right]$$

价值函数$V^{\pi}(s_t)$是状态-动作函数$Q^\pi(s_t,a_t)$对动作求期望:

$$V^\pi(s_t) = \mathbb{E}_{a_t \sim \pi(\cdot|s_t)}\left[Q^\pi(s_t, a_t)\right]$$

### 3. 优势函数

记作 $A^{\pi}(s_t,a_t)$。

含义：在策略$\pi$下，在状态$s$下采取动作$a_t$相对基线的优势。

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

 
为了估计优势函数引出了：TD 和 MC 和 GAE

### 4.蒙特卡洛估计（MC）

完整的统计整条轨迹的真实回报：

$$\hat{A}_t^{\mathrm{MC}} = G_t - V(s_t) = \sum_{l=0}^{T-t} \gamma^l r_{t+l} - V(s_t)$$

这种估计方法无偏，但是方差大。适用于轨迹较短，且有确定性终止条件的情形。
### 5.时序差分估计（TD)

引入$V^{\pi}$，通过单步自举估计，可以实现对$A_t$的单步近似：

$$\delta_t = r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$$

$$\mathbb{E}[\delta_t] = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)] = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

这种估计方法依赖$V^{\pi}$预估的准确性，如果$V^{\pi}$预估有偏，则单步 $\delta_t$ 偏差大。不过因为只有单步，方差相对较小。
### 5.广义优势估计（GAE)

单步 $\delta_t$ 偏差大,方差小；MC 偏差小，但是方差大。GAE估计在偏差和方差之间做 trade-off。

K 步TD展开估计：

$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t) = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}$$

GAE是对所有k步估计进行指数加权平均（指数衰减权重$\lambda^{k}$)，权重之和为 1.

$$\hat{A}_t^{\text{GAE}} = (1-\lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)}$$

权重 $(1-\lambda)\lambda^{k-1}$ 是一个几何分布，所有权重加和为 1。代入上面k 步 advantage 公式得到：

$$\hat{A}_t^{\text{GAE}} = (1-\lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}$$

现在交换求和顺序。$\delta_{t+l}$ 出现在所有 $k > l$ 的项里，对应的 $\lambda^{k-1}$ 从 $k=l+1$ 开始累加：

$$= (1-\lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \sum_{k=l+1}^{\infty} \lambda^{k-1} = (1-\lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \cdot \frac{\lambda^l}{1-\lambda}$$

$(1-\lambda)$ 和 $\frac{1}{1-\lambda}$ 相消，得到：

$$\boxed{\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}}$$

边界情况分析：

**当 $\lambda = 0$**：

$$\hat{A}_t^{\mathrm{GAE}(\gamma,0)} = \delta_t$$

退化为 TD(0) 单步估计，高偏差、低方差。

**当 $\lambda = 1$**：

$$\hat{A}_t^{\mathrm{GAE}(\gamma,1)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$

退化为蒙特卡洛估计，低偏差、高方差。

---
## RL优化方法

### Policy Gradient 基础

优化目标对 policy 参数求梯度，得到：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, \, a \sim \pi_\theta(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a) \right]$$

由于 $\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = 0$ 对任意基线 $b(s)$ 成立，可以无偏地将 $Q^{\pi_\theta}$ 替换为 $A^{\pi_\theta}$

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a) \right]$$

公式推导如下:

$$\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) \cdot R(\tau) \, d\tau$$

用 **log-trick**（$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau)$）：

$$= \int p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \cdot R(\tau) \, d\tau$$

$$= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log p_\theta(\tau) \cdot R(\tau) \right]$$

展开 $\log p_\theta(\tau)$（环境动态项与 $\theta$ 无关，梯度为零）：

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t)$$

因此策略梯度定理：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]}$$

因为对任意只依赖 $s_t$ 的基线 $b(s_t)$，有：

$$\mathbb{E}_{a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t) \right] = b(s_t) \cdot \nabla_\theta \underbrace{\int \pi_\theta(a|s)\,da}_{=1} = 0$$

所以减掉 $V^\pi(s_t)$ 后梯度期望**完全不变**：

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \underbrace{(G_t - V^\pi(s_t))}_{= A_t} \right]$$

取 $V^\pi(s_t)$ 作为基线是**方差最小化的最优选择**，$A_t$ 的方差远小于 $G_t$。

**核心问题**：上面梯度计算方法在每次更新 $\theta$ 后，必须用新策略 $\pi_\theta$ **重新采样**数据，数据利用率低，采样成本高。

改为从 $\pi_{\theta_\text{old}}$ 采样，通过重要性采样保证无偏性。

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}} \left[ \sum_t \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t \right]$$

其中：

$$\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} \cdot \nabla_\theta \log \pi_\theta(a|s) = \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} \cdot \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)}$$

反过来看，这正是对下面这个目标函数求梯度：

$$\nabla_\theta \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} \cdot A_t \right] = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} \cdot A_t$$

因此，策略梯度等价于对**代理目标（Surrogate Objective）求梯度**：

> **CPI** = Conservative Policy Iteration，这是 TRPO 论文的叫法。注意 $r_t(\theta_\text{old}) = 1$，即在旧策略处，代理目标与真实目标**一阶吻合**。

$$\boxed{L^{\text{CPI}}(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} \cdot A_t \right] = \mathbb{E}_t \left[ r_t(\theta) \cdot A_t \right]}$$

其中定义概率比值：

$$r_t(\theta) \triangleq \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

虽然重要性采样可以保证无偏性，但是方差却会随着$\pi_\theta / \pi_{\theta_\text{old}}$ 偏离 1 而急剧增大，策略更新非常不稳定，容易导致训练崩溃。

### PPO

> 原始论文：[PPO Paper](https://arxiv.org/pdf/1707.06347)

#### CLIP LOSS

针对上面的问题，PPO 的给出的解法是对概率比进行Clip。**核心思路**：直接在目标函数里对 $r_t(\theta)$ 做截断，阻止它偏离 1 太远。定义 Clip 后的代理目标：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \cdot A_t, \;\; \text{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon) \cdot A_t \right) \right]$$

其中 $\varepsilon$ 是超参数，通常取 $0.1$ 或 $0.2$。
#### Value Loss

Critic 网络 $V_\phi(s_t)$ 需要单独训练，目标是让它准确预测状态价值。用 TD 目标或 GAE 构造的回报估计 $\hat{V}_t$ 作为监督信号：

$$\boxed{L^{\text{VF}}(\phi) = \mathbb{E}_t \left[ \left( V_\phi(s_t) - \hat{V}_t \right)^2 \right]}$$


$\hat{V}_t$ 的构造方式通常是：

$$\hat{V}_t = \hat{A}_t + V_{\phi_\text{old}}(s_t)$$

即 GAE 算出来的 Advantage 加上旧 Critic 的预测值，这样 $\hat{V}_t$ 和 $\hat{A}_t$ 在数学上是自洽的。
#### Entropy Bonus

$$\boxed{L^{\text{ENT}}(\theta) = \mathbb{E}_t \left[ H(\pi_\theta(\cdot|s_t)) \right] = -\mathbb{E}_t \left[ \sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) \right]}$$

鼓励策略保持多样性，防止过早收敛到某个次优的确定性策略。没有它，策略很容易在局部最优陷住。

#### 完整的 PPO Loss

三项合并，同时优化：

$$\boxed{L^{\text{PPO}}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 \cdot L^{\text{VF}}(\phi) - c_2 \cdot L^{\text{ENT}}(\theta)}$$

### GRPO

PPO 需要维护一个和策略网络**同等规模**的 Critic 网络来估计 $V(s)$ 。对于 70B 的模型，这意味着：

- 显存加倍
- 训练成本加倍
- Critic 本身也可能估得不准（引入额外偏差）

**GRPO 的核心思想**：干掉 Critic，用**组内相对比较**代替价值函数！
