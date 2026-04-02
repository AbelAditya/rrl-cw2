// ─── Page & Typography Setup ───────────────────────────────────────────────────
#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
  number-align: right,
)

#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.72em, spacing: 1.2em)
#set heading(numbering: none)

#show math.equation: set text(font: "New Computer Modern Math")

#let pd(top, bottom) = $frac(partial #top, partial #bottom)$
#let ddot(s) = $dot(dot(#s))$

// ─── Accent colour ─────────────────────────────────────────────────────────────
#let accent = rgb("#1a5276")   // deep navy-blue

// ─── Section heading style ─────────────────────────────────────────────────────
#show heading.where(level: 1): it => {
  v(1.4em)
  block(
    fill: accent,
    inset: (x: 10pt, y: 6pt),
    radius: 3pt,
    width: 100%,
  )[
    #text(fill: white, weight: "bold", size: 12pt)[#it.body]
  ]
  v(0.6em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(fill: accent, weight: "bold", size: 11pt)[#it.body]
  v(0.4em)
}

// ─── Figure helper: cap height so figures never dominate a page ────────────────
#show figure: it => {
  block(width: 100%)[
    #it.body
    #if it.caption != none {
      v(0.3em)
      align(center)[
        #text(size: 9.5pt, style: "italic")[
          #it.supplement #context it.counter.display(). #it.caption.body
        ]
      ]
    }
  ]
  v(0.5em)
}

// ─── Header (no names – two-column layout) ─────────────────────────────────────
#set page(header: [
  #set text(size: 9pt)
  #grid(
    columns: (1fr, 1fr),
    align: (left, right),
    [Robot and Reinforcement Learning], [*Coursework 2: PPO vs SAC*],
  )
  #line(length: 100%, stroke: 0.4pt + luma(160))
])

// ─── Algorithm Introduction (10 marks) ───────────────────────────────────── 

= Introduction to PPO and SAC

Proximal Policy Optimization @PPO is an on policy gradient method that optimizes a clipped surrogate objective function. It was a direct improvement over the previously proposed Trust Region Policy Optimization's (TRPO) pitfalls due to the involvement of KL Divergence as a constraint over optimizing the surrogate objective function. PPO proposes the use of a clip function to prevent large policy updates instead of KL divergence penalty or constraint. "PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample
complexity, simplicity, and wall-time"@PPO.

Soft Actor Critic (SAC) is an off-policy actor-critic RL algorithm based on the maximum entropy reinforcement leanring framework. The soft actor-critic algorithm incorporates three key ingredients: an actor-critic architecture with separate policy and value function networks, an off-policy formulation that enables reuse of previously collected data for efficiency, and entropy maximization to enable stability and exploration @SAC. This was an improvement over previously proposed RL methods in terms of sample efficiency as it is an off-policy method and stability to convergence and sensitivity to hyperparameters which was notoriously tough to achieve with off-policy model free methods. 

// TODO: may choose to include relevant equations

// ─── Relevance to Robotics (3 marks)─────────────────────────────────────

= Relevance to Robotics

Robotics pose specific set of challenges originating from real world setting that make using classical RL techniques infeasible. Methods like PPO and SAC mitigate these challenges quite well making them a good choice for robotics tasks. For instance, PPO and SAC can support large sizes of state space and action space and can generalize well to scenarios that these models have not encountered enough while training. These methods also work quite well with continuous action and state spaces making them an ideal choice for robotics. PPO and SAC both show fair amount of resilience towards noisy inputs through clipping policy update preventing large policy shifts in PPO and entropy regularized objective reducing over reliance on any single observation in SAC.



// ─── Environment Description (6 marks) ─────────────────────────────────────

= Environment Description

We chose to train in the Ant-v4 and HalfCheetah-v4 environments in MuJoCo gymnasium @mujoco. These were chosen because the represent different complexities of the same locomotion goal. This also makes them directly comparable for benchmarking PPO and SAC. 

The *HalfCheetah* is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws). The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward. The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins).

The *Ant* is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts. The goal is to coordinate the four legs to move in the forward (right) direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges).

// ─── Hyperparameter Analysis (12 marks) ─────────────────────────────────────

= PPO Hyperparameter Tuning

We chose to tune clip coefficient ($epsilon$) and $lambda$ (in GAE) as these affect the algorithm's working to its core. 

The clip coefficient controls the policy update size, this effectively dictates how quickly or slowly the policy is moving towards the optimal. Having the correct step size is important because if the step size is too small then the policy would take a lot of timesteps to reach the optimal or else if the step size is to big the policy would end up overshooting the optimal, both cases result in the formulation of a poor policy. 

Tuning lambda controls the bias-variance trade off in estimating the advantage term. As lambda moves from 0 to 1 it cause the shift in advantage estimation being carried out as TD(0) at $lambda = 0$ (high bias) and as Monte Carlo method at $lambda = 1$ (high variance). Temporal Difference introduces bias as it bootstraps value estimates from the immediate next step. Monte Carlo methods are unbiased but inherently show high variance, this originates from variance being accumulated over the length of the episodes. Hence, requires the agent to be run on many episodes for the value estimates to reliably converge. 

The following values of $epsilon$ were chosen: $epsilon in {0.1,0.2,0.3}$. We have explored values around the published default @PPO ($epsilon = 0.2$) testing change in agent behaviour under a transition from a strict ($epsilon = 0.1$) to a permissive ($epsilon = 0.3$) policy update constraint. For $lambda$ the following values were chosen: $lambda in {0.9, 0.95, 1.00}$. Similar strategy of exploring around the published defualt @PPO is followed, testing change in agent behaviour as advatage estimation moves from pure monte carlo at $lambda = 1$ to sligthy towards TD(0) at $lambda = 0.90$.  

#figure(
  table(
    columns: (auto, auto, auto, auto,),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header(
      [], 
      [*$lambda = 0.90$*], 
      [*$lambda = 0.95$*], 
      [*$lambda = 1.00$*]
    ),
    [*$epsilon = 0.1$*], [1471.80], [1378.69], [1657.81],
    [*$epsilon = 0.2$*], [*4449.39*], [2706.83], [2397.13],
    [*$epsilon = 0.3$*], [1131.88], [*4568.06*], [150.90],
  ),
  caption: [Average episodic return in HalfCheetah-v4 environment with different $gamma$ and $tau$ configurations averaged over 10 episodes using PPO algorithm],
) <tab1>

#figure(
  table(
    columns: (auto, auto, auto, auto,),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header(
      [], 
      [*$lambda = 0.90$*], 
      [*$lambda = 0.95$*], 
      [*$lambda = 1.00$*]
    ),
    [*$epsilon = 0.1$*], [2498.17], [1419.31], [11.57],
    [*$epsilon = 0.2$*], [2277.72], [3049.39], [107.12],
    [*$epsilon = 0.3$*], [3204.60], [507.76], [72.97],
  ),
  caption: [Average episodic return in Ant-v4 environment with different $gamma$ and $tau$ configurations averaged over 10 episodes using PPO algorithm],
) <tab2>

// - Two sets of $epsilon$ & $lambda$ values have shown good performance with close episodic returns
//   - $[epsilon = 0.2, lambda = 0.9]$ : Average Episodic Return = 4449.39
//   -  $[epsilon = 0.3, lambda = 0.95]$ : Average Episodic Return = 4568.06


= SAC Hyperparameter Tuning

For the purpose of hyperparameter tuning, $gamma$, the reward discount and tau, the polyak averaging constant were tuned. 

$gamma$ plays a pivotal role in dictating the agents behaviour. It controls, to describe it intuitively, the patience of the agent. Having a larger $gamma$ forces to the agent to make more long sighted decisions that is give more weight to future rewards, on the other hand a smaller $gamma$ forces the agents to make short sighted decisions. 

$tau$, or the polyak averaging constant, is used in calculating a moving average of the value network weights directly contribute to Q-function leanring. It weighs the contribution of current value network weights and target value network weights (previous moving average estimate). Essentially controlling the influence of immediate changes encountered as part of learning.  

For hyperparameter tuning the number of training steps were reduced to 450k from 1000k(1 million). 

The following values for $gamma$ were chosen: $gamma in {0.90, 0.95, 0.99}$. Here we have tested the change that arises from agressive discounting($gamma = 0.90$) to long horizon planning ($gamma = 0.99$). For $tau$ we chose: $tau in {0.05, 0.001, 0.005}$. Here we have explored towards the upper range of $tau$ relative to the published default @SAC because exploring even lower values of $tau$ on a limited training budget would be uninformative.

#figure(
  table(
    columns: (auto, auto, auto, auto,),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header(
      [], 
      [*$gamma = 0.90$*], 
      [*$gamma = 0.95$*], 
      [*$gamma = 0.99$*]
    ),
    [*$tau = 0.005$*], [2294.81], [2170.51], [8556.36],
    [*$tau = 0.01$*],  [2371.91], [8707.92], [8552.37],
    [*$tau = 0.05$*],  [2385.95], [2174.07], [*9418.79*],
  ),
  caption: [Average episodic return in HalfCheetah-v4 environment with different $gamma$ and $tau$ configurations averaged over 10 episodes using SAC algorithm],
) <tab3>

#figure(
  table(
    columns: (auto, auto, auto, auto,),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header(
      [], 
      [*$gamma = 0.90$*], 
      [*$gamma = 0.95$*], 
      [*$gamma = 0.99$*]
    ),
    [*$tau = 0.005$*], [], [], [],
    [*$tau = 0.01$*],  [], [], [],
    [*$tau = 0.05$*],  [], [], [],
  ),
  caption: [Average episodic return in Ant-v4 environment with different $gamma$ and $tau$ configurations averaged over 10 episodes using SAC algorithm],
) <tab4>

// ─── Results and Comparison (20 marks) ─────────────────────────────────────
= Results and Comparison

Even though these methods work generally well and accomodate the above mentioned challenges, it is important to note that they individually work best for certain cases. PPO is feasible and effective in simulation-driven robotics workflows, where large amounts of data can be generated cheaply. In contrast, SAC is more feasible for real-world robotics applications. Its sample efficiency, robustness to noise, and ability to learn from off-policy data make it better suited for physical systems with limited interaction budgets, however it is tough to tune and is rather sensitive to hyperparameter tuning.

- SAC
  - *high $gamma$*: locomotion tasks are long horizon planning tasks
  - *high $tau$*: performed better with incorporating immediate changes

- PPO
  - *high $epsilon$*: permissive policy update constraint
  - *mid $lambda$*: exact monte carlo ($lambda=1$) is not good should just lean towards monte carlo 

// ─── Proposed Robotics Task (25 marks) ─────────────────────────────────────
= Proposed Robotics Task

- study POMDP

#bibliography("ref.bib", title: "References")
