// ─── Page & Typography Setup (NeurIPS-style) ───────────────────────────────────
#set page(
  paper: "us-letter",
  margin: (left: 1.5in, right: 1.5in, top: 1.0in, bottom: 1in),
  numbering: "1",
  number-align: center + bottom,
  footer-descent: 25pt - 10pt,
  footer: context {
    let i = counter(page).at(here()).first()
    align(center, text(size: 10pt, [#i]))
  },
)

#set text(
  font: ("Times New Roman", "Nimbus Roman", "TeX Gyre Termes"),
  size: 10pt,
)
#set par(
  justify: true,
  leading: 0.43em,
  first-line-indent: 0em,
)
#set block(spacing: 1.0em)
#set heading(numbering: "1.")

#show math.equation: set text(font: "TeX Gyre Termes Math")
#set math.equation(numbering: "(1)")

#let pd(top, bottom) = $frac(partial #top, partial #bottom)$
#let ddot(s) = $dot(dot(#s))$

// ─── Section heading style (NeurIPS) ───────────────────────────────────────────
#show heading.where(level: 1): it => {
  let number = if it.numbering != none {
    counter(heading).display(it.numbering)
  }
  let ex = 7.95pt
  text(size: 12pt, weight: "bold", {
    v(2.7 * ex, weak: true)
    set align(left)
    set par(first-line-indent: 0em)
    [#number #h(1em, weak: true) #it.body]
    v(2 * ex, weak: true)
  })
}

#show heading.where(level: 2): it => {
  let number = if it.numbering != none {
    counter(heading).display(it.numbering)
  }
  let ex = 6.62pt
  text(size: 10pt, weight: "bold", {
    v(2.70 * ex, weak: true)
    set align(left)
    set par(first-line-indent: 0em)
    [#number #h(1em, weak: true) #it.body]
    v(2.03 * ex, weak: true)
  })
}

#show heading.where(level: 3): it => {
  let number = if it.numbering != none {
    counter(heading).display(it.numbering)
  }
  let ex = 6.62pt
  text(size: 10pt, weight: "bold", {
    v(2.6 * ex, weak: true)
    set align(left)
    set par(first-line-indent: 0em)
    [#number #h(1em, weak: true) #it.body]
    v(1.8 * ex, weak: true)
  })
}

// ─── Figure & caption style (NeurIPS) ──────────────────────────────────────────
#set figure.caption(separator: [:])
#show figure: set block(breakable: false)

#show figure.caption: it => {
  set align(center)
  block({
    set align(left)
    set text(size: 10pt)
    it.supplement
    if it.numbering != none [
      #h(0.25em)#it.counter.display(it.numbering)
    ]
    it.separator
    [ ]
    it.body
  })
}

// ─── Header (publication-style, minimal) ───────────────────────────────────────
#set page(header: context {
  let i = counter(page).at(here()).first()
  set text(size: 9pt)
  align(center)[Robot and Reinforcement Learning — Coursework 2]
  line(length: 100%, stroke: 0.35pt + luma(170))
})

// ─── Bibliography font size (NeurIPS uses 9pt small) ───────────────────────────
#show bibliography: set text(size: 9pt)

// ─── Algorithm Introduction (10 marks) ─────────────────────────────────────

= Introduction to PPO and SAC

Proximal Policy Optimization (PPO) @PPO and Soft Actor-Critic (SAC) @SAC are two widely used deep reinforcement learning algorithms which are introduced in the late 2010s. These algorithms are use for continous control but they differ in how they use the _data_. PPO is an on-policy method, which means that each policy iteration update is computed using tragectories collected by the current policy, and when the policy is updated all previously gathered data is discarded. SAC, on the other hand, is an off-policy method, it stores the previous transitions in a replay buffer, meaning that it can continue to learn from older transitions stored in the buffer, which usually makes it much more sample efficient.

PPO is a gradient method that seek to take the largest possible improvement step without destablizing the policy. It was developed as a practical improvement over Trust Region Policy Optimization (TRPO) \@trpo, which enforced a KL divergence constraint on the policy updates. PPO replaces the hard trust-region style of KL divergence with a clipped surrogate objective. The core mechanism relies on the ratio between the updated and the old policies:

$ r_t(theta) = (pi_theta(a_t | s_t)) / (pi_(theta_"old")(a_t | s_t)) $ <r_func>

Though we implement this a bit differently, we use,

$ log r_t (theta) = log pi_theta (a_t | s_t) - log pi_theta_"old" (a_t | s_t) $

then exponentiate to obtain, @r_func. When this ratio is close to 1, the new policy behaves similary to the old one. PPO prevents this ratio from deviating too far by clipping it:

$ L^"CLIP"(theta) = 1/(|cal(B)|) sum_(t in cal(B)) max(-hat(A)_t dot r_t(theta), -hat(A)_t dot "clip"(r_t(theta), 1 - epsilon, 1 + epsilon)) $

$hat(A)_t$ here is the advantage estimation. Before, PPO updates the policy we must estimate how better each action was compared to the baseline. We implement this via Generalised Advantage Estimation, which iterates backward through a collected rollout, and at each timestep t, we compute TD error:

$ delta_t = r_t + gamma V(s_(t + 1)) (1  - d_t) - V(s_t) $

where $d_1$ is 1 if the episode is terminated at step $t$. The advantage is then accumulate via recursively going back the timesteps

$ hat(A)_t  = delta_t + gamma lambda(1 - d_t) hat(A)_(t + 1) $

with, $hat(A) = 0$ at the rollout boundary.



Soft Actor Critic (SAC) is an off-policy actor-critic RL algorithm based on the maximum entropy reinforcement leanring framework. The soft actor-critic algorithm incorporates three key ingredients: an actor-critic architecture with separate policy and value function networks, an off-policy formulation that enables reuse of previously collected data for efficiency, and entropy maximization to enable stability and exploration @SAC. It consists of an actor trying arrive at an optimal policy and a critic that evaluates the generated policy. Both actor critic tend to get better with training, critic building more accurate estimations of state value and Q functions and the actor generating policies with higher returns. The algorithm maintains four different set of weights as described ahead: $psi$ for a soft value function approximator, *$V_psi$* that essentially estimates state value functions of various states, $theta$ for Q value network or critic, *$Q_theta$*, that essentially estimates Q-function values for various state action pairs, $phi$ for the policy network or actor,*$pi_phi$*, that generates policies and finally $overline(psi)$ which represents a target value function *$V_overline(psi)$* and is updated as a moving average of the weights of the soft value function approximator $V_psi$. The following loss functions were proposed in the original paper and the weights are updated by minimising over the mentioned loss functions.

$
&J_(V)(psi) = E_(s_t ~ D)[1/2(V_(psi)(s_t) - E_(a_t ~ pi_phi)[Q_(theta)(s_t, a_t) - log pi(a_t|s_t)])^2] \ $ <eq1>
$
&J_(Q)(theta)=  E_((s_t,a_t)~D)[1/2(Q_(theta)(s_t,a_t) - hat(Q)(s_t,a_t))^2]  "where",\  &hat(Q)(s_t,a_t) = r(s_t,a_t) + gamma E_(s_(t+1) ~ p)[V_(overline(psi))(s_(t+1))]\ $ <eq2>

$
&J_(pi)(phi) =  E_(s_t ~ D,epsilon_t ~ N)[log pi_(phi)(f_(phi)(epsilon_t;s_t)|s_t) - Q_(theta)(s_t,f(epsilon_t;s_t))]
$ <eq3>

$
&overline(psi) <- tau psi + (1-tau)psi
$ <eq4>

However in the code implementation we don't maintain a separate soft value network and directly train $psi$ on critic losses replacing $theta$ in @eq2. Essentially @eq2 can be rewritten as the following:-
$
  &J_(Q)(psi)=  E_((s_t,a_t)~D)[1/2(Q_(psi)(s_t,a_t) - hat(Q)(s_t,a_t))^2]  "where",\  &hat(Q)(s_t,a_t) = r(s_t,a_t) + gamma E_(s_(t+1) ~ p)[V_(overline(psi))(s_(t+1))]\
$

The target netwrok weights $overline(psi)$ are now calculated directly as a moving average of the critic weights.

A neat little trick that was proposed in the original paper and is also a part of the code implementation that is worth noting is that, we actually train two sets of Q-function network weights ${psi_1, psi_2}$ that are trained independently to mitigate positive bias. The minimum of the two Q-functions is then utilised in @eq1 and @eq3. 


SAC was an improvement over previously proposed RL methods in terms of sample efficiency as it is an off-policy method and stability to convergence and sensitivity to hyperparameters which was notoriously tough to achieve with off-policy model free methods.

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
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header([], [*$lambda = 0.90$*], [*$lambda = 0.95$*], [*$lambda = 1.00$*]),
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
    [*$epsilon = 0.3$*], [*3204.60*], [507.76], [72.97],
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
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header([], [*$gamma = 0.90$*], [*$gamma = 0.95$*], [*$gamma = 0.99$*]),
    [*$tau = 0.005$*], [2294.81], [2170.51], [8556.36],
    [*$tau = 0.01$*], [2371.91], [8707.92], [8552.37],
    [*$tau = 0.05$*], [2385.95], [2174.07], [*9418.79*],
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
    [*$tau = 0.005$*], [652.43], [2143.56], [],
    [*$tau = 0.01$*],  [892.43], [1616.85], [5016.65],
    [*$tau = 0.05$*],  [700.09], [2290.87], [3480.42],
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
