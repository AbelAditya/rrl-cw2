// ─── Page & Typography Setup (NeurIPS-style) ───────────────────────────────────
#set page(
  paper: "us-letter",
  margin: (left: 1.5cm, right: 1.5cm, top: 2.7cm, bottom: 2.2cm),
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

PPO is a gradient method that seeks to take the largest possible improvement step without destabilising the policy. It was developed as a practical improvement over Trust Region Policy Optimization (TRPO) \@trpo, which enforced a KL divergence constraint on the policy updates. PPO replaces the hard trust-region style of KL divergence with a clipped surrogate objective.

Before PPO updates the policy, we must estimate how much better each action was compared to the baseline. We implement this via Generalised Advantage Estimation (GAE), which iterates backward through a collected rollout. At each timestep $t$, we compute the TD error:

$ delta_t = r_t + gamma V(s_(t + 1)) (1 - d_t) - V(s_t) $

where $d_t$ is 1 if the episode terminated at step $t$ (masking the bootstrap when there is no valid next state). The advantage is then accumulated via recursively going back the timesteps:

$ hat(A)_t = delta_t + gamma lambda(1 - d_t) hat(A)_(t + 1) $

with $hat(A)_T = 0$ at the rollout boundary. The parameter $lambda$ controls the bias-variance tradeoff in advantage estimation: at $lambda = 0$, estimates reduce to one-step TD errors i.e. low variance but biased because they rely on the value function's accuracy. At $lambda = 1$, estimation becomes equivalent to Monte Carlo returns, unbiased but high variance due to summing over full trajectories. The return targets used to train the critic are then $R_t = hat(A)_t + V(s_t)$.

The core update mechanism relies on the ratio between the updated and the old policies:

$ r_t (theta) = (pi_theta (a_t | s_t)) / (pi_(theta_"old")(a_t | s_t)) $ <r_func>

Though we implement this a bit differently; we compute in log-space for numerical stability:

$ log r_t (theta) = log pi_theta (a_t | s_t) - log pi_(theta_"old") (a_t | s_t) $

then exponentiate to obtain @r_func. When this ratio is close to 1, the new policy behaves similarly to the old one. PPO prevents this ratio from deviating too far by clipping it:

$
  L^"CLIP" (theta) = 1/(|cal(B)|) sum_(t in cal(B)) max(-hat(A)_t dot r_t (theta), -hat(A)_t dot "clip"(r_t (theta), 1 - epsilon.alt, 1 + epsilon.alt))
$

Note that this is equivalent to the original paper's formulation $min(r_t hat(A)_t, "clip"(r_t) hat(A)_t)$ after negation for gradient descent. The intuition here is simple: if the advantage is positive, i.e. the action was good, we would want to increase its probability, but only up to a factor of $(1 + epsilon.alt)$. If the advantage is negative, we do the same but clip at $(1 - epsilon.alt)$. The clipping creates a "pessimistic" lower bound on the policy improvement i.e. the gradient vanishes once the ratio leaves the trusted interval, preventing the catastrophic failures observed with naive policy gradient methods.

The critic $V_phi.alt (s)$ is trained alongside the policy by minimising MSE against the GAE return targets:

$ L^"VF" (phi.alt) = 1/(|cal(B)|) sum_(t) (V_phi.alt (s_t) - R_t)^2 $

This lets the critic learn a smoother estimate of the expected future return, which in turn improves the quality of the advantage estimates used by the actor. In practice, PPO collects a rollout of 2048 steps, computes GAE over the entire buffer, then performs 10 epochs of minibatch gradient descent, recomputing policy log-probabilities on each minibatch and updating both actor and critic. The importance-sampling ratio corrects for the fact that by later epochs, the policy has drifted from the one that collected the data, and the clipping mechanism ensures this drift remains bounded.


*Soft Actor Critic (SAC)* is an off-policy actor-critic RL algorithm based on the maximum entropy reinforcement leanring framework. The soft actor-critic algorithm incorporates three key ingredients: an actor-critic architecture with separate policy and value function networks, an off-policy formulation that enables reuse of previously collected data for efficiency, and entropy maximization to enable stability and exploration @SAC. 

The entropy maximization framework proposes the use of an entropy augmented return objective:
$
  V(s_t) = E_(a_t~pi)[Q(s_t, a_t) + alpha H(pi(dot|s_t)))]
$ 

This encourages the policy to remain sotchastic, promoting exploration and robustness while still optimising for high reward trajectories. The temperature parameter $alpha$ controls the trade off between reward maximization and entropy, with a higher $alpha$ favouring stochasticity and a lower $alpha$ favouring determinism.

The theory from the orginal paper proposes that we have a network for critic ($Q_psi$) and a network for actor ($pi_phi$), in addition to this it was also proposed to have an extra soft value function approximator as it was found to improve training stability. However, we keep the code implementation simple and only implement a critic and an actor.

For training the Q-function network we use the following loss: 
$
  J_(Q)(psi) = E_((s_t,a_t) ~ D)[1/2(Q_(psi)(s_t,a_t) - hat(Q)(s_t,a_t))^2]  "where",\
  hat(Q)(s_t,a_t) = r + gamma (Q_(overline(psi))(s_(t+1),a') - alpha log pi_(phi)(a'|s_(t+1))) ; a' ~ pi_(phi)(dot|s_t)

$<critic_loss_SAC>

The $overline(psi)$ is a set of target value network weights that are calculated as a moving average of the weights of the Q-function network ($psi$).
$
  overline(psi)_(t'+1) = tau psi_t' + (1-tau) overline(psi)_t'
$
Here, $tau$ is the polyak averaging constant and controls how significantly changes in $psi$ influence the change in $overline(psi)$.

A neat little trick that we do employ in the code implementation (also proposed in the original paper) is to have two different set of critic weights ${psi_1, psi_2}$ that are trained simultaneously. We then feed $min(Q_(psi_1)(s_t,a_t),Q_(psi_2)(s_t,a_t))$ and $min(Q_(overline(psi)_1)(s_t,a_t),Q_(overline(psi)_2)(s_t,a_t))$ into respective loss terms to mitigate positive bias. Therefore, @critic_loss_SAC can be rewritten as :-

$
  J_(Q)(psi) = E_((s_t,a_t) ~ D)[1/2(Q_(psi_i)(s_t,a_t) - hat(Q)(s_t,a_t))^2]  "where",\
  hat(Q)(s_t,a_t) = r + gamma (attach(min, b:i={1,2})(Q_(overline(psi_i))(s_(t+1),a')) - alpha log pi_(phi)(a'|s_(t+1))) ; a' ~ pi_(phi)(dot|s_t)

$<critic_loss_code_SAC>

We train the actor using following loss: 
$
  J_(pi)(phi) = E_(a ~ pi_phi)[alpha log pi_(phi)(a_phi|s_t) - attach(min, b:i={1,2})(Q_(psi_i)(a_phi,s_t))]
$
Here we employ the reparameterization trick to reparameterize actions $a$ as $a_phi (s,xi) "where" xi ~ N(0,I)$ making $a_phi$ a deterministic function of $phi "and" xi$ allowing us to push the gradient through the sampling operation. 

Additionally, instead of treating $alpha$ as a fixed parameter we update it along with the other parameters.

$
  J(alpha) = E_(a ~ pi_phi)[- alpha (log pi_phi (a|s) + overline(HH))]
$

Using the above mentioned automatic entropy tuning we manage to remove a hyperparameter to optimize for, rather it now adapts to how well the policy is performing, $alpha$ decreases as the policy is improving allowing the policy to exploit more confidently.

SAC was an improvement over previously proposed RL methods in terms of sample efficiency as it is an off-policy method and stability to convergence and sensitivity to hyperparameters which was notoriously tough to achieve with off-policy model free methods.

// ─── Relevance to Robotics (3 marks)─────────────────────────────────────

= Relevance to Robotics

Robotics problems typically involve continuous state and action spaces, noisy observations, and a strong need for stable learning that makes classical RL approaches impractical. PPO and SAC address several of these, which has led to them becoming baseline algorithms in robotics research.

Both methods natively operate in continuous state and action spaces. PPO parameterises a Gaussian policy from which actions are sampled, while SAC uses a squashed Gaussian to produce bounded continuous actions. This is fundamental to robotics where the agent must output real-valued torques, joint angles, or velocities rather than choosing from a discrete set.

PPO is comparatively robust to catastrophic policy updates because of the clipped objective, and is straightforward to implement i.e. there is a single optimiser, no replay buffer, no target networks. This simplicity also makes it easy to parallelise across many simulation instances, making PPO most feasible in simulation-driven workflows (sim-to-real pipelines) where data generation is cheap.

SAC is especially appealing for real-world robotics, where data collection is expensive and limited. Being an off-policy method, it reuses all past experience via a replay buffer, making it substantially more sample efficient. SAC's entropy-regularised objective also encourages the policy to remain stochastic, reducing over-reliance on any single observation and providing inherent robustness to the sensor noise present in physical systems.


// ─── Environment Description (6 marks) ─────────────────────────────────────

= Environment Description

// TODO: We need to mention the MDP formulation!!

We chose to train in the Ant-v4 and HalfCheetah-v4 environments in MuJoCo gymnasium @mujoco. These were chosen because the represent different complexities of the same locomotion goal. This also makes them directly comparable for benchmarking PPO and SAC.

The *HalfCheetah* is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws). The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward. The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins).

*MDP Formulation for HalfCheetah-v4*:-

#let obs_space_hc = [*$S subset.eq RR^17$*\
- Positional observations ($q_"pos"$, 8 elements):
  - $z$: z-coordinate of the front tip (torso height)
  - $theta$: angle of the front tip
  - Angular positions of: back thigh, back shin, back foot, front thigh, front shin, front foot
- Velocity observations ($q_"vel"$, 9 elements):
  - $dot(x)$: velocity of the x-coordinate of the front tip
  - $dot(z)$: velocity of the z-coordinate of the front tip
  - $dot(theta)$: angular velocity of the front tip
  - Angular velocities of: back thigh, back shin, back foot, front thigh, front shin, front foot
  ]

#let action_space_hc = [*$A subset.eq [-1,1]^6$*
- torque on 6 joints 
  - 3 in hind leg 
  - 3 in front leg]

#let term_condn_hc = [The Half Cheetah never terminates.]

#let reward_fn_hc = [
  *Reward* = _forward_reward_ - _ctrl_cost_
  - *Forward Reward*: reward for moving forward $w_"forward" times (d x)/(d t)$
  - *Control Cost*: a negative reward to penalize the Half Cheetah for taking actions that are too large $w_"control" times attach(||"action"||,tr:2,br:2)$
]

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    stroke: 0.5pt,
    table.header([*Observation Space*], [*Action Space*], [*Termination Condition*], [*Reward Function*]),
    obs_space_hc, action_space_hc, term_condn_hc, reward_fn_hc
  ),
  caption: [MDP formulation for HalfCheetah-v4],
) <mdp_form_hc>


The *Ant* is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts. The goal is to coordinate the four legs to move in the forward (right) direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges).

*MDP Formulation for Ant-v4*:-

#let obs_space_a = [
  *$S subset.eq RR^27$*
  - Positional observations ($q_"pos"$, 13 elements):
    - $z$: z-coordinate of the torso
    - $bold(q)_"orient" in RR^4$: orientation of the torso as a quaternion $(w, x, y, z)$
    - Angular positions of 8 joints: hip and ankle joints for each of the 4 legs
  - Velocity observations ($q_"vel"$, 14 elements):
    - $dot(x), dot(y), dot(z)$: translational velocities of the torso
    - $omega_x, omega_y, omega_z$: angular velocities of the torso
    - Angular velocities of 8 joints: hip and ankle joints for each of the 4 legs
]

#let action_space_a = [
  *$A subset.eq [-1,1]^8$*\
  - Torque on 8 joints
    - 2 joints per limb
]

#let term_condn_a = [
  - Any of the state space values is no longer finite.
  - The z-coordinate of the torso (the height) is not in the closed interval given by the healthy_z_range argument (default is $[0.2,1.0]$).
]

#let reward_fn_a = [
  *Reward* = _healthy_reward_ + _forward_reward_ - _ctrl_cost_ - _contact_cost_
  - *Healthy Reward*: every timestep that the Ant is healthy (defined by termination conditions), it gets a reward of fixed value _healthy_reward_ (default is 1).
  - *Forward Reward*: reward for moving forward $w_"forward" times (d x)/(d t)$
  - *Control Cost*:A negative reward to penalize the Ant for taking actions that are too large $w_"control" times attach(||"action"||,tr:2,br:2)$
  - *Contact Cost*: a negative reward to penalize the Ant if the external contact forces are too large $w_"contact" times attach(||F_"contact"||,tr: 2, br: 2)$
]

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    stroke: 0.5pt,
    table.header([*Observation Space*], [*Action Space*], [*Termination Condition*], [*Reward Function*]),
    obs_space_a, action_space_a, term_condn_a, reward_fn_a
  ),
  caption: [MDP formulation for Ant-v4],
) <mdp_form_hc>

// ─── Hyperparameter Analysis (12 marks) ─────────────────────────────────────

= PPO Hyperparameter Tuning

We chose to tune clip coefficient ($epsilon.alt$) and $lambda$ (in GAE) as these affect the algorithm's working to its core.

The clip coefficient controls the policy update size, this effectively dictates how quickly or slowly the policy is moving towards the optimal. Having the correct step size is important because if the step size is too small then the policy would take a lot of timesteps to reach the optimal or else if the step size is to big the policy would end up overshooting the optimal, both cases result in the formulation of a poor policy.

Tuning lambda controls the bias-variance trade off in estimating the advantage term. As lambda moves from 0 to 1 it cause the shift in advantage estimation being carried out as TD(0) at $lambda = 0$ (high bias) and as Monte Carlo method at $lambda = 1$ (high variance). Temporal Difference introduces bias as it bootstraps value estimates from the immediate next step. Monte Carlo methods are unbiased but inherently show high variance, this originates from variance being accumulated over the length of the episodes. Hence, requires the agent to be run on many episodes for the value estimates to reliably converge.

The following values of $epsilon.alt$ were chosen: $epsilon.alt in {0.1,0.2,0.3}$. We have explored values around the published default @PPO ($epsilon.alt = 0.2$) testing change in agent behaviour under a transition from a strict ($epsilon.alt = 0.1$) to a permissive ($epsilon.alt = 0.3$) policy update constraint. For $lambda$ the following values were chosen: $lambda in {0.9, 0.95, 1.00}$. Similar strategy of exploring around the published defualt @PPO is followed, testing change in agent behaviour as advatage estimation moves from pure monte carlo at $lambda = 1$ to sligthy towards TD(0) at $lambda = 0.90$.


#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header([], [*$lambda = 0.90$*], [*$lambda = 0.95$*], [*$lambda = 1.00$*]),
    [*$epsilon = 0.1$*], [1471.80], [1378.69], [1657.81],
    [*$epsilon = 0.2$*], [4449.39], [2706.83], [2397.13],
    [*$epsilon = 0.3$*], [1131.88], [*4568.06*], [150.90],
  ),
  caption: [Average episodic return in HalfCheetah-v4 environment with different $epsilon$ and $lambda$ configurations averaged over 10 episodes using PPO algorithm],
) <tab1>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header([], [*$lambda = 0.90$*], [*$lambda = 0.95$*], [*$lambda = 1.00$*]),
    [*$epsilon.alt = 0.1$*], [2498.17], [1419.31], [11.57],
    [*$epsilon.alt = 0.2$*], [2277.72], [3049.39], [107.12],
    [*$epsilon.alt = 0.3$*], [*3204.60*], [507.76], [72.97],
  ),
  caption: [Average episodic return in Ant-v4 environment with different $epsilon$ and $lambda$ configurations averaged over 10 episodes using PPO algorithm],
) <tab2>


// TODO: We don't analyze the data, at all. We just say these things happened, we never comment on why is one change better than the other one. Same for SAC.
// WE should say, which is the best configuration, i know these are highlighted in the table but we need to explain the relation between the hyperparameter, and why this particular results in the best config
// And what is gamma = 1.0 always failing for PPO.

// - Two sets of $epsilon.alt$ & $lambda$ values have shown good performance with close episodic returns
//   - $[epsilon.alt = 0.2, lambda = 0.9]$ : Average Episodic Return = 4449.39
//   -  $[epsilon.alt = 0.3, lambda = 0.95]$ : Average Episodic Return = 4568.06


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
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header([], [*$gamma = 0.90$*], [*$gamma = 0.95$*], [*$gamma = 0.99$*]),
    [*$tau = 0.005$*], [810.21], [2272.73], [4823.77],
    [*$tau = 0.01$*], [1102.83], [1572.79], [*5369.59*],
    [*$tau = 0.05$*], [809.55], [1369.41], [3276.02],
  ),
  caption: [Average episodic return in Ant-v4 environment with different $gamma$ and $tau$ configurations averaged over 10 episodes using SAC algorithm],
) <tab4>

// ─── Results and Comparison (20 marks) ─────────────────────────────────────
= Results and Comparison

- Comment on sample efficiency, training stability, wall-clock training time, and reliability across seeds; 
- Also comment on the qualitative behaviour of the learned policies: what do they look like? Do you think they have adequately solved the task? Why or why not?

- *Sample efficiency*: SAC higher 
  - theoretically, since SAC is an off policy method 
  - SAC was only trained over 450K time steps as opposed to PPO's 1 million timesteps still SAC shows better performance 

- Training stability:

- wall clock training time: SAC higher
  - SAC 
    - half cheetah: 2 hrs
    - ant: 1hr 40min
  - PPO
    - ant: 1hr 10 min
    - half cheetah: 1hr 

  - from fig we can tell ppo shows better stability across seeds

We found a general trend of decreasing performance while shifting from HalfCheetah-v4 environment to Ant-v4 environment. This can be attributed to an increase in complexity for achieving locomotion. [Ant-v4 more complex because more joints to control $6->8$, there is also a complexity of maintaining a heading or building a heading agnostic locomotion strategy.]. SAC consistently outperformed PPO across both environments; this has been discussed in detail later in this section.
// TODO: This needs much more work, it is worth 20 marks.
// missing subsections: sample efficiancy, training stablity,  wall-clock time, reliability across seeds and qualitative policy behaviour. We only talk about hyperparameters and nothing else.


We found a general trend of decreasing performance while shifting from HalfCheetah-v4 environment to Ant-v4 environment. This can be attributed to an increase in complexity for achieving locomotion. [Ant-v4 more complex because more joints to control $6->8$, there is also a complexity of maintaining a heading or building a heading agnostic locomotion strategy.]. SAC consistently outperformed PPO across both environments; this has been discussed in detail later in this section.

$lambda = 0.95$ & $epsilon.alt = 0.3$
We got the best performance from *PPO* for *HalfCheetah-v4* environment using *${lambda=0.95, epsilon.alt = 0.3}$* giving an average episodic return of *4568.06* and for *Ant-v4* environment using *${lambda = 0.9,epsilon.alt = 0.3}$* giving an average episodic return of *3204.60*. We can infer that a more permissive policy update constraint was required because of having a limited training budget. We notice a shift towards Temporal Difference approximations of advantage showing better performance dk y.

Even though these methods work generally well and accomodate the above mentioned challenges, it is important to note that they individually work best for certain cases. PPO is feasible and effective in simulation-driven robotics workflows, where large amounts of data can be generated cheaply. In contrast, SAC is more feasible for real-world robotics applications. Its sample efficiency, robustness to noise, and ability to learn from off-policy data make it better suited for physical systems with limited interaction budgets, however it is tough to tune and is rather sensitive to hyperparameter tuning.

- comparatively SAC gave better performance than PPO

// NOTE: We don't need to mention this here, we cover the same thing in Relevance section
// Even though these methods work generally well and accomodate the above mentioned challenges, it is important to note that they individually work best for certain cases. PPO is feasible and effective in simulation-driven robotics workflows, where large amounts of data can be generated cheaply. In contrast, SAC is more feasible for real-world robotics applications. Its sample efficiency, robustness to noise, and ability to learn from off-policy data make it better suited for physical systems with limited interaction budgets, however it is tough to tune and is rather sensitive to hyperparameter tuning.

- SAC
  - *high $gamma$*: locomotion tasks are long horizon planning tasks
  - *high $tau$*: performed better with incorporating immediate changes

- PPO
  - *high $epsilon.alt$*: permissive policy update constraint
  - *mid $lambda$*: exact monte carlo ($lambda=1$) is not good should just lean towards monte carlo

// ─── Proposed Robotics Task (25 marks) ─────────────────────────────────────
= Proposed Robotics Task

The 2026 Formula 1 regulations introduced a fundamental shift in power unit architecture: roughly half the car's power now comes from the electrical deployment via the upgraded MGU-K (tripled the capacity of the last year to 350kW), and the MGU-H is removed entirely. However, battery capacity is stil the same 4MJ, this means that the battery is charged and depleted multiple times during the lap. This makes within-lap energy management a continous, high-freqency control problem, the driver must constantly decide when to deploy battery energy, when to harvest it (basically slowing down to charge up the battery/super clipping), and to coordinate this with new active aerodynamic, because lifting off turns off the active aero more harvesting but less straight line speed. We propse framing this within-lap energy and pace management task as a POMDP, because most of the important variables critical to the decision making are hidden from the agent. Tyre degradation is only indirectly observed through surface temperature, yet internal rubber state can collapse suddenly (the "type cliff"). Competitor energy levels and strategy are invisible, the agent can see their lap times and gaps but their battery state or pit strategy. Track grip evolves each lap as rubber is laid down, but cannot be measured directly, wind and weather condition affect the grip levels. Battery health degrade over a race distance, which cannot be fully captured.

The true state $s$ would include the car's exact postion, velocity, battery state, tyre condition and type, fuel load, aerodynamic state, track grip, weather effects, and the hidden state of the rival's cars. The observation $o$ would consist of car telemetry available to the engineers and driver, such as speed, throttle and brake traces, battery charge, type temperature and presures, lap and sector time, time gaps to the nearby cars and overtake availability, active aero deployments, and noisy weather forecast. The action $a in RR^3$ is continous: an energy deployment level (harvest to full deploy), a target pace delta (push v/s conserve) and harvesting intensity (trading lap times for energy but also disables active aero). The reward function combine sector times relative to target sector pace, and energy management penalty for depleting the battery at critical momemt (e.g. end of the straights) and tyre preservation.

We would choose SAC for this task. Our experimments showed that high $gamma$ was dominant factor for SAC's performance. When $gamma = 0.90$, the agent could "see" about 10 step into the future $1 slash (1 - gamma) approx 10$ and the performance was stuck around $2300$ regardless of what we did with $tau$. But when $gamma = 0.99$ (effective horizon of \~100 steps) the performance jumped to 8500-9400. The agent needed to reason about the long chain cause and effect, and locomotion requires these long horizons. On top of that within-lap energy mangement involves dozes of coupled deployments decisions per lap whose consequences computer over subsequent sections and laps. A deployment burst to overtake now might make you vulnerable to few corners later, and if a opponent is within an second of you they will have additional power to deploy as well.

SAC's entropy-regularised objective suits this task because there is not single correct deployement strategy, the optimial profile would depend on competitor behaviour, tyre state, and track postion, all of which are uncertain, so a stochastic policy that maintains viable strategies is more robust than a single determinstic one. PPO would only be attractive with access to massively parallel simulation, whereas high fidelity race simuation involving tyre theromodymaics, aero mode interactions, and competitor AI is too expensive per rollout for on-policy data gathering. 

Key challenges with the approach is sim-to-real gap (tyre models are notoriously unreliable, even Pirelli's tyre models often misjudge the degradation speed), multi-agent non-stationarity (competitor strategies are adversarial and partially observable, but SAC and even PPO assumes stationary environments) and reward shaping across competiting objectives (trying to optimize race position, lap time, energy efficiency & tyre preservation and these objectives directly conflict with each other). 

Having said everything, RL may not be the ideal approach in isolation. Classical optimzation techiniques are excellent when you have a good model and well defined contraints. F1 teams already use these for pit-stop strategies, pit-stop timing a relatively low-frequency decision (2-3 times per race), the option space is discrete and enumeralble (pit on lap 15, 16, 17... with soft, medium or hard tyres) and the models, though still inperfect, are good enough of this level of abstraction. But when the classical techniques struggle with high-freqency, continous, reactive decision making that the 2026 reguations demand. "How much energy should I deploy on the second straight given that the car behind just closed to within 1.2 seconds and they have a 3 lap tyre advantage?" this is not a question that the we can pre-compute, the state-space is too large, decision are too frequent and the interation between energy, tyres, and competitor behaviour are too complex for hand-crafted heuristics to capture optimality. This is excatly where RL excels: learning a reactive policy through experience that maps high dimensional observations to continous action spaces. _An hybrid approch_ bridges the gap between this boundary, classical methods for strategic decisions with RL policy handling continous within-lap energy deployment decision initialised by _Imitation Learning_. Rather than starting from random exploartion, we can initialize the RL policy by leanring from historical data, how drivers and race engineers managed the previous races and avoid dangerous random explorations. The policy start by mimicking human experts behaviour and then improves beyond it through RL (same as AlphaGo).

#bibliography("ref.bib", title: "References")
