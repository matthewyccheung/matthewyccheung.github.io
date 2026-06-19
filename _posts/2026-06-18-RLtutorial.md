---
layout: post
title: "A Tutorial Guide to TRPO and PPO"
math: true
---

TRPO and PPO can look intimidating because they sit at the intersection of constrained optimization, probability ratios, value functions, and rollout data. The good news is that the core story is much simpler than the notation first suggests: collect data with the current policy, estimate which actions were better than expected, then update the policy without moving it too far.

This tutorial walks through that story from the ground up. We start with the constrained-optimization language behind TRPO, move into the reinforcement-learning objects that PPO and TRPO share, and then connect the pieces to GAE, surrogate objectives, trust regions, clipping, and the most common points of confusion. The reinforcement-learning sections are grounded in the OmniSafe TRPO and PPO tutorials, with additional context from the original TRPO, PPO, and GAE papers.

## 1. Constrained optimization background

TRPO is built around a constrained policy update, so it helps to first recall how constrained optimization is usually written. A standard minimization problem looks like this:

$$\min_x f(x)$$

subject to inequality constraints

$$h_i(x)\le 0,\qquad i=1,\dots,m$$

and equality constraints

$$\ell_j(x)=0,\qquad j=1,\dots,r.$$

This format is not restrictive; it is just a convenient convention. If a constraint is written in a different direction, we move everything to the left-hand side. For example,

$$x\le 5$$

can be rewritten as

$$x-5\le 0.$$

Likewise,

$$x\ge 5$$

can be rewritten as

$$5-x\le 0.$$

Equality constraints work the same way. The equation

$$x+y=3$$

becomes

$$x+y-3=0.$$

So when you see $h_i(x)\le 0$, read it as "the $i$-th inequality constraint is satisfied." When you see $\ell_j(x)=0$, read it as "the $j$-th equality constraint is exactly satisfied."

## 2. Primal problem, Lagrangian, and dual problem

The **primal problem** is the original problem we actually care about:

$$\min_x f(x)$$

subject to

$$h_i(x)\le 0,\qquad \ell_j(x)=0.$$

Here $x$ is the variable being optimized, and the constraints describe which values of $x$ are feasible.

The **Lagrangian** folds the objective and constraints into one function:

$$
L(x,u,v)
=
f(x)
+
\sum_i u_i h_i(x)
+
\sum_j v_j \ell_j(x).
$$

The inequality multipliers satisfy

$$u_i\ge 0$$

and the equality multipliers $v_j$ can be positive, negative, or zero. The Lagrangian is not the dual problem yet; it is the bridge that lets us move from the primal variables to the dual variables.

The **dual function** minimizes that bridge over the primal variable:

$$g(u,v) = \inf_x L(x,u,v).$$

For fixed multipliers $u,v$, we search over $x$. After that minimization, the result depends only on the multipliers.

The **dual problem** then chooses the best multipliers:

$$\max_{u,v} g(u,v)$$

subject to

$$u\ge 0.$$

It is useful to keep the three layers separate:

| Object | What it optimizes over | Meaning |
|---|---:|---|
| Primal problem | $x$ | Original constrained problem |
| Lagrangian | $x,u,v$ | Objective plus weighted constraints |
| Dual problem | $u,v$ | Best lower bound on the primal optimum |

For a minimization primal problem, every feasible dual point gives a lower bound on the primal optimum. The dual problem searches for the tightest lower bound it can prove.

## 3. Why the dual is a convex optimization problem

One pleasant fact about the dual is that it has a convex-optimization structure even when the original primal problem is not convex. Start with the Lagrangian:

$$L(x,u,v) = f(x) + u^T h(x) + v^T\ell(x).$$

If $x$ is fixed, then $L$ is affine in $u,v$. As a function of the dual variables, it is a plane or hyperplane.

The dual function takes the infimum over all those affine functions:

$$g(u,v) = \inf_x L(x,u,v).$$

That makes $g$ the pointwise infimum of many affine functions. The pointwise infimum of affine functions is always concave. Intuitively, $g$ is the lower envelope of a collection of planes, and a lower envelope bends downward rather than upward.

The dual problem is

$$\max_{u\ge 0,v} g(u,v).$$

Maximizing a concave function over a convex feasible set is a convex optimization problem. This remains true even if the primal problem itself is not convex, because the concavity comes from the construction of the dual function.

# Reinforcement learning setup

## 4. States, actions, rewards, policies, and trajectories

In reinforcement learning, an agent repeatedly interacts with an environment. At time $t$, it observes a state $s_t$, samples or selects an action, and then receives a reward plus the next state:

$$a_t\sim \pi_\theta(\cdot\mid s_t).$$

$$s_t \xrightarrow{a_t} (r_t,s_{t+1}).$$

Repeating this interaction gives a rollout, also called a trajectory:

$$\tau=(s_0,a_0,r_0,s_1,a_1,r_1,s_2,\dots).$$

The policy is the distribution over actions:

$$\pi_\theta(a\mid s),$$

which means "the probability of choosing action $a$ in state $s$, under parameters $\theta$."

For continuous action spaces, the policy is often a Gaussian:

$$\pi_\theta(a\mid s) = \mathcal N(\mu_\theta(s),\Sigma_\theta).$$

For discrete action spaces, it is often a softmax distribution:

$$\pi_\theta(a\mid s) = \mathrm{softmax}(f_\theta(s)).$$

## 5. How rewards are computed in practice

The reward $r_t$ is usually designed by the environment creator. It is not predicted by the policy, and it is not known before the action is taken. The policy chooses an action, the environment responds, and only then does the agent observe the reward.

For a robot-arm grasping task, a reward function might combine distance, control effort, and task success:

$$
r_t
=
-\|x_{\text{gripper},t}-x_{\text{object},t}\|_2
-0.01\|a_t\|^2
+
10\cdot \mathbf 1\{\text{object grasped}\}.
$$

This reward gives a penalty for being far from the object, a small penalty for large motor commands, and a large bonus for successfully grasping the object.

For example, suppose

$$\|x_{\text{gripper},t}-x_{\text{object},t}\|_2=0.20,\qquad \|a_t\|^2=3,$$

and the object is not grasped. The reward is then

$$r_t=-0.20-0.01(3)+0=-0.23.$$

If the object is grasped, the reward might become

$$r_t=-0.04-0.01(1)+10=9.95.$$

The timing matters. In ordinary model-free RL, $r_{t+1}$ is not available at time $t$. The agent must first take the next action, transition again, and observe the next reward. The timeline is:

$$s_t \xrightarrow{a_t} (r_t,s_{t+1})$$

then

$$s_{t+1} \xrightarrow{a_{t+1}} (r_{t+1},s_{t+2}).$$

Only after collecting the rollout do we have the reward sequence

$$r_t,r_{t+1},r_{t+2},\dots$$

available for computing returns and advantages. That distinction is the source of many confusions about GAE and policy updates: the policy acts online, but the training calculation happens after data collection.

## 6. Discounted returns and the role of $\gamma$

The discounted return from time $t$ is the future reward stream, with later rewards multiplied by powers of $\gamma$:

$$G_t = r_t + \gamma r_{t + 1} + \gamma^2r_{t + 2} + \gamma^3r_{t + 3} + \cdots.$$

Equivalently,

$$G_t = \sum_{k = 0}^{\infty}\gamma^k r_{t + k}.$$

The discount factor satisfies

$$0\le \gamma \le 1.$$

The exponent appears because we discount once per time step. If

$$\gamma=0.9,$$

then the future rewards are weighted by

$$1,\;0.9,\;0.81,\;0.729,\dots$$

so a reward two steps in the future is worth $0.9^2=0.81$ times as much as an immediate reward.

There are three main reasons for discounting:

1. Future rewards are usually less directly attributable to the current action.
2. Future rewards may be more uncertain.
3. In infinite-horizon tasks, discounting helps keep the sum finite.

A higher $\gamma$ makes the agent more far-sighted. A lower $\gamma$ makes it more short-sighted.

## 7. Why use future rewards at all?

The reason we care about future rewards is that some actions look bad immediately but are good because of what they make possible later. Suppose a robot arm gets

$$r_t=-1,\qquad r_{t+1}=-1,\qquad r_{t+2}=10.$$

With

$$\gamma=0.9,$$

the return is

$$
G_t
=
-1+0.9(-1)+0.9^2(10)
=
-1-0.9+8.1
=
6.2.
$$

The immediate reward is negative, but the long-term return is positive. Without future rewards, the algorithm would incorrectly punish an action that set up later success.

# Value functions and advantage functions

## 8. What the value function means

The value function under policy $\pi$ is

$$
V^\pi(s)
=
\mathbb E_\pi
\left[
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\mid s_t=s
\right].
$$

It answers:

> If I start in state $s$, and then follow policy $\pi$, how much discounted future reward do I expect?

The value of a state depends on the policy. The same state can be good under one policy and bad under another.

For example, a robot standing near a cliff is not inherently good or bad. A cautious policy may move away from the cliff, so the state has high value. A reckless policy may fall off, so the state has low value.

In neural-network actor-critic methods, we usually approximate the value function with a learned critic:

$$V_\phi(s)\approx V^\pi(s).$$

Here $\phi$ are the critic parameters.

## 9. How the value function is learned

The value function is learned during RL training, not usually known beforehand.

The training loop is:

$$
\text{collect trajectories}
\rightarrow
\text{compute return targets}
\rightarrow
\text{fit }V_\phi(s)\text{ to those targets}.
$$

For example, suppose one robot trajectory has rewards

$$r_0=-0.20,\qquad r_1=-0.15,\qquad r_2=-0.08,\qquad r_3=9.95.$$

With

$$\gamma=0.99,$$

the return target for $s_0$ is

$$
G_0
=
-0.20
+0.99(-0.15)
+0.99^2(-0.08)
+0.99^3(9.95)
\approx 9.23.
$$

So the critic should learn

$$V_\phi(s_0)\approx 9.23.$$

The critic is trained by regression:

$$
\min_\phi
\frac{1}{N}
\sum_{t=1}^N
\left(
V_\phi(s_t)-G_t
\right)^2.
$$

In longer tasks, we often use a bootstrapped target:

$$
\widehat G_t^{(K)}
=
r_t+\gamma r_{t+1}
+\cdots
+\gamma^{K-1}r_{t+K-1}
+
\gamma^K V_\phi(s_{t+K}).
$$

This means: use observed rewards for $K$ steps, then use the value function to estimate the rest.

## 10. What function class is used for $V_\phi(s)$?

For a small tabular environment, $V$ can literally be a table:

$$V(s_1)=3.2,\qquad V(s_2)=5.7,\qquad V(s_3)=-1.0.$$

For a simple continuous problem, it can be linear:

$$V_\phi(s)=\phi^Ts.$$

For modern TRPO and PPO implementations, it is usually a neural network:

$$V_\phi(s)=\mathrm{MLP}_\phi(s).$$

For a robot arm, the state might include joint angles, velocities, gripper position, and object position. The value network outputs one scalar number: the predicted future return.

## 11. Advantage function

The advantage function is

$$A^\pi(s,a) = Q^\pi(s,a)-V^\pi(s).$$

It answers:

> Was action $a$ better or worse than what policy $\pi$ usually does in state $s$?

Here,

$$V^\pi(s)$$

is the expected return if the policy behaves normally starting from $s$, while

$$Q^\pi(s,a)$$

is the expected return if we force the first action to be $a$, then follow policy $\pi$ afterward.

So:

- $A^\pi(s,a)>0$: action $a$ was better than expected.
- $A^\pi(s,a)<0$: action $a$ was worse than expected.

The policy update uses this signal:

- Increase probability of actions with positive advantage.
- Decrease probability of actions with negative advantage.

# GAE: Generalized Advantage Estimation

## 12. The one-step TD residual

The temporal-difference residual is

$$\delta_t^V = r_t + \gamma V(s_{t + 1})-V(s_t).$$

This is a one-step estimate of advantage.

The term

$$r_t+\gamma V(s_{t+1})$$

is an estimate of the return after taking action $a_t$. Subtracting

$$V(s_t)$$

compares that result to what was expected from $s_t$.

Example:

$$r_t=-0.08,\qquad V(s_t)=5.0,\qquad V(s_{t+1})=6.0,\qquad \gamma=0.99.$$

Then

$$
\delta_t^V
=
-0.08+0.99(6.0)-5.0
=
0.86.
$$

Even though the immediate reward was negative, the action was good because it moved the robot into a more promising state.

## 13. $k$-step advantage estimates

A one-step estimate is

$$\widehat A_t^{(1)} = \delta_t^V.$$

A two-step estimate is

$$\widehat A_t^{(2)} = \delta_t^V + \gamma\delta_{t + 1}^V.$$

This equals

$$\widehat A_t^{(2)} = -V(s_t) + r_t + \gamma r_{t + 1} + \gamma^2V(s_{t + 2}).$$

A $k$-step estimate is

$$\widehat A_t^{(k)} = \sum_{\ell = 0}^{k-1}\gamma^\ell\delta_{t + \ell}^V.$$

Equivalently,

$$\widehat A_t^{(k)} = -V(s_t) + r_t + \gamma r_{t + 1} + \cdots + \gamma^{k-1}r_{t + k-1} + \gamma^kV(s_{t + k}).$$

Small $k$ uses more bootstrapping from $V$. It tends to have lower variance but more bias.

Large $k$ uses more observed rewards. It tends to have lower bias but more variance.

## 14. GAE formula

GAE combines the $k$-step estimators with exponentially decaying weights:

$$
\widehat A_t^{\mathrm{GAE}}
=
(1-\lambda)
\left(
\widehat A_t^{(1)}
+
\lambda \widehat A_t^{(2)}
+
\lambda^2\widehat A_t^{(3)}
+\cdots
\right).
$$

This simplifies to

$$
\widehat A_t^{\mathrm{GAE}}
=
\sum_{\ell=0}^{\infty}
(\gamma\lambda)^\ell
\delta_{t+\ell}^V.
$$

The parameter $\lambda$ controls the bias-variance tradeoff.

If

$$\lambda=0,$$

then

$$\widehat A_t^{\mathrm{GAE}}=\delta_t^V.$$

This is low variance but can be biased because it relies heavily on $V$.

If

$$\lambda=1,$$

then

$$\widehat A_t^{\mathrm{GAE}} = \sum_{\ell = 0}^{\infty}\gamma^\ell r_{t + \ell}-V(s_t).$$

This is Monte Carlo return minus the value baseline. It has lower bias but higher variance.

Intermediate $\lambda$ values compromise between the two.

### 14.1 When can GAE use future timesteps?

GAE does not give the acting policy access to the future. At time $t$, the policy only observes $s_t$ and samples $a_t$. Future rewards such as $r_{t+1}$ and states such as $s_{t+2}$ are only available after the rollout has already happened.

During training, after the trajectory is collected, GAE can compute

$$
\widehat A_t^{\mathrm{GAE}}
=
\delta_t^V
+
\gamma\lambda\delta_{t+1}^V
+
(\gamma\lambda)^2\delta_{t+2}^V
+\cdots.
$$

So the advantage estimate at time $t$ is forward-looking relative to $t$, but it is computed afterward from observed rollout data. If an implementation loops backward through the rollout, that is only a dynamic-programming trick using

$$\widehat A_t = \delta_t^V + \gamma\lambda\widehat A_{t + 1}.$$

The formula still means that $\widehat A_t$ contains information from later observed timesteps.

For a finite rollout ending at $T$, the estimator is

$$\widehat A_t^{\mathrm{GAE}} = \sum_{\ell = 0}^{T-t-1}(\gamma\lambda)^\ell\delta_{t + \ell}^V.$$

If the episode truly terminates at $T$, the terminal value is usually set to $V(s_T)=0$. If the rollout is truncated but the episode has not ended, the algorithm bootstraps from $V(s_T)$ to estimate the unobserved continuation.

### 14.2 What is GAE bias biased toward?

Yes: in the GAE bias-variance tradeoff, “bias” means bias in the estimator $\widehat A_t$ of the true advantage $A^\pi(s_t,a_t)$. The bias usually comes from bootstrapping with an imperfect value function $V_\phi(s)$, and from truncating rollouts.

Lower $\lambda$ relies more heavily on short-horizon TD estimates and $V_\phi$, so it usually has lower variance but more bias if the critic is inaccurate. Higher $\lambda$ uses more observed future rewards, so it usually has lower bias but higher variance.

# Policy training and initialization

## 15. Is the policy trained during RL?

Yes. The policy is trained during RL, alongside the value function. The policy is the actor,

$$\pi_\theta(a\mid s).$$

and the value function is the critic,

$$V_\phi(s).$$

One training cycle looks like this:

$$
\text{initialize actor and critic}
\rightarrow
\text{collect trajectories with current actor}
\rightarrow
\text{compute returns and advantages}
\rightarrow
\text{update actor}
\rightarrow
\text{update critic}
\rightarrow
\text{repeat}.
$$

They are trained in parallel but with different objectives.

The policy is trained to choose better actions. The value function is trained to predict future returns.

## 16. How is the policy initialized?

Most commonly, the policy is initialized randomly, using normal neural-network initialization methods such as Xavier initialization or orthogonal initialization.

For a continuous-control robot, the initial policy might be

$$\pi_{\theta_0}(a\mid s)=\mathcal N(\mu_{\theta_0}(s),\Sigma_{\theta_0}).$$

The mean network $\mu_{\theta_0}(s)$ may output small near-zero values or random small values. The covariance or standard deviation controls exploration.

For a discrete action environment, a randomly initialized softmax policy often begins close to uniform:

$$\pi_{\theta_0}(\text{left}\mid s)\approx 0.25,\quad \pi_{\theta_0}(\text{right}\mid s)\approx 0.25,$$

$$\pi_{\theta_0}(\text{up}\mid s)\approx 0.25,\quad \pi_{\theta_0}(\text{down}\mid s)\approx 0.25.$$

Sometimes the policy is initialized from imitation learning, a hand-coded controller, or a pretrained model. But in standard TRPO/PPO, random initialization is common.

# Performance difference and surrogate objectives

## 17. Performance difference lemma

The OmniSafe TRPO tutorial writes the performance difference between two policies as

$$
J^R(\pi')
=
J^R(\pi)
+
\mathbb E_{\tau\sim\pi'}
\left[
\sum_{t=0}^{\infty}\gamma^tA_\pi^R(s_t,a_t)
\right].
$$

This says:

> The return of the new policy equals the return of the old policy plus the expected discounted advantages of the new policy’s actions, evaluated relative to the old policy.

The expectation is over trajectories generated by $\pi'$.

## 18. Rewriting trajectory expectation into state-action probabilities

Start with

$$
\mathbb E_{\tau\sim\pi'}
\left[
\sum_{t=0}^{\infty}\gamma^tA_\pi(s_t,a_t)
\right].
$$

By linearity of expectation,

$$
=
\sum_{t=0}^{\infty}
\gamma^t
\mathbb E_{\tau\sim\pi'}
\left[
A_\pi(s_t,a_t)
\right].
$$

Now expand the expectation over possible state-action pairs:

$$
\mathbb E[A_\pi(s_t,a_t)]
=
\sum_s\sum_a
P(s_t=s,a_t=a\mid \pi')
A_\pi(s,a).
$$

Factor the joint probability:

$$P(s_t = s,a_t = a\mid \pi') = P(s_t = s\mid\pi')\pi'(a\mid s).$$

Therefore,

$$
\mathbb E[A_\pi(s_t,a_t)]
=
\sum_s
P(s_t=s\mid\pi')
\sum_a
\pi'(a\mid s)A_\pi(s,a).
$$

Substituting back gives

$$
J^R(\pi')
=
J^R(\pi)
+
\sum_{t=0}^{\infty}
\sum_s
P(s_t=s\mid\pi')
\sum_a
\pi'(a\mid s)
\gamma^tA_\pi(s,a).
$$

Define the discounted state visitation measure

$$d_{\pi'}(s) = \sum_{t = 0}^{\infty} \gamma^tP(s_t = s\mid\pi').$$

Then

$$
J^R(\pi')
=
J^R(\pi)
+
\sum_s d_{\pi'}(s)
\sum_a \pi'(a\mid s)A_\pi(s,a).
$$

This form is useful because it separates:

1. how often the new policy visits each state, and
2. what actions the new policy takes in those states.

## 19. Surrogate objective: what is being approximated?

The exact expression is

$$
J^R(\pi')
=
J^R(\pi)
+
\sum_s d_{\pi'}(s)
\sum_a \pi'(a\mid s)A_\pi(s,a).
$$

The hard part is

$$d_{\pi'}(s),$$

because it depends on the new policy $\pi'$. Computing the future state distribution under every candidate new policy is hard.

TRPO therefore uses the surrogate

$$
L_\pi(\pi')
=
J^R(\pi)
+
\sum_s d_\pi(s)
\sum_a \pi'(a\mid s)A_\pi(s,a).
$$

The approximation is:

$$d_{\pi'}(s)\approx d_\pi(s).$$

This is the crucial clarification:

$$\boxed{\text{TRPO is not approximating }\pi'\text{ with }\pi.}$$

It is approximating the new policy’s state visitation distribution with the old policy’s state visitation distribution.

The new policy still appears inside

$$\pi'(a\mid s).$$

So the surrogate asks:

> If the states we visit stayed approximately the same as under the old policy, would the new policy choose better actions in those states?

This approximation is only reasonable when the new policy stays close to the old one. That is why TRPO uses a KL trust region.

## 20. Importance-sampling ratio

In practice, the data are collected from the old policy

$$\pi_{\theta_k}.$$

But the surrogate evaluates a candidate new policy

$$\pi_\theta.$$

To use old-policy data for a new-policy objective, we use the probability ratio

$$\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)} {\pi_{\theta_k}(a_t\mid s_t)}.$$

Here I write $\rho_t(\theta)$ instead of $r_t(\theta)$ to avoid confusion with reward $r_t$.

The sampled surrogate is

$$
\widehat L_k(\theta)
=
\frac{1}{N}
\sum_{t=1}^{N}
\rho_t(\theta)\widehat A_t.
$$

If

$$\widehat A_t>0,$$

then maximizing $\widehat L_k$ increases the probability of action $a_t$ in state $s_t$.

If

$$\widehat A_t<0,$$

then maximizing $\widehat L_k$ decreases the probability of action $a_t$ in state $s_t$.

## 21. Why optimize after the trajectory is already observed?

The observed trajectory is not the final goal. It is a training sample.

After rollout, we know what happened under the old policy. But the policy has not yet been improved.

The point of optimizing the surrogate is to update future behavior:

$$\theta_k\rightarrow \theta_{k+1}.$$

The data say which actions were better or worse than expected. The objective changes the policy so that next time, in similar states, better actions are more likely and worse actions are less likely.

This is analogous to supervised learning. Once you have a dataset of images and labels, you still optimize a loss function because you want the model to perform better on future images.

In RL:

$$\text{rollout} = \text{training data},$$

$$\widehat L_k(\theta)=\text{policy training objective}.$$

# TRPO

## 22. TRPO core idea

TRPO tries to improve the surrogate objective while keeping the new policy close to the old policy.

The practical constrained problem is

$$\max_\theta L_{\theta_{\text{old}}}(\theta)$$

subject to

$$\bar D_{\mathrm{KL}}(\theta_{\text{old}},\theta)\le \delta.$$

The average KL divergence is

$$
\bar D_{\mathrm{KL}}(\theta_{\text{old}},\theta)
=
\mathbb E_{s}
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\text{old}}}(\cdot\mid s)
\;\|\;
\pi_\theta(\cdot\mid s)
\right)
\right].
$$

The KL constraint says:

> Improve the policy, but do not change it so much that the old trajectory becomes irrelevant.

The OmniSafe TRPO tutorial notes that the theoretical maximum-state KL constraint is impractical, so practical TRPO uses an average KL approximation.

## 23. How the KL constraint is imposed in TRPO

TRPO imposes the KL constraint as a constrained optimization problem, not merely as a regular loss term.

It solves approximately:

$$\max_\theta L_{\theta_{\text{old}}}(\theta)$$

subject to

$$\bar D_{\mathrm{KL}}(\theta_{\text{old}},\theta)\le \delta.$$

Locally, let

$$\Delta\theta=\theta-\theta_{\text{old}}.$$

Linearize the objective:

$$L(\theta_{\text{old}} + \Delta\theta) \approx L(\theta_{\text{old}}) + g^T\Delta\theta.$$

Quadratically approximate the KL constraint:

$$\bar D_{\mathrm{KL}}(\theta_{\text{old}},\theta_{\text{old}} + \Delta\theta) \approx \frac{1}{2}\Delta\theta^TH\Delta\theta.$$

Then TRPO solves the local subproblem

$$\max_{\Delta\theta} g^T\Delta\theta$$

subject to

$$\frac{1}{2}\Delta\theta^TH\Delta\theta\le \delta.$$

The natural-gradient direction is approximately

$$s\approx H^{-1}g.$$

Rather than compute $H^{-1}$, TRPO solves

$$Hs=g$$

using conjugate gradient.

The step is scaled by

$$\beta = \sqrt{ \frac{2\delta}{s^THs} }.$$

So the full candidate step is

$$\Delta\theta_{\mathrm{full}}=\beta s.$$

Then TRPO performs backtracking line search to make sure the actual neural-network update improves the surrogate objective and satisfies the KL constraint.

## 24. TRPO algorithm with robot-arm example

Suppose the task is robot-arm grasping.

State:

$$s_t = \text{joint angles, velocities, gripper position, object position}.$$

Action:

$$a_t = \text{motor torque command}.$$

Reward:

$$
r_t
=
-\|x_{\text{gripper},t}-x_{\text{object},t}\|_2
-0.01\|a_t\|^2
+
10\cdot\mathbf 1\{\text{object grasped}\}.
$$

Algorithm:

1. Initialize policy parameters $\theta_0$ and value parameters $\phi_0$.

2. At iteration $k$, collect trajectories using

$$a_t\sim\pi_{\theta_k}(\cdot\mid s_t).$$

3. Observe rollout data:

$$\mathcal D_k = \{(s_t,a_t,r_t,s_{t + 1})\}_{t = 1}^{N}.$$

4. Use the value function to compute TD residuals:

$$\delta_t^V = r_t + \gamma V_{\phi_k}(s_{t + 1})-V_{\phi_k}(s_t).$$

5. Compute GAE advantages:

$$
\widehat A_t
=
\sum_{\ell=0}^{\infty}
(\gamma\lambda)^\ell
\delta_{t+\ell}^V.
$$

6. Define the sampled surrogate:

$$
\widehat L_k(\theta)
=
\frac{1}{N}
\sum_{t=1}^{N}
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_k}(a_t\mid s_t)}
\widehat A_t.
$$

7. Define the average KL constraint:

$$
\widehat{\bar D}_{\mathrm{KL}}(\theta_k,\theta)
=
\frac{1}{N}
\sum_{t=1}^{N}
D_{\mathrm{KL}}
\left(
\pi_{\theta_k}(\cdot\mid s_t)
\;\|\;
\pi_\theta(\cdot\mid s_t)
\right).
$$

8. Approximate the constrained problem locally:

$$\max_{\Delta\theta}g^T\Delta\theta$$

subject to

$$\frac{1}{2}\Delta\theta^TH\Delta\theta\le \delta.$$

9. Use conjugate gradient to solve

$$Hs=g.$$

10. Scale the step:

$$\Delta\theta_{\mathrm{full}} = \sqrt{\frac{2\delta}{s^THs}}\,s.$$

11. Backtrack until both conditions hold:

$$\widehat L_k(\theta_k + \alpha\Delta\theta_{\mathrm{full}}) \ge \widehat L_k(\theta_k)$$

and

$$\widehat{\bar D}_{\mathrm{KL}} (\theta_k,\theta_k + \alpha\Delta\theta_{\mathrm{full}}) \le \delta.$$

12. Accept:

$$\theta_{k + 1} = \theta_k + \alpha\Delta\theta_{\mathrm{full}}.$$

13. Update the value function by regression:

$$
\min_\phi
\frac{1}{N}
\sum_t
\left(
V_\phi(s_t)-\widehat G_t
\right)^2.
$$

For the robot, TRPO makes successful reaching and grasping actions more likely, while using the KL constraint to prevent the motor-control policy from changing too violently between updates.

# CPI and TRPO

## 25. CPI is not “TRPO with temporal weighting”

CPI stands for **Conservative Policy Iteration**.

CPI updates the policy by mixing the old policy with an improved policy:

$$\pi_{\mathrm{new}} = (1-\alpha)\pi_{\mathrm{old}} + \alpha\pi^*.$$

The small mixture coefficient $\alpha$ makes the update conservative.

TRPO generalizes this idea. Instead of explicitly mixing policies, TRPO constrains the distance between the old and new policies:

$$\bar D_{\mathrm{KL}}(\pi_{\mathrm{old}},\pi_{\mathrm{new}})\le \delta.$$

So the connection is:

$$\text{CPI conservatism via mixture coefficient }\alpha$$

becomes

$$\text{TRPO conservatism via KL trust region }\delta.$$

The temporal discounting term

$$d_\pi(s)=\sum_{t=0}^{\infty}\gamma^tP(s_t=s\mid\pi)$$

appears because the objective is discounted return. It is not what distinguishes CPI from TRPO.

# PPO

## 26. PPO core idea

PPO has the same broad goal as TRPO:

> Improve the policy using current data, but do not move the new policy too far from the old policy.

TRPO uses a hard KL-constrained optimization problem. PPO replaces that with a simpler first-order objective.

The most common version is PPO-Clip.

Define the probability ratio

$$\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)} {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.$$

The unclipped objective would be

$$\rho_t(\theta)\widehat A_t.$$

PPO clips the ratio into the interval

$$[1-\varepsilon,1+\varepsilon].$$

The PPO-Clip objective is

$$
L^{\mathrm{CLIP}}(\theta)
=
\widehat{\mathbb E}_t
\left[
\min
\left(
\rho_t(\theta)\widehat A_t,
\operatorname{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)\widehat A_t
\right)
\right].
$$

This objective discourages very large policy updates without requiring TRPO’s conjugate-gradient and line-search machinery.

## 27. PPO clipping intuition

Suppose

$$\varepsilon=0.2.$$

Then PPO clips the probability ratio into

$$[0.8,1.2].$$

### Good action

Suppose the robot moved toward the object, so

$$\widehat A_t=+2.$$

The old policy assigned probability

$$\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)=0.20.$$

The new policy assigns

$$\pi_\theta(a_t\mid s_t)=0.30.$$

Then

$$\rho_t(\theta)=\frac{0.30}{0.20}=1.5.$$

Unclipped:

$$1.5(2)=3.0.$$

Clipped:

$$1.2(2)=2.4.$$

PPO uses

$$\min(3.0,2.4)=2.4.$$

The policy still gets credit for making the good action more likely, but it does not get unlimited extra credit for making a large jump.

### Bad action

Suppose the robot moved away from the object, so

$$\widehat A_t=-2.$$

The old probability was

$$0.20,$$

and the new probability is

$$0.05.$$

Then

$$\rho_t(\theta)=0.25.$$

Unclipped:

$$0.25(-2)=-0.5.$$

Clipped:

$$0.8(-2)=-1.6.$$

PPO uses

$$\min(-0.5,-1.6)=-1.6.$$

Since PPO maximizes the objective, $-1.6$ is worse than $-0.5$. This prevents the objective from rewarding a huge one-step suppression of a bad action too much.

The general rule is:

- If the action was good, PPO limits how much extra reward the objective gives for increasing its probability.
- If the action was bad, PPO limits how much extra reward the objective gives for decreasing its probability.

## 28. PPO algorithm with robot-arm example

1. Initialize policy parameters $\theta_0$ and value parameters $\phi_0$.

2. At iteration $k$, collect rollouts using the current policy:

$$a_t\sim\pi_{\theta_k}(\cdot\mid s_t).$$

3. Store

$$(s_t,a_t,r_t,s_{t+1},\log\pi_{\theta_k}(a_t\mid s_t)).$$

4. Compute TD residuals:

$$\delta_t^V = r_t + \gamma V_{\phi_k}(s_{t + 1})-V_{\phi_k}(s_t).$$

5. Compute GAE advantages:

$$
\widehat A_t
=
\sum_{\ell=0}^{\infty}
(\gamma\lambda)^\ell
\delta_{t+\ell}^V.
$$

6. Compute critic targets:

$$\widehat G_t = \widehat A_t + V_{\phi_k}(s_t).$$

7. For several epochs, sample minibatches from the collected rollout data.

8. Compute the probability ratio:

$$
\rho_t(\theta)
=
\exp
\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\theta_k}(a_t\mid s_t)
\right).
$$

9. Maximize the PPO-Clip objective:

$$
L^{\mathrm{CLIP}}(\theta)
=
\widehat{\mathbb E}_t
\left[
\min
\left(
\rho_t(\theta)\widehat A_t,
\operatorname{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)\widehat A_t
\right)
\right].
$$

In code, this is often implemented as minimizing the negative objective:

$$\mathcal L_{\mathrm{actor}}(\theta) = - L^{\mathrm{CLIP}}(\theta).$$

10. Update the value function by minimizing

$$
\mathcal L_{\mathrm{critic}}(\phi)
=
\widehat{\mathbb E}_t
\left[
\left(
V_\phi(s_t)-\widehat G_t
\right)^2
\right].
$$

11. Optionally monitor KL divergence:

$$
\widehat D_{\mathrm{KL}}
=
\widehat{\mathbb E}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_k}(\cdot\mid s_t)
\;\|\;
\pi_\theta(\cdot\mid s_t)
\right)
\right].
$$

Some PPO implementations stop early if the KL gets too large.

12. Set the updated parameters to

$$\theta_{k + 1}\leftarrow \theta,\qquad \phi_{k + 1}\leftarrow \phi.$$

13. Discard the old batch and collect a fresh batch with the new policy.

PPO is still on-policy. It reuses a fresh batch for multiple minibatch epochs, but it cannot keep using old data indefinitely because the policy changes.

## 29. Entropy bonus in PPO

Many PPO implementations add an entropy bonus to encourage exploration. If the actor is optimized by minimizing a loss, a common form is

$$
\mathcal L_{\mathrm{actor}}(\theta)
=
-
L^{\mathrm{CLIP}}(\theta)
-
\eta\,\widehat{\mathbb E}_t\left[H(\pi_\theta(\cdot\mid s_t))\right].
$$

The entropy term is subtracted from the loss. Because the optimizer minimizes the loss, subtracting entropy rewards higher entropy. This is intentional: higher entropy means the policy remains more stochastic, which encourages exploration and prevents premature collapse to a deterministic policy.

For a robot arm, entropy helps the policy keep trying varied motor commands early in training instead of immediately committing to one motion pattern that may be locally bad.

## 30. TRPO versus PPO side by side

| Concept | TRPO | PPO |
|---|---|---|
| Full name | Trust Region Policy Optimization | Proximal Policy Optimization |
| Main idea | Improve the policy while enforcing a hard trust region | Improve the policy while softly discouraging large updates |
| Update mechanism | Constrained optimization | Unconstrained first-order optimization |
| Trust-region mechanism | Explicit KL-divergence constraint | Clipped probability ratio; sometimes KL early stopping or KL penalty |
| Main objective | Maximize the surrogate subject to a KL bound | Maximize the clipped surrogate objective |
| Optimization method | Natural-gradient-style step, conjugate gradient, step scaling, line search | Minibatch gradient descent or Adam |
| Complexity | More mathematically involved | Simpler and easier to implement |
| Data usage | On-policy | On-policy, but usually reuses each fresh batch for several epochs |
| Value function | Usually uses a learned critic $V_\phi(s)$ | Usually uses a learned critic $V_\phi(s)$ |
| Advantage estimation | Commonly uses GAE, but does not require it | Commonly uses GAE, but does not require it |
| What happens if update is too large? | Line search rejects or shrinks the update | Clipping removes much of the objective incentive for large ratio changes |
| Practical popularity | Strong theoretical motivation but harder to implement | Widely used because it is simpler and robust |

TRPO solves

$$\max_\theta \widehat{\mathbb E}_t\left[\rho_t(\theta)\widehat A_t\right]$$

subject to

$$\widehat{\mathbb E}_t\left[D_{\mathrm{KL}}\left(\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)\|\pi_\theta(\cdot\mid s_t)\right)\right]\le\delta.$$

PPO-Clip solves the unconstrained objective

$$\max_\theta\widehat{\mathbb E}_t\left[\min\left(\rho_t(\theta)\widehat A_t,\operatorname{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)\widehat A_t\right)\right].$$

In one sentence: TRPO enforces a hard KL trust region; PPO approximates the same trust-region idea by clipping the policy probability ratio.

Both TRPO and PPO commonly use GAE, but GAE is not part of the definition of either algorithm. TRPO/PPO are policy-update rules; GAE is a method for estimating $\widehat A_t$, the advantage values used inside those policy-update rules.

# Main misunderstandings and clarifications

## 31. Misunderstanding: “The surrogate approximates $\pi$ with $\pi'$.”

The surrogate does not replace $\pi'$ with $\pi$. The actual approximation is about where the new policy will take us. TRPO replaces

$$d_{\pi'}(s)$$

with

$$d_\pi(s).$$

The new policy $\pi'$ still appears in the action probabilities:

$$\sum_a \pi'(a\mid s)A_\pi(s,a).$$

So the approximation is about state visitation frequencies, not the action policy itself.

## 32. Misunderstanding: “If we already observed the trajectory, why optimize anything?”

The trajectory is training data. It tells us how good or bad past sampled actions were, and the objective uses that information to update the policy for future rollouts. The completed trajectory cannot be changed, but the policy that generates the next trajectory can.

## 33. Misunderstanding: “How can we know $r_{t+1}$ at time $t$?”

We usually do not know it at time $t$. In model-free RL, $r_{t+1}$ is observed later, after taking the next action. Returns and GAE are computed after rollout data are collected, not while the policy is choosing $a_t$.

## 34. Misunderstanding: “The value function is known before RL.”

Usually it is not known beforehand. It is learned during training as a critic. Early in training, $V_\phi(s)$ may be very inaccurate; it improves by fitting observed returns from rollouts.

## 35. Misunderstanding: “The policy is fixed before RL.”

The policy is initialized before RL, often randomly, but it is not fixed. At iteration $k$, $\pi_{\theta_k}$ generates data. The update produces $\pi_{\theta_{k+1}}$, and that new policy generates the next batch.

## 36. Misunderstanding: “CPI is TRPO with temporal weighting.”

CPI is conservative because it mixes old and new policies:

$$\pi_{\mathrm{new}} = (1-\alpha)\pi_{\mathrm{old}} + \alpha\pi^*.$$

TRPO is conservative for a different reason: it constrains KL divergence.

$$\bar D_{\mathrm{KL}}(\pi_{\mathrm{old}},\pi_{\mathrm{new}})\le\delta.$$

Temporal discounting appears in both because the return objective is discounted, but that is not the defining CPI-to-TRPO relationship.

## 37. Misunderstanding: “The symbol $r_t$ always means reward.”

There is a notation collision in RL. Often,

$$r_t$$

means reward.

But in PPO/TRPO derivations, many authors use

$$r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)} {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$$

to mean the probability ratio.

To avoid confusion, these notes used

$$\rho_t(\theta)$$

for the probability ratio.

# One-page mental model

TRPO and PPO both follow this template:

1. Run the current policy and collect trajectories.
2. Use rewards and the value function to compute advantages.
3. Use advantages to tell whether sampled actions were better or worse than expected.
4. Update the policy to make good sampled actions more likely and bad sampled actions less likely.
5. Prevent the policy update from being too large.

TRPO prevents large updates with a KL constraint:

$$\bar D_{\mathrm{KL}}(\pi_{\mathrm{old}},\pi_{\mathrm{new}})\le\delta.$$

PPO prevents large updates with clipping:

$$\operatorname{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon).$$

The value function is a helper. It predicts expected future return so that the algorithm can estimate whether an action was better or worse than expected.

GAE is a helper for the helper. It produces a stable advantage estimate by combining one-step and multi-step TD errors.

The rollout is not the thing being optimized. The rollout is the dataset. The policy is what is optimized for future behavior.

# Minimal Python demo: PPO and TRPO on a discrete-action environment

The following code is intentionally small and educational. It uses PyTorch and Gymnasium on a discrete-action task such as CartPole. PPO is relatively short because it uses ordinary gradient descent. TRPO is longer because it needs a KL Hessian-vector product, conjugate gradient, step scaling, and line search.

```python
# pip install torch gymnasium

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def mlp(in_dim, out_dim, hidden=64):
    """Small neural network used for both the policy and value function."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )

class ActorCritic(nn.Module):
    """Actor outputs an action distribution; critic outputs a scalar value."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.pi = mlp(obs_dim, act_dim)   # policy logits for discrete actions
        self.v = mlp(obs_dim, 1)          # value estimate V_phi(s)

    def dist(self, obs):
        logits = self.pi(obs)
        return Categorical(logits=logits), logits

    def value(self, obs):
        return self.v(obs).squeeze(-1)

def collect_rollout(env, ac, steps=2048, gamma=0.99, lam=0.95):
    """Collect on-policy data and compute GAE advantages."""
    obs, _ = env.reset()
    obs_buf, act_buf, rew_buf, next_obs_buf, done_buf = [], [], [], [], []
    logp_buf, val_buf, logits_buf = [], [], []
    ep_returns, ep_ret = [], 0.0

    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, logits = ac.dist(obs_t)
            action = dist.sample()
            logp = dist.log_prob(action)
            value = ac.value(obs_t)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_buf.append(obs.copy())
        act_buf.append(action.item())
        rew_buf.append(reward)
        next_obs_buf.append(next_obs.copy())
        done_buf.append(float(done))
        logp_buf.append(logp.item())
        val_buf.append(value.item())
        logits_buf.append(logits.squeeze(0).numpy())

        ep_ret += reward
        if done:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    obs = torch.as_tensor(np.array(obs_buf), dtype=torch.float32)
    next_obs = torch.as_tensor(np.array(next_obs_buf), dtype=torch.float32)
    acts = torch.as_tensor(act_buf, dtype=torch.long)
    rews = torch.as_tensor(rew_buf, dtype=torch.float32)
    dones = torch.as_tensor(done_buf, dtype=torch.float32)
    old_logp = torch.as_tensor(logp_buf, dtype=torch.float32)
    vals = torch.as_tensor(val_buf, dtype=torch.float32)
    old_logits = torch.as_tensor(np.array(logits_buf), dtype=torch.float32)

    with torch.no_grad():
        next_vals = ac.value(next_obs)

    # One-step TD residual: delta_t = r_t + gamma V(s_{t+1}) - V(s_t).
    deltas = rews + gamma * (1.0 - dones) * next_vals - vals

    # GAE is computed backward, but mathematically it is a forward-looking sum.
    adv = torch.zeros_like(rews)
    last_adv = 0.0
    for t in reversed(range(steps)):
        last_adv = deltas[t] + gamma * lam * (1.0 - dones[t]) * last_adv
        adv[t] = last_adv

    returns = adv + vals
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalization stabilizes training

    return {
        "obs": obs,
        "acts": acts,
        "old_logp": old_logp,
        "old_logits": old_logits,
        "adv": adv.detach(),
        "returns": returns.detach(),
        "ep_returns": ep_returns,
    }

def ppo_update(ac, data, pi_optimizer, v_optimizer, clip_eps=0.2, epochs=10, minibatch_size=64, ent_coef=0.01):
    """PPO update: clipped policy loss plus value regression."""
    obs, acts = data["obs"], data["acts"]
    old_logp, adv, returns = data["old_logp"], data["adv"], data["returns"]
    n = obs.shape[0]
    indices = np.arange(n)

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            mb = torch.as_tensor(indices[start:start + minibatch_size])
            dist, _ = ac.dist(obs[mb])
            logp = dist.log_prob(acts[mb])
            ratio = torch.exp(logp - old_logp[mb])

            unclipped = ratio * adv[mb]
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()

            # Entropy is subtracted from the loss, so higher entropy is rewarded.
            entropy_bonus = dist.entropy().mean()
            actor_loss = policy_loss - ent_coef * entropy_bonus

            pi_optimizer.zero_grad()
            actor_loss.backward()
            pi_optimizer.step()

            value_loss = ((ac.value(obs[mb]) - returns[mb]) ** 2).mean()
            v_optimizer.zero_grad()
            value_loss.backward()
            v_optimizer.step()

def flat_grad(y, params, retain_graph=False, create_graph=False):
    """Flatten gradients into one vector."""
    grads = torch.autograd.grad(y, params, retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grads])

def conjugate_gradient(Avp, b, nsteps=10, residual_tol=1e-10):
    """Approximately solve Ax=b using only matrix-vector products Avp(v)."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_old = torch.dot(r, r)
    for _ in range(nsteps):
        Avp_p = Avp(p)
        alpha = r_dot_old / (torch.dot(p, Avp_p) + 1e-8)
        x += alpha * p
        r -= alpha * Avp_p
        r_dot_new = torch.dot(r, r)
        if r_dot_new < residual_tol:
            break
        beta = r_dot_new / (r_dot_old + 1e-8)
        p = r + beta * p
        r_dot_old = r_dot_new
    return x

def trpo_update(ac, data, v_optimizer, max_kl=0.01, damping=0.1, cg_iters=10, backtrack_iters=10, backtrack_coeff=0.5, vf_iters=10):
    """TRPO update: natural-gradient step constrained by average KL."""
    obs, acts = data["obs"], data["acts"]
    old_logp, old_logits = data["old_logp"], data["old_logits"]
    adv, returns = data["adv"], data["returns"]
    policy_params = list(ac.pi.parameters())

    def surrogate_and_kl():
        dist, _ = ac.dist(obs)
        logp = dist.log_prob(acts)
        ratio = torch.exp(logp - old_logp)
        surrogate = (ratio * adv).mean()
        old_dist = Categorical(logits=old_logits)
        kl = kl_divergence(old_dist, dist).mean()
        return surrogate, kl

    # Policy gradient g of the surrogate objective.
    surrogate, _ = surrogate_and_kl()
    g = flat_grad(surrogate, policy_params).detach()

    # Hessian-vector product for the KL curvature matrix H.
    def hvp(v):
        _, kl = surrogate_and_kl()
        grad_kl = flat_grad(kl, policy_params, create_graph=True, retain_graph=True)
        grad_kl_v = (grad_kl * v).sum()
        hess_v = flat_grad(grad_kl_v, policy_params, retain_graph=True).detach()
        return hess_v + damping * v

    # Natural-gradient direction: solve Hs = g.
    step_dir = conjugate_gradient(hvp, g, nsteps=cg_iters)
    shs = 0.5 * torch.dot(step_dir, hvp(step_dir))
    if shs <= 0:
        return

    # Scale step so the quadratic KL approximation is within max_kl.
    full_step = torch.sqrt(max_kl / (shs + 1e-8)) * step_dir
    old_params = parameters_to_vector(policy_params).detach()
    old_surrogate, _ = surrogate_and_kl()
    old_surrogate = old_surrogate.detach()

    # Backtracking line search checks actual improvement and actual KL.
    accepted = False
    for i in range(backtrack_iters):
        frac = backtrack_coeff ** i
        vector_to_parameters(old_params + frac * full_step, policy_params)
        new_surrogate, new_kl = surrogate_and_kl()
        if new_surrogate.item() >= old_surrogate.item() and new_kl.item() <= max_kl:
            accepted = True
            break
    if not accepted:
        vector_to_parameters(old_params, policy_params)

    # Critic update by supervised regression to return targets.
    for _ in range(vf_iters):
        value_loss = ((ac.value(obs) - returns) ** 2).mean()
        v_optimizer.zero_grad()
        value_loss.backward()
        v_optimizer.step()

def train(algo="ppo", env_id="CartPole-v1", iterations=50):
    env = gym.make(env_id)
    assert isinstance(env.action_space, gym.spaces.Discrete), "Demo supports discrete actions only."
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    ac = ActorCritic(obs_dim, act_dim)
    pi_optimizer = optim.Adam(ac.pi.parameters(), lr=3e-4)
    v_optimizer = optim.Adam(ac.v.parameters(), lr=1e-3)

    for k in range(iterations):
        data = collect_rollout(env, ac, steps=2048)
        if algo == "ppo":
            ppo_update(ac, data, pi_optimizer, v_optimizer)
        elif algo == "trpo":
            trpo_update(ac, data, v_optimizer)
        else:
            raise ValueError("algo must be 'ppo' or 'trpo'")
        avg_ret = np.mean(data["ep_returns"]) if data["ep_returns"] else float("nan")
        print(f"{algo.upper()} iter {k:03d} | avg return: {avg_ret:.1f}")

if __name__ == "__main__":
    train(algo="ppo", env_id="CartPole-v1", iterations=50)
    # To try TRPO instead, comment the line above and uncomment this one:
    # train(algo="trpo", env_id="CartPole-v1", iterations=50)
```

# References

- OmniSafe TRPO tutorial: <https://omnisafe.readthedocs.io/en/latest/baserl/trpo.html>
- OmniSafe PPO tutorial: <https://omnisafe.readthedocs.io/en/latest/baserl/ppo.html>
- Schulman et al., “Trust Region Policy Optimization,” 2015: <https://arxiv.org/abs/1502.05477>
- Schulman et al., “Proximal Policy Optimization Algorithms,” 2017: <https://arxiv.org/abs/1707.06347>
- Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation,” 2015: <https://arxiv.org/abs/1506.02438>
