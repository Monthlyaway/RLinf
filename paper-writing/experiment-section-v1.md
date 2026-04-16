### 5. Experiments

This section evaluates whether the proposed structure-regularized potential can close the gap between offline preference fitting and stable online reinforcement learning. Our experiments are designed around three questions. First, does a pure Bradley-Terry objective already exhibit the monotonicity trap in practice, namely high offline ranking fidelity but unstable downstream policy optimization? Second, can the proposed structural prior reduce cross-seed variance and improve sample efficiency without relying on privileged progress labels? Third, in a practical setting where no human frame-level annotation is available, can automatically constructed trajectory preferences recover a reward signal that approaches an engineer-usable shaping baseline?

#### 5.1 Experimental Questions

We center the evaluation around the following three questions:

1. **Offline-online mismatch.** Can a BT-only potential achieve strong held-out pairwise ranking performance while still inducing poor or highly unstable online RL dynamics?
2. **Variance reduction and efficiency.** Does adding the structural prior improve sample efficiency and reduce cross-seed variance under the same downstream RL budget?
3. **Practicality without dense labels.** Can automatically generated trajectory preferences, without any human progress annotation, recover a shaping signal that narrows the gap to a practical dense-reward baseline?

These questions directly target the main claim of the paper: the core failure mode is not inaccurate ordinal supervision per se, but the lack of a well-behaved cardinal increment structure for PBRS deployment.

#### 5.2 Benchmarks and Tasks

We instantiate the evaluation on four representative MetaWorld v3 tasks:

- `button-press-v3`
- `drawer-open-v3`
- `sweep-into-v3`
- `hammer-v3`

These tasks were selected to cover complementary manipulation patterns: contact-rich actuation, articulated object interaction, long-horizon transport, and tool-mediated control. This diversity is sufficient to test whether the proposed method improves reward geometry rather than overfitting to a single interaction pattern.

To keep the experimental interpretation clean, we treat each task as an independent problem. Preference construction, offline potential learning, normalization, and downstream RL are all performed separately for each task. This avoids conflating the contribution of the reward model with multi-task representation learning or task transfer effects.

We intentionally focus the first experimental version on a single simulator family. The goal of the paper is to diagnose and repair the reward-structure failure induced by trajectory ranking, not to maximize cross-simulator coverage. Using MetaWorld as the main benchmark allows us to stress the central hypothesis under a controlled and engineering-feasible setup.

#### 5.3 Automatic Preference Construction Without Human Labels

All preference supervision is generated automatically from collected trajectories; no human frame-level progress labels or privileged oracle distances are used. For each task, we first collect a mixed-quality trajectory pool using a practical shaping policy obtained from an online PPO collector. This produces both successful and failed episodes, as well as successful trajectories with varying path efficiency.

From this pool, we construct three types of pairwise comparisons:

1. **Outcome pairs (`success > failure`).** A successful trajectory is always ranked above a failed trajectory from the same task.
2. **Efficiency pairs (`shorter success > longer success`).** Among successful trajectories, shorter episodes are preferred over longer ones whenever the difference in horizon exceeds a fixed threshold. This encourages the model to prefer efficient progress rather than merely eventual completion.
3. **Temporal segment pairs (`later segment > earlier segment`).** Within a successful trajectory, a later segment is ranked above an earlier segment of comparable length. These pairs provide a weak but scalable notion of local temporal directionality without introducing absolute progress targets.

The resulting supervision remains strictly preference-based. Importantly, the temporal segment pairs do not tell the model what the absolute progress value should be at any state; they only constrain relative ordering within a trajectory. This makes them substantially cheaper than dense progress annotation while still providing a useful inductive signal for the offline potential learner.

#### 5.4 Compared Methods

We compare four training conditions:

- **Sparse.** The downstream RL agent receives only the sparse task completion reward from the environment.
- **Practical Shaping.** The agent is trained with the native MetaWorld dense reward exposed by the simulator. This serves as an engineering baseline rather than a label-free preference method.
- **BT-only Potential + PBRS.** A task-specific potential is learned from automatic preference pairs using only the Bradley-Terry ranking loss, and then deployed through frozen PBRS during online RL.
- **BT+Struct Potential + PBRS.** The same offline preference learner is augmented with the proposed structural prior before PBRS deployment.

The comparison between the last two conditions isolates the effect of the structural regularizer. The comparison against `Sparse` and `Practical Shaping` locates the proposed method between an annotation-free but weak baseline and a stronger engineering baseline that depends on simulator-side dense shaping.

We do not place Oracle-style progress supervision in the main table of this first experimental version. Such models are useful as optional upper bounds, but they depend on privileged information that is unavailable in the intended practical setting.

#### 5.5 Offline Reward Learning and Online RL Protocol

For offline reward learning, we train one small state-based potential network per task. The network takes the low-dimensional state observation as input and predicts a scalar potential. This choice intentionally removes representation learning as a confounder and lets the experiments focus on whether structural regularization improves the learned reward geometry.

The potential is trained on the automatically constructed pair dataset using either the BT loss alone or the joint `BT + L_struct` objective. After convergence, we freeze the reward model and export an affine normalization based on successful trajectory endpoints. The normalized potential is then deployed via PBRS in the downstream RL loop.

For downstream control, PPO is used as the primary online RL algorithm. SAC is included as a secondary robustness check to verify that the observed effect is not a PPO-specific artifact. We use three random seeds for each downstream RL condition. To keep the analysis focused on online optimization stability, the offline potential is trained once per task and then reused across all downstream seeds rather than retrained independently for each seed.

#### 5.6 Metrics

We report both offline reward-quality metrics and online control metrics.

For offline evaluation, we measure:

- held-out pair accuracy,
- held-out Kendall-\(\tau\) or its pairwise proxy,
- increment entropy / concentration statistics to quantify reward curvature degeneracy.

For downstream RL, we measure:

- final success rate,
- area under the learning curve (AUC),
- sample efficiency at fixed success thresholds,
- cross-seed standard deviation of success rate and return.

This separation is important. Our central claim is precisely that offline ranking metrics alone are insufficient to predict whether a learned potential is safe to deploy through PBRS.

#### 5.7 Main Results

The main quantitative results are summarized in [Table 1] and [Figure 3 here]. The table reports held-out pairwise ranking quality together with downstream control performance on all four tasks. The figure shows mean learning curves with standard deviation bands over three seeds.

We expect the first key pattern to be an explicit offline-online mismatch: `BT-only Potential + PBRS` can achieve competitive held-out ranking accuracy, e.g., `[XX.X]%`, and a strong pairwise correlation score, e.g., `[0.XX]`, while still exhibiting unstable or highly variable online performance. In contrast, `BT+Struct Potential + PBRS` is expected to preserve comparable offline ranking quality while improving final success and reducing seed-wise collapse.

The second key pattern is the relationship to the engineering baseline. On some tasks, `Practical Shaping` may still remain the strongest condition in absolute sample efficiency, but the gap between `BT+Struct` and `Practical Shaping` should be substantially smaller than the gap between `BT-only` and `Practical Shaping`. This would support the claim that most of the missing ingredient is not additional privileged labeling, but a better-behaved increment structure.

The third key pattern concerns stability. Across tasks, we expect `BT+Struct` to reduce cross-seed standard deviation from `[XX.X]` to `[YY.Y]`, and to replace early-learning failures with smoother, more repeatable improvement. This is the most direct empirical signature of escaping the monotonicity trap.

#### 5.8 Diagnostics and Ablations

We include two classes of diagnostic experiments.

**BT-only vs. BT+Struct.** The first diagnostic compares the potential curvature induced by the two objectives. In [Figure 4 here], we visualize representative potential trajectories and the distribution of per-step increments on held-out successful rollouts. We expect BT-only models to produce overly concentrated or erratic increment profiles, whereas the proposed model should allocate progress more evenly while still preserving task-relevant spikes.

**Sensitivity to \(\lambda_{struct}\).** The second diagnostic varies the structural weight \(\lambda_{struct}\). If the coefficient is too small, the model should revert toward the BT-only regime and recover the same instability. If it is too large, ranking fidelity may deteriorate because the learner is over-regularized toward uniformity. We therefore expect a broad but non-trivial middle range in which the structural prior stabilizes PBRS without erasing informative preference gradients.

These ablations are complemented by the offline-good / online-bad narrative analysis. For runs where BT-only achieves strong held-out pair accuracy but poor online control, we inspect the learned increment statistics and the early-stage RL reward traces. This analysis is meant to make the monotonicity trap concrete rather than merely correlational.

#### 5.9 Practical Notes

The first-version experimental protocol is intentionally conservative and closely matched to the current RLinf engineering stack. MetaWorld is already supported in the repository; trajectory collection can be performed with existing episode-export utilities; and downstream PPO/SAC training is already available for state-based embodied agents. The new components introduced for this paper are therefore limited to: (i) automatic preference construction from collected trajectories, (ii) offline potential training under BT or `BT + L_struct`, and (iii) a frozen PBRS reward relabel wrapper during downstream RL.

This design keeps the experimental burden aligned with the actual deadline and hardware constraints while preserving the core scientific question. Optional extensions, such as Oracle-lite upper bounds, ManiSkill portability checks, or pretrained VLA case studies, are left outside the first main text and can be added later if time allows.
