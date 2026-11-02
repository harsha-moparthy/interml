from author import write
write(r'''
@@ 792|prml scipy calibration metrics islp
Apply Bayes' rule using prevalence, not just sensitivity and specificity. Here $\pi=0.0005$, sensitivity is 0.99, and false-positive rate is 0.01. [1]

$$P(F\mid +)=\frac{0.99\cdot0.0005}{0.99\cdot0.0005+0.01\cdot0.9995}=0.0471878\approx4.72\%.$$

Among 100,000 transactions, expect 50 fraudulent transactions, 49.5 true alerts, and 999.5 false alerts. Fractional counts represent expectations over repeated populations. False alerts dominate because legitimate transactions are so numerous. This is the base-rate effect, not a contradiction of 99% sensitivity/specificity. Changing the alert threshold changes both rates; choose it using investigation capacity and error costs, and evaluate on representative prevalence.
@@ 793|prml scipy islp pml2 linear
Probability treats parameters as fixed and possible data as variable. Likelihood treats the observed data as fixed and compares parameter values; it is not automatically a normalized distribution over parameters. [1]

For Bernoulli observations with $s$ successes, $\ell(p)=s\log p+(N-s)\log(1-p)$, so setting its derivative to zero gives $\hat p=s/N$, including boundary solutions. For iid Gaussian observations,

$$\hat\mu=\bar x,\qquad \hat\sigma^2_{MLE}=\frac1N\sum_i(x_i-\bar x)^2.$$

These follow by differentiating the Gaussian log likelihood with respect to mean and variance. Independence and distributional assumptions matter; dependent samples or a misspecified likelihood change the estimator's interpretation and uncertainty. A prior is needed to turn likelihood information into a Bayesian posterior.
@@ 794|prml scipy islp linear pml2
Estimating the mean from the same sample uses one degree of freedom. The identity

$$\sum_i(x_i-\bar x)^2=\sum_i(x_i-\mu)^2-N(\bar x-\mu)^2$$

gives expectation $N\sigma^2-N\operatorname{Var}(\bar x)=(N-1)\sigma^2$ for iid observations with finite variance. Therefore dividing by $N$ has expectation $(N-1)\sigma^2/N$, while dividing by $N-1$ is unbiased. Gaussianity is needed for the stated Gaussian MLE derivation, but not this unbiasedness calculation. [1]

Unbiased variance does not imply unbiased standard deviation after taking a square root. For dependent or weighted samples, the simple degrees-of-freedom correction may not apply. Also, minimum mean-squared error and unbiasedness are different estimator-selection criteria.
@@ 795|prml pml2 islp reg scipy
MLE maximizes the data likelihood. MAP maximizes likelihood times prior, equivalently log likelihood plus log prior. A Gaussian coefficient prior yields quadratic regularization; a Laplace prior yields L1 regularization under the corresponding likelihood. [1]

$$\hat\theta_{MAP}=\arg\max_\theta\left[\sum_i\log p(x_i\mid\theta)+\log p(\theta)\right].$$

With a fixed regular prior, identifiable correctly specified model, and enough information, the likelihood usually dominates as $N$ grows and MLE/MAP approach the same limit. This is not unconditional: nonidentifiability, singular models, hard prior exclusions, growing dimension, or an $N$-dependent penalty can preserve differences. MAP is a point estimate and omits posterior uncertainty; it also depends on parameterization through the prior density.
@@ 796|prml dl pml2 metrics scipy
For discrete distributions on compatible support,

$$H(P)=-\sum_xP(x)\log P(x),\quad H(P,Q)=-\sum_xP(x)\log Q(x),$$

$$D_{KL}(P\|Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}=H(P,Q)-H(P).$$

If $P$ is fixed, its entropy does not depend on model parameters, so minimizing cross-entropy over $Q_\theta$ is equivalent to minimizing forward KL. [1] A zero model probability where $P$ has mass gives infinite loss. Log base determines units: natural logs give nats, base two bits. Empirical one-hot training labels approximate this population objective, but limited/noisy data and model misspecification still affect what is learned. Continuous differential entropy has additional coordinate-dependence caveats.
@@ 797|prml pml2 dl ddpm distill
KL weights discrepancies using its **first** argument. Forward KL $D_{KL}(P\|Q)$ heavily penalizes putting too little $Q$ mass wherever data distribution $P$ has mass. Reverse KL samples from $Q$ and strongly penalizes placing model mass in low-density regions of $P$. [1]

For a two-mode target approximated by one Gaussian, forward KL may cover both modes with a broad distribution; reverse KL may choose one mode rather than put mass in the low-density gap. This is a useful tendency under restricted model families, not a universal theorem that every reverse-KL optimum drops modes.

The support conditions differ, and neither direction is a distance metric. In distillation, state which distribution is teacher/student and which direction is optimized before interpreting coverage or sharpness.
@@ 798|prml scipy pml2 islp forecast
The law of large numbers concerns convergence of the sample average to the expectation under suitable assumptions. The classical iid CLT states that finite-variance averages, after centering and scaling, converge in distribution to a normal law. [1]

$$\sqrt N\frac{\bar X-\mu}{\sigma}\Rightarrow\mathcal N(0,1).$$

LLN does not say the finite-sample error is Gaussian or provide its rate. A Cauchy distribution has no finite mean/variance; averaging iid Cauchy observations remains Cauchy, so this CLT does not apply. Some Pareto distributions have finite mean but infinite variance: an LLN can hold while the classical normal CLT fails. Dependence also requires additional conditions. Diagnose tail behavior and dependence before using standard-error formulas or treating observations as independent.
@@ 799|islp prml ensemble reg metrics
At fixed $x$, write $y=f(x)+\epsilon$ with zero conditional noise mean and variance $\sigma_\epsilon^2$. Let $m=E_D[\hat f_D(x)]$ over training datasets. Add and subtract $m$ inside the squared prediction error. [1]

$$E[(y-\hat f)^2]=(f-m)^2+E[(\hat f-m)^2]+\sigma_\epsilon^2.$$

The cross terms vanish because centered estimator fluctuations and test noise have zero expectation, assuming the test noise is independent of training. The terms are squared bias, variance, and irreducible noise. This decomposition is for squared loss at a specified distribution; it does not directly decompose classification accuracy. More flexible models can change both bias and variance, and optimization/regularization determines which function is actually fitted.
@@ 800|prml pca linear scipy dl
An eigenvector identifies a direction preserved by a linear map: $Av=\lambda v$. The eigenvalue describes scaling and possible sign reversal along that direction. A real symmetric matrix has an orthonormal eigenbasis by the spectral theorem. [1]

For eigenpairs $(\lambda,u),(\mu,v)$, symmetry gives $u^TAv=(Au)^Tv$, hence $(\mu-\lambda)u^Tv=0$. Distinct eigenvalues therefore have orthogonal eigenvectors; within a repeated-eigenvalue subspace one can choose an orthonormal basis. Eigenvalues are real because a symmetric real matrix is Hermitian.

Covariance eigenvectors describe variance directions and have nonnegative eigenvalues. Hessians are symmetric under appropriate smoothness but can have negative eigenvalues at nonconvex points. Symmetric does not mean positive definite.
@@ 801|pca prml scipy linear islp
SVD writes an $m\times n$ matrix as $U\Sigma V^T$ with orthonormal singular vectors and nonnegative singular values. It exists even for rectangular or nonsymmetric matrices. Keeping the top $k$ singular terms gives the best rank-$k$ approximation in Frobenius norm and spectral norm. [1]

$$A_k=\sum_{i=1}^k\sigma_i u_iv_i^T,\qquad \|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2.$$

For a centered data matrix, right singular vectors are PCA directions and scores are $U\Sigma$. Covariance eigenvalues are $\sigma_i^2/(N-1)$ under sample-covariance normalization. Centering and scaling choices change PCA. Repeated singular values can make individual basis vectors nonunique even when the retained subspace is well defined.
@@ 802|prml svm pca linear scipy
For a real symmetric matrix, PSD means every quadratic form $a^TAa$ is nonnegative, equivalently all eigenvalues are nonnegative. Covariance is PSD because [1]

$$a^T\operatorname{Cov}(X)a=\operatorname{Var}(a^TX)\ge0.$$

A Gram matrix from feature vectors is PSD since $\sum_{ij}c_ic_j\langle\phi_i,\phi_j\rangle=\|\sum_i c_i\phi_i\|^2$. A valid kernel must have this property for every finite input set, not just one sampled matrix. Small negative computed eigenvalues may reflect rounding; large ones can indicate an invalid kernel or covariance construction. PSD may be singular, so ordinary inversion can fail; regularization or a pseudoinverse may be appropriate depending on the model.
@@ 803|optim prml torch scipy dl
A Jacobian contains first derivatives of a vector-valued function: for $f:\mathbb R^n\to\mathbb R^m$, it is $m\times n$. A Hessian contains second derivatives of a scalar function and is $n\times n$. [1]

Newton solves $H\Delta=-g$ and updates parameters by $\Delta$. Storing a dense Hessian costs $O(n^2)$ and a generic factorization $O(n^3)$, which is prohibitive for large networks. Nonconvex Hessians can be indefinite, and minibatch curvature is noisy.

Hessian-vector products avoid forming the full matrix and support truncated Newton or diagnostics; quasi-Newton and structured approximations trade accuracy for cost. First-order optimizers remain attractive because gradients are cheap via reverse-mode differentiation and scale well with distributed minibatches.
@@ 804|optim linear prml scipy pca
For a positive-definite quadratic with Hessian $H$, different eigenvalues determine curvature along different directions. A step stable in the steepest direction may be tiny relative to what the flattest direction needs. [1]

$$\kappa(H)=\lambda_{max}/\lambda_{min},\qquad 0<\eta<2/\lambda_{max}.$$

This causes slow progress and oscillation across a narrow valley. For a general matrix, the 2-norm condition number is the singular-value ratio; an indefinite Hessian needs additional care because negative curvature invalidates the simple convex convergence picture. Standardization, preconditioning, momentum, and adaptive methods can improve effective geometry. A singular Hessian has infinite condition number under this definition, though the optimization problem may still have useful flat directions rather than a unique minimizer.
@@ 805|scipy prml islp experiments causal
A p-value is the probability, **assuming the null model and test procedure**, of a statistic at least as extreme as observed. It is not the probability that the null is true. Power is the probability of rejecting the null under a specified alternative effect, $1-\beta$. [1]

A p-value of 0.04 says neither “96% chance the treatment works” nor that the effect is practically important. Report effect size, uncertainty interval, design, sample size, and relevant assumptions. Power depends on the alternative, noise, allocation, and significance threshold; it is not a fixed property of a metric alone. Repeated looks, selective reporting, and multiple comparisons can invalidate an unadjusted interpretation.
@@ 806|scipy experiments cuped prml metrics
For equal-sized independent arms and a binary metric near baseline $p$, a common normal approximation for per-arm size is [1]

$$n\approx\frac{2p(1-p)(z_{1-\alpha/2}+z_{1-\beta})^2}{\delta^2}.$$

For $p=0.10$, absolute MDE $\delta=0.01$, two-sided $\alpha=0.05$, and 80% power, $n\approx14{,}112$ per arm using 1.96 and 0.84. A relative 1% lift would instead mean $\delta=0.001$ and about 100 times more data. Duration is required eligible independent units divided by daily enrollment, with complete business cycles and outcome-delay windows added. Clustered users, repeated events, attrition, unequal allocation, and multiple testing require modified calculations.
@@ 807|sequential scipy prml cuped causal
A fixed-horizon p-value controls error for one prespecified analysis, not for stopping whenever repeated checks look favorable. Noise eventually creates apparently significant excursions more often than the nominal threshold permits. [1]

Choose a fixed sample/time horizon, a prespecified group-sequential alpha-spending design, or an always-valid procedure such as a valid mixture likelihood-ratio/e-process. Under its null assumptions, a nonnegative test supermartingale can bound the probability of ever crossing $1/\alpha$.

These methods do not make arbitrary peeking harmless: the statistic, treatment assignment, dependence assumptions, and stopping rule must match the guarantee. Report effect estimates and time-uniform intervals where appropriate. Operational safety monitoring can continue even when efficacy decisions follow a fixed-horizon plan.
@@ 808|scipy experiments prml metrics islp
With 20 independent true null tests at 0.05, the chance of at least one false rejection is $1-0.95^{20}\approx64.2\%$. Bonferroni tests each of $m$ hypotheses at $\alpha/m$, controlling family-wise error even without independence. [1]

Benjamini–Hochberg sorts p-values and finds the largest $k$ satisfying $p_{(k)}\le kq/m$, then rejects through $k$. Under its required dependence conditions, it controls expected false-discovery proportion rather than probability of any false discovery.

Define the family before inspecting results. Correlated metrics, exploratory slices, and selective reporting complicate guarantees; use appropriate dependence-robust methods when necessary. Pick control according to decision cost: a safety-critical confirmatory claim may require stronger error protection than an exploratory discovery screen.
@@ 809|cuped experiments scipy prml linear
CUPED adjusts the outcome using a pre-treatment covariate correlated with it. A centered adjustment preserves the expected treatment contrast under randomization. [1]

$$Y_{adj}=Y-\theta(X-E[X]),\quad \theta=\frac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)},\quad \operatorname{Var}(Y_{adj})=\operatorname{Var}(Y)(1-\rho^2).$$

If $\rho^2=0.5$, ideal variance and required sample size are roughly halved; “half the time” is conditional, not universal. Do not use a covariate caused by treatment, which can remove part of the effect or create bias. Handle new users with no history explicitly and estimate adjustment coefficients without leaking treatment-specific information. Validate randomization, metric definitions, missingness, and confidence intervals after adjustment.
@@ 810|causal experiments rl prml scipy
Interference means one user's outcome depends on others' treatment assignments. A treated driver or feed user can change the environment experienced by control users, so an individual-level contrast may not estimate the all-treated versus all-control policy effect. [1]

Graph clusters reduce cross-arm edges; geographic splits separate markets but provide fewer independent units; switchbacks alternate treatment over time in a market, requiring washout and control of temporal effects. Each trades bias, variance, and operational feasibility.

Define the exposure and estimand first. Analyze at the randomization unit, account for within-cluster dependence, and test carryover/spillover sensitivity. Increasing the number of logged events does not compensate for having very few independent markets. Cluster boundaries and time blocks must reflect how interference actually propagates.
@@ 811|experiments causal scipy prml metrics
Novelty effects are temporary changes caused by something being new; primacy or learned-habit effects occur because established users are adapted to the old experience. An early treatment effect may therefore differ from the steady-state effect. [1]

Track effects by time since exposure and pre-treatment tenure, using prespecified cohorts. New-user cohorts reduce legacy-habit effects but may not represent existing users. Long-term randomized holdouts measure persistence but face attrition, crossovers, and evolving product context.

Do not compare an arbitrary late treatment cohort to an early control cohort: calendar changes confound the comparison. Retain contemporaneous randomization, inspect engagement and retention jointly, and define whether the business decision concerns launch impact, equilibrium behavior, or both. Stopping at the most favorable day creates selection bias.
@@ 812|experiments causal scipy prml metrics
Simpson's paradox occurs when aggregated comparisons reverse within-group comparisons because group composition differs. Randomization and prespecified covariate analysis help distinguish real effect heterogeneity from compositional artifacts. [1]

SRM tests whether observed assignment counts match intended probabilities. For 502,100 versus 497,900 under an intended 50/50 split of one million, the Pearson statistic is

$$\chi^2=2\frac{2100^2}{500000}=17.64,$$

strong evidence against the simple allocation model. Possible causes include logging loss, eligibility bugs, redirects, or treatment-dependent exclusion. SRM is a diagnostic alarm, not automatic proof of one particular bug. Pause causal interpretation until its cause is understood; deleting inconvenient observations to restore 50/50 can create further bias. Check assignment counts before outcome-based filtering.
@@ 813|causal experiments prml scipy linear
DiD compares changes between treated and comparison groups and requires a credible parallel-trends counterfactual. Propensity matching seeks comparable observed covariates but cannot remove unmeasured confounding. Synthetic control builds a weighted donor counterfactual, relying on pre-treatment fit and suitable unaffected donors. [1]

RDD identifies a local effect near a cutoff when potential outcomes vary smoothly and assignment is not precisely manipulated. IV uses variation affecting treatment but not outcomes except through treatment, plus relevance and additional assumptions; it often identifies a local effect for compliers.

Choose by the actual assignment mechanism, not which estimator is easiest. Run placebo/pretrend/sensitivity checks, define the population and effect, and distinguish lack of evidence against an assumption from proof it holds. None automatically recreates randomization from arbitrary observational logs.
''')
