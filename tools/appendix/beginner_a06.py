from expand import write
write(r'''
@@ 792
**The detector's accuracy rates and the probability of fraud after an alert are different conditional probabilities.** Sensitivity asks how often fraud triggers an alert. The question asks how often an alert came from fraud. Reversing that condition requires the base rate. [1]

**Imagine 100,000 transactions.** At 0.05% prevalence, only 50 are fraudulent and 99,950 are legitimate. A 99% sensitive detector catches an expected 49.5 fraudulent transactions. A 99% specific detector falsely alerts on 1% of legitimate transactions: an expected 999.5 false alerts.

There are therefore 1049 expected alerts, of which only 49.5 are true fraud. Dividing gives about 4.72%. The fractional counts are expectations over repeated populations; they are a convenient way to calculate proportions, not a claim that one actual transaction can be half fraudulent.

**Why the result is not paradoxical.** The detector makes errors on a small fraction of a huge legitimate population. That small fraction still outnumbers the true alerts from a tiny fraud population. A test can have impressive sensitivity and specificity yet low positive predictive value when the target event is rare.

Bayes' formula below expresses exactly this counting argument. For decisions, also consider investigation capacity and the costs of missed fraud versus false alerts. Changing the threshold changes sensitivity and specificity together; a quoted “99%” performance figure alone is insufficient to choose a useful operating policy.
@@ 793
**Probability varies the outcome; likelihood varies the explanation of a fixed outcome.** With a coin bias p fixed, probability asks how likely a sequence of flips is. After observing the sequence, likelihood treats those flips as fixed and compares candidate values of p. The same formula appears, but the variable and interpretation change. [1]

For N Bernoulli observations with s successes, likelihood is proportional to $p^s(1-p)^{N-s}$. Logs turn products into sums: $\ell(p)=s\log p+(N-s)\log(1-p)$. Its derivative is $s/p-(N-s)/(1-p)$. Setting it to zero gives $s(1-p)=(N-s)p$, hence p=s/N. All-success or all-failure samples give boundary estimates one or zero.

**For Gaussian data, expose both derivatives.** Write variance as v. Ignoring constants, $\ell(\mu,v)=-\tfrac N2\log v-\frac1{2v}\sum_i(x_i-\mu)^2$. Differentiating in mu gives $\sum_i(x_i-\mu)/v=0$, so mu is the sample mean. Differentiating in v gives $-N/(2v)+\sum_i(x_i-\mu)^2/(2v^2)=0$, yielding the average squared residual with denominator N.

Likelihood is not a probability distribution over parameters unless combined with a prior and normalized in a Bayesian calculation. These MLE derivations also assume the stated IID models. A neat closed form does not validate independence, Gaussianity, or the absence of outliers in real data.
@@ 794
**The sample mean is chosen to fit the same data whose spread you measure.** That makes residuals around it systematically smaller than residuals around the unknown true mean. Dividing their squared sum by N therefore underestimates the population variance on average. [1]

**A two-point intuition.** With only one observation, the fitted sample mean equals that observation, so the residual spread is zero regardless of the population's real variability. Estimating the center has used information from the sample and removed one degree of freedom. For N observations, residuals must sum to zero, so they cannot vary independently in all N directions.

The identity below separates squared deviations around the true mean from the amount explained by moving to the sample mean. Taking expectations gives $(N-1)\sigma^2$ for the residual sum under IID finite-variance assumptions. Dividing by N-1 therefore returns an unbiased variance estimate.

**Why the MLE still uses N.** Maximum likelihood optimizes the likelihood under its model; it does not promise that every resulting estimator is unbiased. Unbiasedness and likelihood optimality are different criteria. The N-denominator estimator can have different mean-squared-error properties despite its downward bias.

Gaussianity is unnecessary for this Bessel-correction expectation argument, though it matters for the Gaussian MLE derivation and some exact distributional results. Dependence, weighting, or estimating more parameters changes the relevant degrees of freedom. Taking a square root also does not preserve unbiasedness, so an unbiased variance estimate does not automatically give an unbiased standard deviation.
@@ 795
**MLE uses the data likelihood; MAP adds a prior preference over parameters.** When data is sparse, the prior can prevent extreme or implausible estimates. MAP selects the most probable parameter value under the posterior density, while MLE selects the value making the observed data most likely. [1]

**A coin example.** With two heads and no tails, Bernoulli MLE is p=1. A Beta(2,2) prior favors neither extreme; after observing two heads the posterior is Beta(4,2), whose interior mode is 3/4. The prior has moderated the estimate. The posterior mean is a different summary and should not be confused with the MAP mode.

Taking logs turns the prior into an additive term. A Gaussian coefficient prior yields an L2-type MAP penalty; a Laplace prior yields an L1-type penalty under the corresponding likelihood. The strength depends on prior scale and whether the data loss is summed or averaged.

**Why the difference often shrinks with more data.** A fixed regular prior contributes a bounded-scale term while the log likelihood accumulates evidence across observations. Under suitable identifiability and regularity conditions, their preferred estimates approach the same limit.

That conclusion is conditional. A prior excluding the true parameter, nonidentifiability, growing model dimension, or a penalty scaled with sample size can preserve differences. MAP also gives one point rather than uncertainty over possible parameters. Use the full posterior or other uncertainty methods when the spread matters to the decision.
@@ 797
**KL averages under its first argument.** Forward KL samples where P places probability and asks whether Q assigns enough probability there. Reverse KL samples where Q places probability and asks whether those samples land in regions P considers plausible. Changing the averaging distribution changes which mistakes are heavily exposed. [1]

**Use a two-island picture.** Let P have two separated peaks, while Q must be one Gaussian. Forward KL strongly penalizes Q for neglecting either peak, so a broad Gaussian covering both may be preferred even if it also covers the low-density gap. Reverse KL penalizes Q for generating in that gap, so it may prefer concentrating near one peak.

This “mode-covering versus mode-seeking” behavior arises from the restricted approximation family and the target geometry. It is not a theorem that every reverse-KL solution loses modes; if Q can represent P exactly, both directions are minimized by Q=P.

**Support makes the asymmetry sharp.** If P gives positive mass where Q gives zero, forward KL is infinite. Reverse KL instead becomes infinite when Q assigns positive mass where P is zero. The order of the arguments therefore encodes a meaningful modeling choice, not a notation preference.

In distillation or variational inference, name the teacher/target and student/approximation explicitly before interpreting the objective. Then inspect generated coverage, unsupported mass, and task quality. A single scalar divergence value does not explain every practical failure of a restricted model.
@@ 798
**LLN says averages settle; CLT describes the shape of their remaining fluctuations under additional conditions.** These are related but different statements. The law of large numbers concerns convergence toward an expectation. The classical IID central limit theorem concerns the distribution of a centered, properly scaled average when variance is finite. [1]

**A coin illustration.** As more fair-coin flips are averaged, the head fraction tends toward one half. That is the LLN intuition. Across many repeated experiments of N flips, the distribution of the fraction around one half becomes approximately Gaussian after scaling by its standard error. That is the CLT intuition.

The CLT does not say the original observations become normal, nor that every sample size of 30 is automatically sufficient. Tail behavior, skew, and dependence affect approximation quality.

**Heavy tails separate the theorems.** Cauchy observations do not have a finite mean or variance; their average remains Cauchy rather than concentrating in the usual way. Some Pareto distributions have a finite mean but infinite variance, allowing an LLN while the classical finite-variance normal CLT fails. Other limiting laws or normalizations may apply.

Before using standard errors, examine the data-generating process, clustering, serial dependence, and outliers. More rows do not create more independent information if observations are strongly related. State which version of a limit theorem and which assumptions justify the approximation used in an analysis.
@@ 799
**Imagine retraining the same learning algorithm on many different datasets.** At a fixed input x, it produces slightly different predictions each time. Their average may systematically miss the true regression function; individual predictions fluctuate around that average; and the observed target itself contains noise. Those are the three sources in the squared-error decomposition. [1]

Let $m(x)=\mathbb E_D[\hat f_D(x)]$ and $y=f(x)+\epsilon$. Write the prediction error as $[f(x)-m(x)]+[m(x)-\hat f_D(x)]+\epsilon$. Squaring produces three squared terms and cross terms.

**Why cross terms disappear.** The estimator fluctuation has mean zero by definition of m. Test noise has zero conditional mean and is independent of training under the standard setup. Taking expectations therefore removes the cross terms, leaving squared bias, estimator variance, and noise variance.

**A practical interpretation.** A rigid model can miss a curved relationship consistently: high bias. A very flexible model may fit each training sample differently: high variance. Measurement noise can remain even if the true conditional mean is known. More representative data or regularization can alter these components, but no fixed rule says one always moves up when another moves down.

This identity is for squared loss under specified expectations. It does not directly decompose classification accuracy, and “irreducible” noise is relative to the features/information available. A better sensor or an informative new feature can reduce what previously looked like unavoidable noise.
@@ 800
**An eigenvector is a direction a linear transformation does not turn away from.** Applying A to v scales it by lambda: $Av=\lambda v$. Positive lambda stretches or shrinks along the same direction; negative lambda also reverses it; zero collapses that direction to zero. [1]

**A simple matrix.** A diagonal matrix with entries 3 and 1 stretches the horizontal axis threefold and leaves the vertical axis unchanged. Those axes are eigenvectors. A symmetric matrix with off-diagonal entries may have rotated principal directions, but still admits an orthonormal eigenbasis.

For two eigenvectors u and v with distinct eigenvalues, symmetry makes $u^TAv=(Au)^Tv$. Substituting the eigenvalue relations gives $(\lambda_v-\lambda_u)u^Tv=0$, so their inner product must be zero. Within a repeated-eigenvalue subspace, an orthonormal basis can be chosen; arbitrary eigenvectors selected there need not already be orthogonal.

**Why covariance and Hessians benefit.** Covariance eigenvectors identify directions of variation, with nonnegative eigenvalues. A symmetric Hessian's eigenvalues describe local curvature directions, but can be negative in a nonconvex problem. Symmetry guarantees real eigenvalues, not positive ones.

Compare geometric subspaces rather than raw vector signs in numerical tests. An eigenvector and its negative describe the same direction, and repeated eigenvalues allow multiple equally valid bases. Numerical roundoff also means exact orthogonality becomes a tolerance-based check in computation.
@@ 801
**SVD decomposes a linear map into rotation, axis scaling, and another rotation/reflection.** For a rectangular matrix A, the right singular vectors describe input directions, singular values describe their nonnegative amplification, and left singular vectors describe the corresponding output directions. This works even when A is not square and ordinary eigenvectors are not the appropriate tool. [1]

**Why small singular values can be dropped.** If one direction is scaled strongly and another barely at all, the latter contributes little to the matrix's action under standard norm criteria. Keeping only the largest k singular values gives the best rank-k approximation in Frobenius norm and, with the appropriate statement, spectral norm. This is the Eckart–Young–Mirsky result.

For centered data, PCA uses right singular vectors as principal feature directions. Squared singular values divided by N-1 give sample explained variances. Without centering, the decomposition is still an SVD, but it is not the usual mean-centered PCA interpretation.

**A compression example.** A nearly rank-one image matrix may be well approximated by one outer product of a column pattern and row pattern. More retained components add detail, with a measurable reconstruction-error tradeoff.

The largest-variance directions need not be the most predictive of a label, and small singular values can still matter for a particular downstream task. Choose rank using the actual objective and validation, not only a convenient explained-variance percentage.
@@ 802
**PSD means every direction has nonnegative quadratic energy.** For a real symmetric matrix A, the condition $x^TAx\ge0$ for every x is equivalent to all eigenvalues being nonnegative. Zero eigenvalues are allowed, which distinguishes semidefinite from positive definite. [1]

**Covariance gives a direct proof.** For a random vector X and any coefficient vector a, the scalar combination $a^TX$ has variance $a^T\Sigma a$. Variance cannot be negative, so every valid covariance matrix must be PSD. If one feature is an exact copy of another, some combinations have zero variance, making the covariance singular but still valid.

**Kernel matrices have the same structure.** If $K_{ij}=\langle\phi(x_i),\phi(x_j)\rangle$, then $a^TKa=\|\sum_i a_i\phi(x_i)\|^2\ge0$. This is why a valid inner-product kernel must produce PSD Gram matrices on finite samples. A superficially reasonable similarity function is not automatically such a kernel.

Tiny negative eigenvalues can arise from floating-point error; substantial negatives may indicate an invalid construction or inconsistent data processing. Pairwise covariance estimates based on different missing-data subsets, for example, need not assemble into a valid global covariance matrix.

Use appropriate symmetric solvers and distinguish numerical regularization, such as adding a diagonal jitter, from proving the original matrix satisfied the intended statistical or kernel assumptions.
@@ 803
**A Jacobian records first-order output sensitivities; a Hessian records curvature of a scalar function.** For a vector-valued function, Jacobian entry ij tells how output i changes with input j. For a scalar loss, the Hessian contains second derivatives, describing how gradient components change as parameters move. [1]

**A local landscape example.** On a quadratic bowl, the gradient points uphill and the Hessian describes how steeply each direction curves. Newton's method rescales the gradient using that curvature, potentially reaching the optimum in one step for an exact positive-definite quadratic with appropriate conditions.

**Why full Newton is impractical for huge networks.** With P parameters, a dense Hessian has $P^2$ entries. At one million parameters, that is one trillion entries before solving a system. Building, storing, and factorizing it is far more expensive than a gradient calculation. Mini-batch noise and nonconvexity also complicate the step; an indefinite Hessian need not produce a descent direction.

This does not mean second-order information is unusable. Hessian-vector products, quasi-Newton approximations, structured curvature methods, and preconditioners avoid materializing the whole matrix, each with tradeoffs.

Choose an optimization method by total progress per resource budget, not iteration count alone. A method requiring fewer iterations can still be slower or less robust if each iteration is vastly more expensive or estimates curvature poorly from noisy data.
@@ 804
**Ill-conditioning means different directions operate on very different scales.** In a positive-definite quadratic loss, large Hessian eigenvalues correspond to steep directions and small ones to flat directions. One global learning rate must avoid overshooting the steep direction while still making progress along the flat one. [1]

**Picture a long narrow valley.** The gradient points strongly across the valley because that curvature is steep, even though the desired progress is mostly along it. A large step bounces across; a safe small step makes slow forward progress. This produces the familiar zig-zag trajectory.

For a positive-definite Hessian, the eigenvalue ratio describes this disparity. More generally, the 2-norm matrix condition number uses singular values. An indefinite Hessian has negative curvature and needs a different interpretation from a simple convex bowl.

Feature standardization can reduce avoidable scale differences; preconditioning and adaptive methods change the effective coordinate system; momentum can reduce oscillation. None is a guarantee that every nonlinear landscape becomes well-conditioned.

A zero singular value makes the matrix condition number infinite, but a flat parameter direction can reflect symmetry or nonidentifiability rather than a failure to predict. Distinguish solving for a unique parameter vector from obtaining useful predictions, and use stable solvers instead of explicit inverses when analyzing related linear systems.
@@ 805
**A p-value asks how surprising the data would be under a specified null procedure.** It is the probability, assuming that null model and its sampling/test assumptions, of a statistic at least as extreme as the observed one. It does not assign a probability to the null hypothesis itself. [1]

**An analogy with a coin test.** If a fair coin produced an unusually unbalanced result, a p-value measures how often a fair-coin experiment would produce an imbalance at least that large. It does not directly answer “what is the probability this coin is fair?” That latter question requires a different inferential setup, such as prior probabilities and a Bayesian model comparison.

Power is the chance of rejecting the null when a specified alternative effect is true. A study can have low power for a small worthwhile improvement and high power for a much larger effect. Power depends on sample size, noise, allocation, effect size, and the testing rule.

**Why 0.04 is not “96% certainty.”** The calculation conditions on the null; it does not invert that condition. Nor does statistical significance establish practical importance. A huge dataset can detect a tiny effect, while a small noisy dataset can miss a useful one.

Report effect estimates, uncertainty intervals, design, and relevant assumptions. Repeated peeking, selective metric reporting, and unplanned multiple comparisons change the effective procedure and can invalidate the naive interpretation of the reported p-value.
@@ 806
**Sample-size planning starts with the smallest effect worth detecting.** The MDE is a design choice tied to practical value, not a prediction of the true effect. Power specifies the desired chance of detecting that effect under the test assumptions; alpha controls a chosen false-positive rate. [1]

**Distinguish absolute from relative changes.** At a 10% conversion baseline, a one-percentage-point increase means 10% to 11%, so delta is 0.01. A 1% relative increase means 10% to 10.1%, so delta is 0.001. Because required sample size scales roughly with $1/\delta^2$, the latter can require about 100 times as many observations.

The normal-approximation formula below combines baseline Bernoulli variance with quantiles for significance and power. Under its equal-arm assumptions, the 10%-baseline, 0.01-absolute-MDE example needs roughly 14,112 units per arm.

**Translate units into duration carefully.** Divide required eligible randomized units by enrollment rate, then allow complete business cycles, delayed outcomes, ramp-up, and exclusions. Counting page views as independent units when randomization is by user can greatly overstate information.

Clustering, unequal allocation, multiple metrics, non-Bernoulli outcomes, and sequential testing require adapted planning. Use simulation or an appropriate analytic method for the actual design. A sample-size calculator is only as meaningful as the definition of one independent observation and the metric it receives.
@@ 807
**Repeatedly looking changes the chance of eventually seeing a lucky fluctuation.** A fixed-horizon test is calibrated for its specified analysis. Checking daily and stopping at the first p-value below 0.05 gives noise many chances to trigger a win, so the overall false-positive probability can exceed 5%. [1]

**The intuition is a search over stopping times.** Even if treatment and control are identical, their cumulative estimates wander. Selecting the day when the difference looks best is a form of selection. Reporting that day's ordinary p-value ignores the selection rule that produced it.

One solution is to commit to a fixed horizon. Another is a prespecified group-sequential design that allocates error across planned looks. Always-valid approaches build statistics with guarantees that remain valid under allowed data-dependent stopping, using assumptions such as those underlying mixture sequential probability ratio tests or e-processes.

**This is not permission to change everything mid-test.** Always-valid stopping does not automatically justify changing metrics, removing inconvenient cohorts, trying many variants, or violating randomization assumptions. Those choices need their own error control and design.

Plan both stopping for success and stopping for harm/futility. Report the sequential method and thresholds, not just the familiar number 0.05. The goal is to make the actual monitoring behavior match the statistical guarantee rather than pretending an actively monitored experiment had one fixed analysis.
@@ 808
**Testing many hypotheses gives chance more opportunities to look significant.** With 20 independent true nulls each tested at 0.05, the probability of at least one false rejection is $1-0.95^{20}\approx64\%$. Independence is needed for that exact calculation, but the general multiplicity problem does not disappear when metrics are correlated. [1]

**Bonferroni protects against any false discovery in the family.** Test each of m hypotheses at alpha/m. The union bound controls family-wise error without requiring independence, though the method can be conservative, particularly with correlated tests.

**Benjamini–Hochberg targets a different error quantity.** Sort p-values and compare the kth to $k\alpha/m$ under the procedure's assumptions. It controls the expected fraction of false discoveries among discoveries rather than the probability of any false discovery. This can be more appropriate for exploratory screening, where some false leads are tolerable.

Neither method means every declared result has a known individual probability of being false. Also, BH's standard guarantees have dependence conditions; arbitrary selection and repeated peeking are not automatically covered.

Define the testing family and primary outcome before inspecting results. A team that tests fifty metrics but reports only the best one has not avoided multiplicity by showing one chart. Use effect sizes and uncertainty alongside corrected decisions, and reserve independent confirmation for important exploratory findings.
@@ 809
**CUPED uses a pre-treatment measurement to explain predictable noise.** If users who spent more before an experiment also tend to spend more during it, some outcome variation is unrelated to treatment. Adjusting for that predictable component can make the treatment comparison more precise without changing random assignment. [1]

**The adjustment is intuitive.** Subtract a fitted multiple of the centered pre-experiment covariate from the outcome: $Y_{adj}=Y-\theta(X-\mathbb E[X])$. A user with unusually high historical activity has some expected high current activity removed, letting the comparison focus more on unexpected differences.

With the optimal simple linear coefficient and suitable assumptions, variance shrinks by a factor related to $1-\rho^2$, where rho is the outcome–covariate correlation. If squared correlation is 0.5, the adjusted variance is roughly half, which can translate into roughly half the sample size for the same target precision in that regime.

**The time saving is conditional.** Weak correlation yields little gain. Missing pre-period data, new users, ratio metrics, and model estimation require care. Do not use a covariate changed by treatment as if it were harmless baseline information; that can alter the estimand or introduce bias.

Validate the adjustment procedure, randomization balance, and uncertainty calculation. CUPED improves measurement efficiency when useful pre-treatment information exists; it does not create a treatment effect or compensate for broken logging or an invalid experiment design.
@@ 810
**One person's treatment can change another person's outcome.** A new feed changes which creators receive exposure, affecting content available to both groups. A pricing policy changes driver movement and shared supply, affecting riders assigned to control. Simple user-level comparisons can therefore mix direct effects with spillovers. [1]

**The counterfactual is no longer isolated.** A control user in a heavily treated market may not experience the same environment they would under a full control rollout. The experiment can estimate a mixture-specific effect rather than the intended all-users deployment effect.

Graph-cluster randomization groups strongly connected users to reduce cross-group interaction. Geographic splits separate markets when geography contains much of the interference. Switchbacks alternate treatment over time for a shared system, with randomized schedules and appropriate washout/carryover handling.

Each design has costs: clusters reduce the number of independent units and may differ in size; geographies can differ systematically; time switches can be confounded by temporal patterns or persistent effects if designed poorly. There is no universal design that removes all interference.

Define the exposure mechanism and desired estimand before choosing the randomization unit. Analyze uncertainty at that unit, not by pretending every event is independent. Use domain knowledge about networks, movement, and carryover to make the experimental boundary approximate the actual interaction boundary.
@@ 811
**Behavior can change because an experience is new, not because it is better long term.** A redesigned feature may attract curiosity and temporarily increase engagement. That novelty effect can fade. Conversely, established users may initially struggle with a change that later becomes useful, or retain habits from the old interface; the terminology around primacy/learning effects should be stated explicitly. [1]

**A feed example.** Users inspect a new layout intensely during its first week, then return to normal activity. A short experiment may overstate lasting value. Another layout may initially slow users while they learn it, so an early result can understate eventual usefulness.

Plot effects by time since first exposure and by calendar time; those are different axes. New-user cohorts reduce some legacy-habit effects, while long-term holdouts help measure persistence and downstream outcomes. Ensure cohort composition and outcome maturity remain comparable.

**Do not simply wait until the curve looks favorable.** Choose a duration and analysis strategy appropriate to the hypothesized adaptation process, with valid repeated-monitoring rules. Seasonality, concurrent launches, and changing user mix can imitate a fading or growing treatment effect.

Report short-term and longer-term behavior separately when both matter. A product can reasonably value immediate usability and sustained benefit differently, but that tradeoff should be explicit rather than hidden by choosing whichever window makes the experiment look successful.
@@ 812
**Aggregation can reverse a relationship when group composition differs.** Simpson's paradox occurs when an overall comparison points one way while comparisons within relevant groups point another. The aggregate weights groups differently, so the mix can dominate the within-group effect. [1]

**A simple intuition.** Suppose mobile users convert less often than desktop users. Treatment may improve conversion within both device groups but receive a much larger share of mobile traffic. Its pooled conversion can then be lower, even though the within-device comparisons are positive. Which comparison answers the question depends on randomization, the estimand, and whether the groups are pre-treatment or treatment-affected.

**SRM is a separate warning about allocation or observation.** A planned 50/50 assignment producing 502,100 versus 497,900 users may be statistically suspicious at that sample size. Diagnose assignment, eligibility, logging, bot filters, redirects, and differential missingness before trusting outcome comparisons.

SRM does not prove one particular bug or automatically quantify outcome bias. It signals that the experiment as observed does not match its expected allocation process. A small percentage imbalance can be highly significant with large N, while practical consequences depend on the mechanism.

Do not “fix” SRM by arbitrarily dropping users until counts match. Find and correct the cause, then determine which data remains valid or rerun. Statistical adjustment cannot reliably repair an unknown selection process after the fact.
@@ 813
**Without randomization, identification rests on a credible reason treatment comparisons mimic the missing counterfactual.** Each quasi-experimental method supplies a different reason, and each can fail when its assumptions are wrong. A sophisticated estimator does not make those assumptions true. [1]

Difference-in-differences compares changes across treated and comparison groups, relying on an appropriate parallel-trends counterfactual and other design conditions. Propensity matching compares observed covariate-similar units, requiring no relevant unmeasured confounding and adequate overlap. Synthetic control constructs a weighted comparison from untreated units, useful when pre-treatment fit and donor validity support the design.

Regression discontinuity uses a treatment-assignment cutoff: units just above and below can be comparable if other determinants vary smoothly and assignment cannot be manipulated problematically. Instrumental variables uses variation affecting treatment but not the outcome through another path, with relevance, exclusion, and other assumptions determining the identifiable effect.

**Choose by the assignment mechanism.** A genuine threshold suggests RDD; a credible external encouragement may suggest IV; a policy affecting one region with suitable untreated regions may motivate DiD or synthetic control. Choosing whichever method gives significance reverses the logic.

Use pre-trend/placebo checks, covariate balance, overlap diagnostics, sensitivity analyses, and domain evidence. These checks can reveal problems but do not prove every untestable assumption. State the population/effect identified and the remaining uncertainty rather than presenting an observational estimate as an experiment with a different name.
''')
