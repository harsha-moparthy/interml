from expand import write
write(r'''
@@ 1051
**The logit gradient has a simple interpretation: predicted probability minus target.** For a positive example predicted at 0.2, the gradient is -0.8, so gradient descent increases the logit. For a negative example predicted at 0.9, it is +0.9, pushing the logit down strongly. [1] [2]

To derive it, differentiate BCE with respect to p, giving $-y/p+(1-y)/(1-p)$. Then multiply by the sigmoid derivative p(1-p). The first term becomes -y(1-p), the second becomes (1-y)p, and their sum simplifies to p-y.

**The cancellation matters.** A saturated sigmoid alone has a small derivative, but its pairing with cross-entropy cancels problematic denominators and supplies a strong correction for confidently wrong examples. This does not remove every vanishing-gradient problem in a deep network; it explains the output-layer derivative.

Use a stable logits-based implementation, and include the loss reduction: averaging N examples divides their contributions by N. Class weighting changes the derivative. Check a simple sign example before trusting an implementation; a reversed sign would train confidence in the wrong direction.
@@ 1052
**Softmax outputs are coupled because they share one denominator.** Increasing logit j raises its own probability and lowers the others. The Jacobian therefore has diagonal terms $S_i(1-S_i)$ and off-diagonal terms $-S_iS_j$. [1] [2]

Apply the quotient rule to $S_i=e^{z_i}/\sum_k e^{z_k}$. Combining the two cases gives $S_i\delta_{ij}-S_iS_j$. Cross-entropy contributes upstream derivative $-y_i/S_i$ for each output. Summing their products yields $S_j\sum_i y_i-y_j$, which becomes S_j-y_j for normalized targets.

**An example checks the result.** If probabilities are (0.7,0.2,0.1) and class two is correct, the logit gradient is (0.7,-0.8,0.1). Descent decreases the wrong-class logits and increases the correct one. The components sum to zero, consistent with the fact that adding a common constant to all logits changes nothing.

Label smoothing preserves this formula when its target distribution sums to one. Weighted or unnormalized targets need their own derivation. Use stable log-softmax rather than logging potentially underflowed probabilities.
@@ 1053
**Normalization makes every coordinate depend on the row's statistics.** Changing one input changes not only its own centered value but also the mean and variance used by every output coordinate. A backward pass that treats those statistics as constants misses those indirect paths. [1] [2]

Let upstream gradient be g and first absorb the affine scale with $u=g\odot\gamma$. LayerNorm's input gradient starts from u, removes its row mean, then removes a component related to the normalized input, and finally scales by the inverse standard deviation. The full formula below retains epsilon and the variance dependence.

**Why subtract the mean-gradient component?** Adding the same constant to every input leaves centered normalization unchanged. The input gradients must therefore sum to zero along that shift direction. The mean subtraction enforces this invariance.

RMSNorm does not subtract the input mean, so it has no corresponding centering correction. Its backward pass still differentiates the root-mean-square scale. Sum affine gradients over all rows sharing gamma and beta. Verify constant rows, small variance, and mixed precision against autograd; epsilon changes exact scale invariance and must not be dropped casually.
@@ 1054
**Backward distributes each output's sensitivity through the same linear connections used in forward.** With row-batched inputs and $Y=XW^T+b$, upstream gradient G has one sensitivity per output element. The input gradient combines output sensitivities using W; the weight gradient accumulates input–sensitivity products across examples. [1]

For one scalar output $y=w_1x_1+w_2x_2+b$ and upstream derivative g, the input derivatives are gw1 and gw2, weight derivatives are gx1 and gx2, and bias derivative is g. Matrix multiplication simply performs these sums for every row and output.

**Shapes verify the algebra.** G is N by output-width, so GW is N by input-width. $G^TX$ is output-width by input-width, matching W. Summing G across examples matches the bias vector.

If inputs also have sequence dimensions, flatten all leading example axes for parameter-gradient accumulation and restore the input shape afterward. Sum over every broadcast axis for the bias. Do not divide by batch size twice if G already includes a mean-loss factor. Shape agreement is necessary, though a small numerical gradient check can also catch transposition mistakes.
@@ 1055
**Ridge balances fitting the observations against large coefficients.** The objective is squared residual sum plus lambda times squared coefficient norm. Differentiating the residual term gives $2X^T(Xw-y)$; differentiating the penalty gives $2\lambda w$. [1] [2]

Set their sum to zero and collect w: $(X^TX+\lambda I)w=X^Ty$. This is the ridge normal equation. The inverse notation expresses the solution, but software should solve the system rather than materialize an inverse.

**Why regularization helps dependent features.** For any nonzero direction v, the quadratic form is $\|Xv\|^2+\lambda\|v\|^2$, strictly positive for lambda>0 when every direction is penalized. A direction invisible to the data now has a cost, making the solution unique.

This also introduces shrinkage bias, so lambda needs validation. Feature units affect the relative penalty and may justify standardization. Handle an unpenalized intercept by centering or a separate penalty matrix. State whether the data loss is summed or averaged, since that changes the numerical scale of lambda.
@@ 1056
**Choose the smaller linear system when dimensions are very unequal.** Ordinary ridge solves for d feature coefficients. When d is huge but there are only n examples, the same solution can be expressed through n example weights instead. [1] [3]

The identity $(X^TX+\lambda I)X^T=X^T(XX^T+\lambda I)$ lets the inverse move across X transpose in the appropriate way. Solve $(XX^T+\lambda I)\alpha=y$, then recover $w=X^T\alpha$.

**An illustrative size comparison.** With 1000 examples and 100,000 features, a dense d-by-d matrix has ten billion entries, while the n-by-n Gram matrix has one million. Building either still involves data processing, but the factorization and storage difference is substantial.

Kernel ridge replaces inner products with a kernel and predicts from similarities to training examples without explicit feature weights. Sparsity and iterative solvers can change which route is best. The primal and dual formulas are mathematically equivalent under the stated penalty; neither requires explicitly computing an inverse. Consider serving cost too, because a kernel prediction may depend on many training examples.
@@ 1057
**EM alternates between soft cluster assignments and weighted parameter fitting.** In a Gaussian mixture, the component responsible for each observation is hidden. The E-step computes its posterior probability under the current parameters; the M-step treats those responsibilities as fractional memberships. [1] [2]

If one point has responsibilities (0.8,0.2), it contributes 0.8 of an observation to the first component and 0.2 to the second. Summing those fractions gives effective component counts. Weighted means and variances then update each Gaussian, while normalized counts update mixture probabilities.

**Use the new mean when recomputing variance.** Otherwise the update does not match the stated M-step. Compute responsibilities in log space to avoid underflow when densities are tiny.

Exact EM does not decrease the likelihood, but that does not guarantee the best global mixture. A component can collapse onto one point with vanishing variance, creating a singular likelihood. Use variance floors or priors, multiple initializations, and a policy for tiny components. Inspect likelihood improvement and component behavior together rather than calling any monotonic training curve a valid density estimate.
@@ 1058
**Jensen's inequality explains why moving a nonlinear function outside an expectation changes the result.** For a convex function, the function lies above a supporting line at the mean. Taking expectations of that line cancels its linear displacement term, giving $g(E[X])\leq E[g(X)]$. Log is concave, so its inequality points the other way. [2]

A latent-variable model's evidence integrates over z, which can be hard to evaluate. Introduce a tractable q(z|x), write the evidence as an expectation of p(x,z)/q(z|x), then apply concavity of log. This produces a computable lower bound: expected reconstruction log likelihood minus KL from q to the prior. [1]

**The bound has an exact gap.** Log evidence minus ELBO equals KL from the approximate posterior q to the true posterior. If they match, the gap is zero; if q cannot represent the posterior, the gap can remain positive.

Support and integrability assumptions are necessary for the rearrangement. Optimizing the ELBO improves a lower bound, not directly a guarantee of sharp samples or exact inference. The variational family and decoder determine what compromises the model can make.
@@ 1059
**Move randomness into a parameter-independent input so the sampled value becomes a differentiable function.** For a Gaussian latent variable, draw epsilon from a standard normal and set $z=\mu+\sigma\epsilon$. Changes in mu shift the sample; changes in sigma stretch its displacement. [1]

For fixed epsilon, the derivatives are straightforward: dz/dmu=1 and dz/dsigma=epsilon. Backpropagation can therefore differentiate a downstream reconstruction loss through z into the encoder's mean and scale outputs.

**Why ordinary sampling needs care.** Treating a sampled value as a constant misses how changing distribution parameters changes the expectation. Reparameterization provides a pathwise gradient estimator under conditions allowing differentiation and expectation to interchange.

A Monte Carlo sample still gives a noisy estimate, but it often has lower variance than a score-function estimator for suitable continuous distributions. Discrete samples generally need different estimators or relaxations, which introduce their own variance or bias. Stochastic computation is not inherently incompatible with gradients; the estimator must correctly account for the parameter-dependent distribution.
@@ 1060
**Generative models trade off representation, likelihood, coverage, and sampling work in different ways.** A VAE learns an encoder and decoder through a likelihood bound, often enabling quick sampling and latent inference. A GAN learns through a discriminator and can generate sharp samples in one pass but may omit modes. [1] [2]

Normalizing flows use invertible transformations with tractable density calculations, constraining architecture while enabling exact likelihood under the model. Diffusion models learn iterative denoising or related dynamics and often provide strong quality and diversity at greater sampling cost. [3]

**No single metric captures the tradeoff.** A model can create attractive images while missing rare categories, or assign useful likelihoods while producing less visually sharp samples. Compare coverage, perceptual quality, inference needs, and end-to-end latency on matched data and resolution.

Distillation, latent representations, and fewer-step samplers can move the frontier. These are empirical tendencies, not a theorem that only three properties can ever coexist or that one family always wins. Choose based on the task's required outputs and budget, and include the cost of all encoding, iterative steps, and decoding.
@@ 1061
**Mode collapse means different random inputs lead to too few kinds of output.** A generator might repeatedly produce one convincing digit while ignoring the others. Sample sharpness can look good even though distribution coverage is poor. [1] [2]

In the original saturating GAN loss, a discriminator that easily rejects generated samples can produce weak generator gradients through its saturated probability. The common non-saturating loss improves that gradient behavior but does not remove every source of collapse.

**WGAN changes the comparison signal.** A real-valued Lipschitz critic estimates a Wasserstein-style discrepancy, giving a transport-related objective that can remain informative when real and generated supports are separated. Gradient penalty encourages the required smoothness on sampled interpolations.

The penalty is not a global proof that the critic is exactly Lipschitz, and optimization can still fail. Track diversity and coverage alongside visual quality and critic losses. A better-behaved objective is useful, but it does not turn adversarial training into a guaranteed globally convergent procedure.
@@ 1062
**Count matrices by their dimensions, then multiply repeated blocks.** Standard multi-head attention with model width d has query, key, value, and output projections, each roughly d by d: four d squared parameters. A two-matrix feed-forward block contributes $2dd_{ff}$. [1]

With feed-forward width 4d, one block contributes 12d squared before smaller terms. L blocks plus one tied vocabulary embedding table give $12Ld^2+Vd$ under these assumptions.

**A numerical example.** With d=512, L=12, and V=32,000, blocks contribute about 37.75 million parameters and tied embeddings about 16.38 million, for roughly 54.13 million before biases and norms. Untying the output table adds another 16.38 million.

Gated FFNs have three matrices, GQA changes key/value projection sizes, and MoE adds expert parameters that may not all activate per token. Learned positional tables, biases, and final normalization also matter. State the architecture before calling a formula exact. Parameter count, active computation, and activation memory are related but distinct quantities.
@@ 1063
**Training memory includes values saved for backward, not only model weights.** Residual streams and projections scale with batch, sequence length, and width. Feed-forward intermediates also depend on the expanded hidden width. Conventional attention can save a much larger tensor with one score per head and query-key pair. [1]

For batch one, 32 heads, and sequence length 4096, that score tensor has $32\times4096^2=536{,}870{,}912$ elements. At two bytes each, it alone occupies one GiB. Doubling sequence length quadruples this particular storage.

**FlashAttention avoids storing the full matrix.** It tiles computation and retains compact statistics, recomputing needed quantities during backward. Saved attention activations can then scale linearly with sequence length at fixed width, although dense attention arithmetic remains quadratic.

Exact memory depends on fused kernels, dropout, dtype, and checkpointing. Count saved tensors separately from weights, gradients, optimizer state, inference KV cache, and temporary workspaces. Measure a realistic forward/backward peak and distinguish allocated memory from allocator reservations. A theoretical tensor count explains trends but may understate the practical device requirement.
@@ 1064
**A rolling count changes both when events arrive and when old events expire.** Key transactions by card ID and maintain time-indexed state. Each new transaction enters the relevant ten-minute and 24-hour totals; contributions leave when they pass each window boundary. [1]

Keeping every event timestamp supports exact boundaries but uses more state. Minute buckets reduce storage, yet the oldest partial minute creates approximation unless finer detail is retained. Choose that tradeoff explicitly instead of calling a bucketed count exact.

**Event time and arrival time differ.** A late transaction may belong inside a window already processed. Watermarks and an allowed-lateness policy define how to update or reject such records. Deduplicate using transaction IDs so retries do not inflate counts.

Checkpoint state and serve materialized values with timestamps or versions. Define whether the transaction being scored is included, and reconstruct the same definition for training without future events. Monitor hot cards, lag, expiry processing, and recovery. A fast Redis lookup is only trustworthy if the upstream state and time semantics are correct.
@@ 1065
**Membership storage grows with watched pairs, not merely with users or videos.** A Bloom filter stores approximate membership compactly, allowing false positives but no false negatives for inserted items under ordinary operation. Its size depends on the number of retained user-video pairs and the chosen false-positive rate. [1]

At a 1% target false-positive rate, the ideal formula gives about 9.6 bits per pair. Fifty decimal GB spread over 500 million users provides about 800 bits per user, enough for roughly 83 pairs per user before implementation overhead.

**That arithmetic rules out unrestricted lifetime history under the stated budget.** Choose a retention window, larger storage, tiered durable history, or a different error tolerance. False positives mean suppressing an unseen item, which may be acceptable only at a measured rate.

Standard Bloom filters do not support safe arbitrary deletion without additional structure. Count-Min Sketch estimates frequencies and tends to overestimate; it is not a drop-in exact history store. Define whether the product needs lifetime membership, recent exposure, or frequency before selecting the data structure.
@@ 1066
**A random hyperplane turns a direction into one bit.** Hash a nonzero vector according to which side of the hyperplane it lies on. Two vectors with a small angle are likely to land on the same side; opposite directions are unlikely to agree. [1]

Restrict attention to the plane containing the two vectors. For isotropic random orientations, the fraction of separating hyperplanes is theta divided by pi. Thus bit agreement probability is $1-\theta/\pi$. At a 60-degree angle, one-bit agreement is two thirds.

**Several bits make stricter buckets.** With k independent bits, exact bucket collision probability is $(1-\theta/\pi)^k$. More bits reduce unrelated collisions but can also miss relevant neighbors. Multiple tables or probing nearby bit patterns increase recall at additional cost.

This collision formula is not the complete retrieval recall. Candidate budgets, table counts, reranking, and data geometry matter. Use angular conventions consistently and handle zero vectors explicitly. Compare the resulting latency–recall tradeoff with other ANN methods rather than assuming a mathematically elegant collision rule is automatically the fastest index.
@@ 1067
**A huge output vocabulary makes the normalization and scoring work expensive.** Hierarchical softmax replaces one flat choice with decisions along a tree path. In a balanced tree, evaluating the target's probability requires about log V decisions instead of V class scores. [1] [2]

The path probabilities multiply to the leaf probability. Tree design affects which classes share decisions and how long their paths are; frequent classes can receive shorter paths.

**Sampling changes the training problem differently.** Sampled softmax compares the positive with selected classes and applies sampling corrections. A small sampled denominator generally is not exactly the full-softmax objective. NCE instead learns to distinguish data from a known noise distribution, permitting estimation of an unnormalized model under its assumptions.

At ten million classes, also consider distributed lookup, frequency skew, and negative quality. Faster training does not automatically solve inference: retrieving exact top-k classes can still require a search strategy. Compare objective bias, gradient variance, memory, and serving requirements, not merely the number of logits computed during one training example.
@@ 1068
**Unlabeled data help only when their structure supports a valid learning assumption.** Label propagation assumes nearby graph nodes tend to share labels. Pseudo-labeling assumes sufficiently confident model predictions are often correct. FixMatch uses a confident prediction on a weak augmentation as a target for a stronger augmentation. [1] [2]

For an image, a mild crop may produce a reliable class estimate, and a stronger but label-preserving transformation should retain that class. If the transformation removes the only relevant object, the consistency target becomes wrong.

**Errors can reinforce themselves.** A confident mistaken pseudo-label becomes training evidence, making the next model even more confident in the mistake. Class imbalance can then starve minority classes of selected examples.

Use a trusted validation set, inspect class coverage, ramp the unlabeled-loss contribution, and consider class-aware sampling or thresholds. Check domain shift and graph quality. Confidence is produced by the same model being improved; it is not independent ground truth. Compare against a labeled-only baseline at equal compute to establish that the unlabeled pipeline actually helps.
@@ 1069
**Domain adaptation tries to preserve task information while reducing irrelevant source–target differences.** A classifier trained on clean studio images may fail on noisy phone images even when the object classes are the same. DANN adds a domain discriminator to a shared feature encoder. [1]

The discriminator learns to tell source from target. A gradient-reversal layer sends the opposite of its encoder gradient, scaled by lambda, so the encoder learns features that make domain identification harder while still supporting the source task.

**Domain confusion is not sufficient.** If class proportions differ, blindly matching marginal feature distributions can align the wrong classes. If the target contains genuinely new conditions or classes, hiding domain identity may discard useful information rather than solve the shift.

Evaluate on labeled target validation data when available, with the final target test set reserved. Compare targeted augmentations and simpler adaptation baselines. Inspect subgroup quality and calibration. The desired invariance is to nuisance variation, not to every difference between source and target; semantic alignment must be tested rather than inferred from a weak domain classifier.
@@ 1070
**Detection quality requires the right class and a sufficiently accurate box.** For each class and IoU threshold, sort detections by confidence and match them to eligible ground-truth objects in the same image. An unmatched or duplicate detection counts as a false positive under the evaluation rules. [1]

As the confidence cutoff moves down the list, compute precision and recall. COCO forms a precision envelope and samples it at 101 recall levels. AP50 uses an IoU threshold of 0.50; the main COCO AP averages across ten thresholds from 0.50 through 0.95 and across evaluated classes.

**A loose box can pass AP50 and fail stricter thresholds.** That is why the multi-threshold score measures localization more demanding than merely finding the approximate region. A high classification score cannot rescue insufficient overlap.

Use the official evaluator for crowd/ignore handling, area ranges, maximum detections, and absent classes. Report size-specific AP and recall to expose tiny-object failures. A simplified home-grown matcher may produce a plausible number that is not comparable to the published benchmark.
@@ 1071
**Dense detectors see far more easy background examples than difficult objects.** Ordinary cross-entropy can let the sum of thousands of easy negatives dominate the learning signal. Focal loss reduces the contribution of examples already predicted confidently correctly. [1]

For correct-class probability pt, multiply cross-entropy by $(1-p_t)^\gamma$ and an optional class weight. With gamma=2, pt=0.99 gets factor 0.0001, while pt=0.2 gets factor 0.64. The hard example retains much more influence.

**The two parameters have different roles.** Alpha balances classes; gamma controls how strongly easy examples are downweighted. At gamma=0, the modulation disappears and the loss becomes weighted cross-entropy.

The modulation term is part of the differentiable expression, so its derivative also contributes; treating it as a detached weight changes the gradient. Excessive focusing can prioritize mislabeled or inherently ambiguous examples. Tune using detection quality and inspect calibration. Focal loss addresses one imbalance mechanism, not every source of detector error or the entire explanation for an architecture's performance.
@@ 1072
**Ordinary IoU provides no useful overlap change while two boxes remain disjoint.** Small movements can leave intersection area at zero, making the overlap loss locally flat. A learning signal should still indicate whether the predicted box is approaching the target. [1] [2]

GIoU adds a penalty based on unused area in the smallest enclosing box. DIoU adds normalized center distance. CIoU additionally measures aspect-ratio mismatch. These terms provide geometric information beyond intersection alone.

**Imagine two same-size boxes far apart.** Their IoU is zero regardless of a small move toward the target. Center distance decreases under that move, so DIoU supplies a directional incentive. Once boxes overlap, both alignment and overlap influence the loss.

The formulas below specify the enclosure, diagonal normalization, and aspect-ratio term. Handle zero or invalid widths and heights and avoid numerical divisions by zero. These objectives still have nonsmooth or degenerate cases and are not identical to the official AP metric. Compare localization improvements across object sizes and shapes rather than assuming one added geometric term always wins.
@@ 1073
**Multi-step forecasting can reuse one prediction recursively or learn horizons directly.** A recursive model predicts tomorrow, feeds that prediction back, then predicts the next day. It is compact, but an early error becomes a later input and can compound. [1]

Direct forecasting trains a separate predictor for each horizon. It avoids feeding its own errors forward but can require many models and produce inconsistent trajectories. Joint forecasting outputs the whole horizon together, sharing information across future steps.

**Architecture names do not settle the distinction.** A sequence-to-sequence model may still decode recursively, while another predicts a vector in parallel. Specify whether later outputs condition on earlier predicted values.

Use rolling-origin evaluation and report error by horizon, trajectory coherence, and uncertainty. Only provide future covariates actually known at the forecast origin; realized future weather is not available merely because it exists in the historical table. Compare the strategy under realistic missing inputs and compute constraints. The best one-step model need not be the best long-horizon system.
@@ 1074
**Percentage errors become unstable when actual values are near zero.** MAPE divides each absolute error by the actual magnitude, so zero is undefined and a tiny actual can dominate the average. A forecast error of one unit is 100% when actual is one and 1% when actual is 100. [1]

The common sMAPE denominator uses both actual and forecast magnitudes. It is symmetric when swapping the two, but not for equal over- and under-prediction around one fixed actual: actual 100 with predictions 120 and 80 gives about 18.2% and 22.2% respectively.

**Aggregate and baseline-scaled alternatives answer different questions.** WMAPE divides total absolute error by total actual magnitude, weighting large-volume series more heavily. MASE divides error by a training-period naive or seasonal-naive baseline error, supporting scale comparison.

Each has edge cases: zero aggregate volume or a perfect naive baseline creates a zero denominator. State the exact convention and aggregation level. Match the metric to operational costs rather than assuming a percentage is inherently fair across items or meaningful for targets that can be negative.
@@ 1075
**Forecasts should add up when the real quantities add up.** Store sales sum to regional sales, and regions sum to national sales. Independently forecasting each level can produce a national prediction of 100 while regional predictions sum to 120. Reconciliation restores those accounting relationships. [1]

Bottom-up forecasts the leaves and sums them. Top-down predicts an aggregate and allocates it using proportions. Each can ignore useful information at other levels. MinT instead projects a collection of base forecasts onto the coherent space using an estimate of forecast-error covariance.

**The summing matrix expresses the structure.** If bottom-level forecasts are b, all levels are S times b. The reconciled output in the formula below lies in S's column space, so the totals agree by construction.

Covariance estimation is difficult with limited history, making shrinkage useful. Correlation of errors matters because forecasts from different levels may repeat the same information. Standard linear reconciliation can produce negative values and does not automatically make intervals coherent. Use appropriate constrained or probabilistic methods if those properties are required, and evaluate quality at every hierarchy level.
''')
