from author import write
write(r'''
@@ 951|optim dl linear d2l reg
Identically initialized hidden units receive identical inputs and gradients, so they remain copies instead of learning different features. With all-zero multilayer weights, gradients through some layers can also vanish entirely. [1]

Zero biases are usually fine when random weights already break neuron symmetry. Logistic regression has no interchangeable hidden units; zero initialization gives a valid starting probability and generally a nonzero convex-loss gradient. The issue is architectural symmetry, not a rule that every trainable scalar must begin random. Use initialization matched to activation and fan-in/out.
@@ 952|optim dl reg d2l torch
A common nonzero constant does not break hidden-unit symmetry: neurons with identical incoming/outgoing parameters compute the same function and receive matching updates. The layer effectively wastes width. [1]

It can also make activation variance grow with width because same-sign contributions accumulate. Random initialization creates distinct feature directions while controlling variance through fan-in/out scaling. There are specialized deterministic asymmetric schemes, so randomness itself is not essential; **different, appropriately scaled units** are. A constant bias can coexist with asymmetric weights without causing this problem.
@@ 953|optim torch cv d2l rules
Shuffling reduces systematic correlations between successive minibatches, such as batches containing only one class or time period. It makes minibatch gradients more representative and reduces order-specific training artifacts. [1]

Shuffling is without-replacement sampling within an epoch, not exactly iid sampling, but usually works well. It does not repair a biased dataset or invalid validation split. In distributed training, vary the sampler's epoch seed so every epoch does not repeat one order. Preserve group/sequence structure when the model or objective depends on it, and record seeds for reproducibility.
@@ 954|forecast torch cv optim d2l
Do not arbitrarily reorder chunks when hidden state is carried from one chunk to the next, or when an online learner must simulate chronological availability. A deliberate curriculum also has an intended order. [1]

However, time-series **training windows** can often be shuffled if each window is self-contained, hidden state is reset correctly, and train/validation splits remain chronological. “Never shuffle time series” is therefore too broad. Separate the ordering needed inside an example, across stateful batches, and between independent examples. Check labels and timestamps before deciding the sampling policy.
@@ 955|scikit prml cluster islp metrics
At $K=1$, kNN is highly local and sensitive to individual samples: usually low bias and high variance. Increasing $K$ smooths predictions, generally increasing bias and reducing variance. At $K=N$, every query receives the global training mean or class-frequency decision. [1]

These are trends, not strict monotonic guarantees for every dataset or metric. Distance scaling, class imbalance, sample density, and distance weighting affect behavior. Choose $K$ with appropriate cross-validation and evaluate rare-class/local-region performance; a globally stable predictor can ignore the structure that matters for the task.
@@ 956|scikit prml cluster faiss islp
kNN is “lazy” because it largely stores training examples rather than fitting a compact global prediction function. Its effective complexity can grow with the dataset, making it nonparametric. [1]

Brute-force prediction computes $N$ distances in $d$ dimensions, about $O(Nd)$, plus selection/aggregation; storing data costs $O(Nd)$. “Training” may just validate/store data, though building an index can be substantial. Trees or approximate indexes change query/build tradeoffs. Nonparametric does not mean no hyperparameters: $K$, metric, scaling, weighting, and index settings still need validation.
@@ 957|scikit faiss hnsw prml cluster
KD-trees partition coordinate space; ball trees bound groups inside metric balls. Exact search prunes a region when its distance lower bound exceeds the best current candidates. In favorable low-dimensional geometry this avoids many distance evaluations. [1]

As dimensions rise, bounds often become weak and many regions must be visited, approaching brute-force work with extra traversal overhead. There is no universal cutoff at dimension 20; intrinsic dimension, distribution, metric, and sample size matter. Benchmark exact trees against vectorized brute force and approximate search at the required recall, rather than choosing from dimensionality alone.
@@ 958|prml islp scikit linear cluster
A parametric model has a fixed-size parameterization under a specified architecture as sample size grows: linear regression, logistic regression, and a fixed-component Gaussian mixture are examples. Nonparametric model complexity can grow with data: kNN, kernel density estimation, and unconstrained-depth decision trees are examples. [1]

The distinction is about model capacity/representation, not whether parameters exist. A neural network with a fixed architecture is parametric despite many weights. A tree with a fixed leaf cap has a capacity restriction. Both families require assumptions and regularization; “nonparametric” does not mean assumption-free.
@@ 959|prml linear svm islp scikit
A generative classifier models $P(X\mid Y)P(Y)$ and uses Bayes' rule for classification; Naive Bayes and class-conditional GMMs are examples. A discriminative classifier directly models $P(Y\mid X)$ or a decision boundary; logistic regression and SVMs are examples. [1]

Generative assumptions can help with limited data, missing-data modeling, or simulation, but misspecifying $P(X)$ may harm classification. Discriminative methods focus on the boundary and often need fewer distributional assumptions about inputs. An SVM score is not automatically a probability; calibration is separate. Compare under the actual data regime rather than declaring one family universally superior.
@@ 960|ngjordan linear islp scikit calibration
The Ng–Jordan result compares particular generative/discriminative pairs under specific assumptions and uses dimension/sample complexity carefully; the prompt's single symbol $n$ conflates feature dimension with sample count. It is not a universal $O(\log n)$ guarantee for arbitrary Naive Bayes data. [1]

The practical insight is that strong generative assumptions can reach their own limiting error quickly, while a discriminative model may approach a better asymptotic boundary more slowly. If Naive Bayes assumptions are wrong, its bias persists; if they are correct, the comparison changes. Evaluate learning curves at several sample sizes and report assumptions rather than applying the slogan as a theorem about every dataset.
@@ 961|torch optim prml attention scipy
Subtracting the maximum keeps exponent arguments nonpositive while preserving the normalized result. [1]

$$\log\sum_i e^{x_i}=m+\log\sum_i e^{x_i-m},\qquad m=\max_i x_i.$$

Without stabilization, large positive logits can overflow and produce infinity divided by infinity in softmax. Subtraction prevents that overflow for finite inputs, but does not repair NaNs or an all-negative-infinity masked row. Such rows need an explicit policy, often zero output with safe normalization. Use library log-sum-exp/log-softmax operations, especially when computing cross-entropy, rather than taking a log of already rounded tiny probabilities.
@@ 962|linear optim svm metrics prml
Zero-one loss changes only when a prediction crosses the decision boundary, so it gives no useful local gradient almost everywhere and creates a difficult discontinuous optimization problem. Cross-entropy provides a smooth probability-sensitive surrogate. [1]

For binary logits, its gradient is $\sigma(z)-y$, giving a strong correction for confidently wrong examples. Convexity holds for linear logits, not arbitrary neural networks. Minimizing log loss does not directly optimize every downstream threshold/cost metric, so select thresholds and evaluate accuracy/utility separately. The difficulty of exact zero-one minimization depends on the hypothesis class; avoid a universal complexity claim without specifying it.
@@ 963|svm linear optim prml metrics
For signed label $y\in\{-1,1\}$ and score $f(x)$, hinge loss is $\max(0,1-yf(x))$. Once the signed margin is at least one, its data-loss gradient is zero. Logistic loss $\log(1+e^{-yf(x)})$ keeps decreasing smoothly as margin grows, though its gradient becomes small. [1]

The margin is a score, not a probability greater than one. Hinge focuses updates on margin violations; log loss has a probabilistic likelihood interpretation. Regularization still affects correctly classified points through parameters. Calibration, robustness, and decision costs should be evaluated beyond training loss.
@@ 964|ensemble calibration metrics prml islp
Hard voting takes each model's predicted class and chooses the majority or weighted majority. Soft voting averages class probabilities and chooses the largest aggregate probability, retaining confidence information. [1]

Soft voting can improve decisions when probabilities are reasonably comparable, but an overconfident miscalibrated model can dominate. Weighting and calibration must be learned on held-out/out-of-fold predictions, not the final test set. Align class order and handle missing classes consistently. Diversity matters: averaging many identical errors gives little benefit. For abstention or cost-sensitive decisions, use the aggregated probabilities with an explicitly validated threshold rule.
@@ 965|ensemble prml scipy islp reg
Expand the variance of an average into diagonal variance and off-diagonal covariance terms. With equal variance $\sigma^2$ and pairwise covariance $\rho\sigma^2$, [1]

$$Var(\bar f)=\frac{M\sigma^2+M(M-1)\rho\sigma^2}{M^2}=\rho\sigma^2+\frac{1-\rho}{M}\sigma^2.$$

For uncorrelated errors, this is $\sigma^2/M$. Correlation leaves a floor, so more highly similar models eventually add little. These assumptions concern errors/prediction variation under an ensemble-generating process; they do not guarantee identical reduction in classification error. Bias is not removed by averaging identically biased models, and compute cost grows unless predictions can be compressed or distilled.
@@ 966|ensemble xgb islp prml scikit
Bagging changes training samples, but a dominant feature may still drive nearly every tree's early splits. Random forests also restrict candidate features at each split, encouraging different structures and reducing error correlation. [1]

Using roughly $\sqrt p$ candidates is a common classification setting, not a defining fixed rule for every task/library. Smaller subsets can increase individual-tree bias while improving diversity; the ensemble tradeoff must be tuned. Random thresholds and leaf/depth constraints affect it too. Measure validation/OOB quality and stability rather than assuming the lowest correlation alone gives the best predictor.
@@ 967|ensemble scipy prml islp cv
A bootstrap sample draws $N$ times with replacement from $N$ rows. A particular row is omitted with probability $(1-1/N)^N$, which tends to $e^{-1}$ because $N\log(1-1/N)\to-1$. [1]

OOB prediction for a row averages only trees that did not train on it, giving an internal validation estimate. It is not a substitute for a temporal/group holdout when examples are dependent, and repeatedly tuning to OOB results can introduce selection bias. The 36.8% figure is per tree in expectation, not a fixed permanently held-out subset shared by all trees.
@@ 968|ensemble optim islp prml reg
For binary AdaBoost with weighted error $\epsilon_t<1/2$, the learner weight is $\alpha_t=\tfrac12\log((1-\epsilon_t)/\epsilon_t)$. Update example weights proportionally to $w_i e^{-\alpha_ty_ih_t(x_i)}$, then normalize. [1]

Mistakes gain relative weight and correct examples lose it. This corresponds to reducing an exponential margin loss, so persistently mislabeled outliers can attract extreme attention. Shrinkage, early stopping, robust variants, or better labels may help. Random forests do not repeatedly reweight the same errors this way, though they can still overfit noise. Handle zero/error-one edge cases rather than evaluating an infinite coefficient blindly.
@@ 969|ensemble leak cv metrics prml
Stacking trains a second-level model on predictions from several base learners. Training it on in-sample predictions lets it exploit overfitting patterns unavailable at deployment. Generate each training row's base predictions from models fitted without that row, using group/time-aware folds when needed. [1]

Fit the meta-model on those out-of-fold features, then refit base models on allowed training data for serving. In-sample stacking does not literally always assign 100% weight to one model; the real issue is biased meta-training information. Keep preprocessing within folds and reserve an independent evaluation set for the entire stacked pipeline.
@@ 970|optim reg linear prml islp
Early stopping chooses a checkpoint before optimization fully fits noise, using a separate validation criterion and patience rule. For quadratic linear regression, finite gradient-descent time shrinks slowly learned low-curvature directions, resembling spectral regularization. [1]

It is **not generally exactly equivalent** to ridge with one scalar penalty: early-stopping and ridge apply different spectral filters. In neural networks the connection is even less exact. Save the best checkpoint rather than the last patience-expired one, and avoid repeatedly tuning against the final test set. Compare compute, validation uncertainty, and metric alignment when deciding when to stop.
@@ 971|mixup reg vit dl d2l
Mixup forms convex combinations of two inputs and their target vectors, encouraging smoother behavior between examples. CutMix replaces an image region with a region from another image and weights labels by the mixed area. [1]

$$\tilde x=\lambda x_i+(1-\lambda)x_j,\quad \tilde y=\lambda y_i+(1-\lambda)y_j.$$

These are task-dependent augmentations: mixing may create unrealistic data or violate labels for detection, segmentation, or structured inputs unless targets are transformed consistently. Validate augmentation strength and calibration as well as accuracy. Area-based CutMix labels approximate semantic contribution; a small pasted region can contain the only relevant object.
@@ 972|vit reg ensemble scaling metrics
Test-time augmentation averages predictions across label-preserving transformations, reducing sensitivity to orientation/crop choices and acting like a small ensemble. [1]

It multiplies inference work and can increase tail latency and energy, making it unattractive for some interactive systems—but it is used in production when the benefit justifies cost. The transformation must preserve the task: flipping text, medical laterality, or directional objects can change meaning. Align dense outputs back to original coordinates before averaging. Benchmark real end-to-end latency and robustness rather than assuming competition gains transfer to the deployment distribution.
@@ 973|linear torch metrics prml scikit
Multiclass classification chooses one mutually exclusive class; softmax probabilities sum to one and categorical cross-entropy scores the chosen class. Multilabel classification allows several tags simultaneously, commonly using one sigmoid/BCE term per label. [1]

Independent sigmoid outputs do not assert that real-world labels are statistically independent; they specify a factorized output/loss model that can share features. Thresholds may differ per label and should reflect prevalence/cost. Missing labels are not automatically negatives. Evaluate macro/micro metrics and rare-label slices, and use structured modeling if label constraints such as mutual exclusion or hierarchy are essential.
@@ 974|scikit svm linear prml metrics
One-vs-rest trains $K$ classifiers, each separating one class from all others. One-vs-one trains $K(K-1)/2$ classifiers on class pairs and combines their votes or scores. [1]

OvO has more models but each can train on fewer examples; this can benefit some kernel methods. OvR scores may have different scales and need consistent decision/calibration handling. Neither scheme inherently exploits class hierarchy or guarantees optimal multiclass probabilities. Compare class imbalance, model size, training complexity, and prediction cost. A native multiclass objective such as softmax is a third option rather than merely a voting wrapper.
@@ 975|linear prml metrics islp scikit
Ordinal labels have ordered categories, so predicting 1 instead of 5 is usually worse than predicting 4 instead of 5. Model cumulative probabilities $P(Y>k)$ for $k=1,\ldots,K-1$, with ordered thresholds or another mechanism enforcing nonincreasing probabilities. [1]

$$P(Y=k)=P(Y>k-1)-P(Y>k).$$

Use boundary values one and zero at the ends. Independent unconstrained binary heads can violate ordering and produce negative category probabilities after differencing. Cumulative-link models share a score and ordered cutpoints; more flexible models need consistency constraints. Evaluate ordinal cost and calibration, not only exact-match accuracy.
@@ 976|prml linear scipy forecast metrics
A hurdle model first predicts whether the outcome is positive, then models its positive magnitude. For the example, use a Bernoulli purchase/spend gate and a positive distribution such as log-normal. [1]

$$E[Y\mid x]=P(Y>0\mid x)E[Y\mid Y>0,x].$$

Zero-inflated count models instead mix a structural-zero process with a count distribution that can also generate zero; that differs from a hurdle's zero-truncated positive component. Select the model from how zeros arise, not their frequency alone. Evaluate gate calibration, positive-tail fit, and aggregate value. Delayed purchases may be censored rather than permanent zeros.
@@ 977|prml scipy causal forecast metrics
Survival analysis models time to an event while accounting for observations whose event has not happened by the last follow-up. Kaplan–Meier estimates the survival curve; Cox models a baseline hazard multiplied by covariate-dependent relative risk. [1]

$$h(t\mid x)=h_0(t)e^{\beta^Tx}.$$

Treating censored cases as permanent negatives in binary classification loses follow-up information and biases labels. Survival methods require assumptions such as appropriate independent censoring, and Cox requires proportional hazards for its standard interpretation. Competing events and recurrent failures need extensions. Evaluate time-specific calibration and discrimination at clinically/operationally relevant horizons, not only one aggregate concordance score.
@@ 978|scipy scikit prml linear islp
Box–Cox and Yeo–Johnson are power transformations used to reduce skew or stabilize variance under suitable modeling goals. “Bessel function” is a different mathematical object and does not belong in this list. [1]

For positive $x$, Box–Cox uses $(x^\lambda-1)/\lambda$ when $\lambda\ne0$ and $\log x$ when $\lambda=0$. Yeo–Johnson extends a related piecewise family to zero/negative inputs. Fit transformation parameters using training data only. Normal-looking features are not a universal requirement: trees may gain little, while linear residual/variance assumptions can matter more than marginal feature normality.
@@ 979|scipy leak cv metrics prml
Winsorization replaces extreme values with boundary values; trimming removes those observations. Clipping retains sample count but distorts tails, while trimming changes the analyzed population. [1]

Estimate percentile boundaries on training data and reuse them for validation/serving so future information does not influence preprocessing. A 1st/99th percentile rule is a choice, not proof those values are errors. Legitimate rare events may be the most important examples. Keep an outlier/missingness indicator where useful, examine units and collection bugs first, and evaluate tail performance and distribution shift before adopting the transformation.
@@ 980|prml scipy pca cluster scikit
Mahalanobis distance measures displacement after accounting for covariance, equivalent to Euclidean distance in whitened coordinates when covariance is positive definite. [1]

$$d_M(x,\mu)=\sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}.$$

A displacement along a high-variance correlated direction is less surprising than the same raw displacement across a tight direction. Estimating covariance can be unstable when dimension is large relative to samples; use shrinkage or a justified low-rank model. Outliers can distort ordinary covariance, and a single Gaussian ellipsoid may not represent multimodal data. Solve using a factorization rather than explicitly inverting the matrix.
@@ 981|optim torch swa reg d2l
Cosine annealing smoothly lowers learning rate within a cycle; warm restarts reset it to a larger value to explore a new region while retaining model weights. A common schedule is [1]

$$\eta_t=\eta_{min}+\tfrac12(\eta_{max}-\eta_{min})(1+\cos(\pi T_{cur}/T_i)).$$

Cycle lengths may grow over training. A restart is not reinitializing the network, and optimizer-state handling must match the chosen implementation. Benefits depend on task and budget; compare with simpler decay under equal updates. Save validation-selected checkpoints and report the schedule in optimizer steps, especially when changing batch size or accumulation.
@@ 982|optim torch reg swa d2l
The 1-cycle policy raises learning rate from a low starting value to a peak, then decays it to a much smaller value, often varying momentum inversely. The high-rate phase can act as regularization and speed exploration; the final low-rate phase refines the fit. [1]

“Super-convergence” describes favorable empirical cases, not a guarantee. Peak rate, total steps, batch size, weight decay, and augmentation interact. Use a stable range-finding experiment and held-out validation, and ensure the schedule advances once per intended optimizer update. Too large a peak can destroy training rather than accelerate it.
@@ 983|swa optim ddpm reg torch
EMA updates $\theta_{ema}\leftarrow\beta\theta_{ema}+(1-\beta)\theta$; Polyak averaging averages iterates; SWA averages selected later checkpoints, often under a suitable learning-rate schedule. Averaging can smooth optimization noise and improve validation quality. [1]

Diffusion training commonly uses EMA, but “LLMs almost always serve EMA weights” is not a reliable universal claim. BatchNorm statistics may need recomputation after weight averaging, and unrelated checkpoints may not share a compatible basin. Evaluate the averaged model explicitly. Keep averaging decay, start time, dtype, and checkpoint state documented; averaging predictions and averaging weights are different operations.
@@ 984|checkpoint zero scaling torch optim
For a simple chain of $L$ equal-cost/equal-memory layers, divide it into segments of length $s$. Store about $L/s$ boundaries and at most $s$ recomputed internal activations at a time. [1]

$$M(s)=O(L/s+s),\qquad s\approx\sqrt L\Rightarrow M=O(\sqrt L).$$

This concerns activations under an idealized checkpoint schedule, not total model memory. Parameters, optimizer states, non-chain graphs, attention buffers, unequal layers, and recomputation strategy change practical behavior. Other schedules achieve different memory/compute tradeoffs. Preserve randomness and state semantics during replay and measure peak allocated/reserved device memory rather than claiming every checkpointed network has exactly square-root total memory.
@@ 985|torch bn amp optim d2l
`model.eval()` changes module behavior such as dropout and BatchNorm statistics; it does not disable autograd. `torch.no_grad()` disables gradient recording in its scope but does not toggle module training/evaluation mode. [1]

For typical inference, use evaluation mode together with no-grad or a suitable inference-mode context. For gradient-based attribution, you may need evaluation mode with gradients enabled. Conversely, a training-mode forward inside no-grad can still update BatchNorm running statistics. Choose both controls deliberately and restore state after temporary evaluation; confusing them can produce unnecessary memory use or inconsistent predictions.
@@ 986|torch cuda compile d2l optim
`view()` reinterprets shape without copying only when strides are compatible. `reshape()` returns a view when possible and otherwise may allocate a copy. Transpose/permute usually changes strides rather than rearranging storage, producing a noncontiguous view. [1]

A tensor can be noncontiguous yet still permit some views, so “view requires contiguous” is an oversimplification. Use `contiguous()` when an explicit layout copy is appropriate, but account for its cost. Never rely on `reshape()` always aliasing or always copying. Check strides, shape, and mutation semantics when debugging a silently incorrect axis manipulation.
@@ 987|torch optim amp d2l compile
`detach()` returns a tensor disconnected from the current autograd graph while generally sharing storage. `clone()` copies data but remains differentiable, so gradients can flow back to the original tensor. [1]

Use `detach().clone()` when you need an independent snapshot that neither participates in gradient history nor aliases future in-place changes. A detached view is not immutable; modifying shared storage can affect the original or trigger autograd version checks. For logging, a CPU transfer may also be needed. Distinguish stopping gradients, copying memory, and moving devices—each is a separate operation.
@@ 988|torch cuda scaling compile sre
CUDA operations are usually asynchronous relative to the CPU. Reading a device scalar with `.item()` or formatting device tensor values for printing requires the value on the host, introducing synchronization and breaking useful overlap. [1]

The cost depends on context; one occasional scalar read does not universally “destroy” throughput, but doing it every step can create repeated stalls. Accumulate metrics on-device, log less frequently, and transfer batches of summaries at planned boundaries. Use proper CUDA events or synchronized timing for benchmarks; otherwise asynchronous execution can make a slow operation appear deceptively cheap.
@@ 989|torch bn rope optim d2l
`nn.Parameter` registers a tensor as a model parameter, normally returned by `parameters()` and supplied to an optimizer. `register_buffer` tracks non-parameter state for device moves and, by default, serialization. [1]

BatchNorm running statistics are buffers because they are updated by a running-stat rule rather than gradient descent. RoPE tables can be buffers, but may be registered nonpersistent or regenerated instead of saved; “must be in state_dict” is too strong. Buffers can still be mutated by code. Set persistence and dtype/device handling deliberately and exclude fixed constants from optimization.
@@ 990|torch scaling zero cuda sre
Collectives require compatible participation and ordering across ranks. If rank 0 enters a barrier or a DDP-triggered collective while rank 1 skips it, rank 0 waits for a matching operation that never arrives, often ending in timeout. [1]

Use consistent control flow for distributed operations, broadcast decisions when necessary, and ensure loaders/iteration counts match or use supported uneven-input mechanisms. Log rank-local step IDs and the first error, since one rank's earlier exception can cause everyone else's apparent communication failure. Increasing timeout is not a fix for mismatched collective sequences.
@@ 991|attention t5 dl testtime prml
Teacher forcing conditions each training prediction on the correct previous tokens, enabling parallel loss computation over a known sequence. At inference the model conditions on its own generated history, so an early error can move it into unfamiliar contexts and compound. [1]

$$L=-\sum_t\log p_\theta(y_t\mid x,y_{<t}^{gold}).$$

This discrepancy is called exposure bias. Sequence-level training, data augmentation, or carefully designed sampling methods can help, but naive scheduled sampling changes the training objective and has caveats. Evaluate complete generated sequences and recovery from perturbed prefixes, not only teacher-forced token loss. EOS behavior and decoding choices also affect the observed failure.
@@ 992|attention t5 testtime spec prml
Sequence log probability sums nonpositive token log probabilities, so extending a fixed prefix lowers its probability. Beam search may favor early EOS, though normalization across all possible sequences is valid and model/length biases also matter. [1]

A heuristic uses $score(y)=\log P(y)/((5+|y|)/6)^\alpha$ or another length penalty. This changes the decoding objective and must be tuned on task quality, not assumed probabilistically exact. Include EOS consistently and define whether prompt length is excluded. Longer output is not inherently better, and a loose stopping bound can discard a future better normalized beam.
@@ 993|spec attention testtime sentencepiece prml
Greedy decoding repeatedly chooses the highest conditional probability and can enter self-reinforcing loops under the model's learned distribution. Low temperature sharpens that preference. Repetition penalties reduce scores of previously used tokens; min-p sampling retains tokens whose probability exceeds a fraction of the current maximum. [1]

$$A=\{i:p_i\ge p_{min}\max_jp_j\}.$$

Neither guarantees no repetition or factuality. Penalizing repeated tokens can damage names, code, or legitimate repeated phrases; min-p may retain only the dominant loop token if the distribution is very peaked. Evaluate task-specific decoding and consider model/data issues rather than treating a sampling knob as a universal cure.
@@ 994|sentencepiece attention metrics prml t5
Perplexity is the exponential of average token negative log likelihood, equivalent to the inverse geometric mean probability assigned to observed tokens. PPL 10 corresponds to the same average log loss as ten equally likely choices, not literally ten candidates at every step. [1]

Different tokenizers define different prediction units, so raw token PPL is not an apples-to-apples comparison. Compare total log likelihood normalized by a common byte/character count with consistent text preprocessing and encoding. Tokenization, BOS/EOS handling, and context windows must be specified. Lower perplexity on one corpus does not guarantee better instruction-following or reasoning quality.
@@ 995|metrics irl sbert judge t5
BLEU measures clipped n-gram precision with a brevity penalty, often aggregated at corpus level. ROUGE variants emphasize reference overlap through recall, longest common subsequences, or related formulations. BERTScore compares contextual token embeddings; MoverScore uses transport-like semantic alignment. [1] [3]

Lexical metrics penalize valid paraphrases; semantic metrics can overlook changed numbers, negation, or unsupported facts. No single score establishes summary faithfulness. Use multiple references where appropriate, report exact variants/tokenization, and combine automated metrics with source-grounded human or executable checks. Comparing one sentence's BLEU to corpus BLEU is especially misleading.
@@ 996|ewc reg ties lora prml
Catastrophic interference occurs when updates for new tasks overwrite parameters needed by earlier tasks, degrading old performance. Shared features can help transfer, but conflicting gradients and narrow new data can shift them destructively. [1]

Replay old examples, constrain important parameters, allocate task-specific capacity/adapters, or distill old outputs. Each trades storage, compute, flexibility, and retention. Evaluate old/new tasks throughout the sequence, including class-incremental settings where task identity is unavailable at inference. Keeping old labels in a checkpoint is not enough if the new model's decision boundaries or calibration forget them.
@@ 997|dp fedavg membership prml mltest
FedAvg sends local model updates for server aggregation rather than centralizing raw examples. Updates can still leak information; federated learning alone is not differential privacy. [1] [2]

An $(\epsilon,\delta)$-DP mechanism bounds how output-event probabilities change between neighboring datasets: $P(M(D)\in S)\le e^\epsilon P(M(D')\in S)+\delta$. DP-SGD clips contributions, adds calibrated noise, and accounts for composition across training steps. Define whether neighbors differ by one record or an entire user.

DP limits an individual's influence under stated assumptions; it does not guarantee literally no memorized string or no inference from population facts. Privacy strength depends on the budget and implementation, with a utility tradeoff.
@@ 998|membership dp fedavg calibration mltest
Membership inference tries to determine whether a specific record was in training, using model outputs, losses, gradients, or other access. A model may behave more confidently on memorized training examples than comparable nonmembers. [1]

Assess a defined attacker, access level, and realistic member/nonmember distribution; distribution mismatch can make an attack look stronger than true membership leakage. Regularization and output restriction may reduce risk but do not certify privacy. Properly implemented differential privacy offers a formal influence bound. Evaluate sensitive subgroups and rare examples, and distinguish membership inference from reconstructing the full record or extracting model weights.
@@ 999|adversarial torch optim reg vit
FGSM perturbs an input along the sign of the loss gradient within an $L_\infty$ budget. PGD repeats small gradient steps and projects back into the allowed perturbation set. [1]

$$x_{adv}=\operatorname{clip}_{valid}(x+\epsilon\operatorname{sign}(\nabla_xL)).$$

For evaluation, define pixel scale, norm, budget, targeted/untargeted objective, and threat-model access. Robust training can improve resistance to specified attacks but may trade clean accuracy and compute. Test adaptive attacks and avoid mistaking broken gradients for robustness. Small mathematical perturbation does not always correspond to an equally plausible physical-world change, so deployment robustness needs additional corruptions and sensor tests.
@@ 1000|calibration metrics linear torch prml
Platt scaling fits a sigmoid with slope/intercept to a binary score; multiclass extensions require a specified construction. Temperature scaling fits a single positive scalar on held-out multiclass logits. [1]

$$p_i(T)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}},\qquad T>0.$$

Dividing all logits by the same positive value preserves their ordering, so top-1 class predictions remain unchanged aside from numerical/tie effects. Larger temperature softens probabilities; smaller temperature sharpens them. It cannot fix misranked classes or arbitrary class-specific calibration errors. Fit on independent calibration data and recheck under distribution shift.
''')
