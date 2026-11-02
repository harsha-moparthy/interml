from author import write
write(r'''
@@ 764|ensemble xgb catboost islp prml
Trees naturally partition heterogeneous columns with threshold rules, model nonlinear interactions, and ignore many uninformative features. Their axis-aligned inductive bias often matches tabular semantics: rotating “age” and “income” into arbitrary mixtures can make useful thresholds harder to express. Boosting adds regularized corrections and works well at moderate sample sizes. [1] [2]

This is not a theorem that GBDTs consistently beat every neural model. Large datasets, learned representations, multimodal inputs, and strong tabular pretraining can favor neural methods. Compare leak-free tuning under matched compute, categorical handling, missingness, and realistic splits. Trees also struggle with extrapolation and can memorize leakage. Start with a well-tuned GBDT baseline, then justify complexity through measured quality, calibration, and operational cost.
@@ 765|ensemble islp prml xgb scikit
Bagging fits models on resampled data and averages them, mainly reducing variance of unstable learners such as deep trees. Random forests further decorrelate trees by randomizing candidate features. Boosting fits learners sequentially to reduce errors of the current ensemble, often reducing bias but also affecting variance. [1]

$$\operatorname{Var}\left(\frac1M\sum_m f_m\right)=\sigma^2\left[\rho+\frac{1-\rho}{M}\right]$$

under equal variance and pairwise correlation assumptions. The correlation term creates a floor. Boosting uses $F_m=F_{m-1}+\eta h_m$, with shrinkage, depth, subsampling, and early stopping controlling complexity. “Bagging reduces variance, boosting reduces bias” is a useful tendency, not a complete characterization of every loss or dataset.
@@ 766|ensemble xgb islp reg prml
More random-forest trees refine the Monte Carlo average of a fixed tree-building procedure. Under equal-variance/correlation assumptions, its variance approaches a floor as the number of trees rises. This does not guarantee monotonic improvement of every finite test metric on every run. [1]

Boosting adds sequential corrections to the fitted function, increasing effective complexity as well as reducing training loss. Eventually it can fit noise or mislabeled examples; small learning rates and early stopping help. The number of trees therefore plays a different role in the two methods.

Hold validation splits fixed and plot uncertainty across seeds. For forests, tune depth, leaf sizes, and feature subsampling; for boosting, jointly tune tree count, shrinkage, and tree complexity. Avoid the literal claim that any additional forest tree can never worsen measured performance.
@@ 767|xgb ensemble lightgbm islp prml
Expand each example's loss around its current prediction, ignoring terms constant in the new tree. If leaf $j$ predicts $w_j$, define $G_j=\sum_{i\in I_j}g_i$ and $H_j=\sum_{i\in I_j}h_i$. [1]

$$\tilde{\mathcal L}=\sum_j\left[G_jw_j+\tfrac12(H_j+\lambda)w_j^2\right]+\gamma T,\quad w_j^*=-\frac{G_j}{H_j+\lambda}.$$

A split's gain is half the sum of child $G^2/(H+\lambda)$ scores minus the parent score, minus $\gamma$. Compared with basic first-order gradient boosting, this uses local curvature and explicit tree penalties, alongside engineering for sparse data and scalable split finding. Hessian validity and regularization matter; the derivation assumes a differentiable objective with suitable curvature behavior.
@@ 768|xgb ensemble lightgbm catboost scikit
At each candidate split, XGBoost evaluates where missing examples should go and learns a default left/right direction according to objective gain. At inference a missing value follows that stored branch. Observed values use the threshold normally. [1]

This avoids requiring a universal mean/median imputation and lets missingness itself be predictive. It does not infer the missing value's true numeric value, and a missingness pattern changing in production can break the learned rule. Sparse matrices also have representation-specific missing/zero semantics: verify whether an absent entry means missing or a genuine numeric zero.

Audit performance by missingness pattern, validate feature availability at prediction time, and retain preprocessing/version consistency. A learned default branch can exploit leakage just as an explicit indicator can.
@@ 769|lightgbm xgb ensemble catboost scikit
GOSS retains large-gradient examples and subsamples small-gradient ones with reweighting, targeting examples that strongly influence split gain. EFB bundles sparse, nearly mutually exclusive features to reduce effective dimensionality. Leaf-wise growth splits the currently most promising leaf; depth-wise growth expands a level more uniformly. [1]

Best-first growth can reduce training loss with fewer leaves but may create deep, specialized branches and overfit small datasets. Limit leaves, depth, and minimum leaf data. XGBoost also supports multiple growth/histogram strategies, so “LightGBM leaf-wise, XGBoost always depth-wise” is outdated as a universal distinction. [2]

Compare actual configurations, categorical handling, memory, training time, and held-out quality. Neither GOSS nor EFB guarantees a speedup or quality match on dense or highly interacting features.
@@ 770|catboost xgb ensemble leak islp
Naive target encoding leaks each row's label into its own feature. Ordered target statistics use a random permutation and compute a row's category statistic from earlier rows, with prior smoothing; ordered boosting also addresses prediction shift caused by reusing labels in residual estimation. [1]

$$enc_i=\frac{\sum_{j<i:c_j=c_i}y_j+a\mu}{\#\{j<i:c_j=c_i\}+a}.$$

For validation/inference, use statistics learned from allowed training data only. Symmetric or oblivious trees use the same split at every node of a depth, so the path can be encoded by a compact bit pattern and evaluated efficiently. This regularizes tree structure but restricts expressiveness. Chronological or group constraints still matter: a random permutation does not make future information legitimate for a time-dependent task.
@@ 771|ensemble linear forecast xgb islp
A standard regression tree predicts a leaf statistic, usually a training-target mean, and its partitions extend indefinitely beyond the observed feature range. Moving farther beyond the last threshold does not create a continuing slope. Forest averages inherit this piecewise-constant extrapolation limitation. [1]

Boosted trees sum corrections, so their prediction is not universally bounded by the original target minimum and maximum; nevertheless, their dependence on an out-of-range feature remains piecewise constant. Thus the prompt's range claim is too broad.

For an upward trend, model the trend separately, predict differences/residuals, include mechanistic structure, or use an extrapolating model when justified. Evaluate on forward time splits beyond the training range. Random splits within the same range cannot reveal extrapolation failure.
@@ 772|shap ensemble scikit islp causal
Correlated predictors can substitute for each other. Permuting one may show little importance because the other retains the signal; alternatively, permutation creates unrealistic combinations. Split-gain importance can distribute credit arbitrarily and favor features with many candidate splits. [1] [2]

SHAP attributes a local prediction relative to a background distribution using averaged marginal contributions over feature coalitions. Its consistency property concerns the specified attribution game, not proof of causal importance. Conditional and interventional feature assumptions answer different questions, especially with dependence.

Inspect grouped permutation importance, dependence-aware attribution choices, and stability across retraining. Explain which background population is used. A precise additive explanation of a biased or leaked model remains an explanation of that model—not evidence that changing the feature would change the real outcome.
@@ 773|linear islp prml scipy causal
One standard Gauss–Markov formulation assumes a model linear in parameters, full-rank regressors, zero conditional error mean, constant conditional error variance, and uncorrelated errors. Random sampling is often used to justify the last conditions; textbook lists group assumptions differently. [1] [2]

$$y=X\beta+\epsilon,\quad E[\epsilon\mid X]=0,\quad \operatorname{Var}(\epsilon\mid X)=\sigma^2I.$$

Heteroscedasticity violates constant variance. If exogeneity and rank conditions still hold, OLS coefficients remain unbiased, but ordinary homoscedastic standard errors are wrong and OLS need not be most efficient among linear unbiased estimators. Use heteroscedasticity-robust covariance or a justified variance model. Normal errors are not required for the Gauss–Markov BLUE result; they support additional exact finite-sample inference.
@@ 774|linear islp prml pca scipy
Nearly dependent columns make $X^TX$ ill-conditioned, so small data perturbations can create large opposing coefficient changes while fitted predictions remain similar on the observed feature manifold. Interpretation becomes unstable even when i.i.d. predictive error is acceptable. [1]

$$VIF_j=\frac1{1-R_j^2},\qquad \operatorname{Var}(\hat\beta\mid X)=\sigma^2(X^TX)^{-1}.$$

Here $R_j^2$ comes from regressing feature $j$ on the others. Perfect collinearity destroys uniqueness; near collinearity inflates uncertainty. Solve with QR/SVD rather than explicitly forming an inverse, and consider ridge, feature grouping, or domain-driven removal. These improve stability but change estimation/interpretation. Extrapolation away from the training correlation structure can still fail badly, so unchanged i.i.d. accuracy is not universal robustness.
@@ 775|linear reg islp prml optim
L1 constrains coefficients with a diamond-like region whose corners lie on coordinate axes; a loss contour often first touches a corner, producing zeros. L2 uses a smooth sphere, usually shrinking coefficients continuously. [1]

$$\mathcal L_{lasso}=\mathcal L+\lambda\sum_j|w_j|,\quad \mathcal L_{ridge}=\mathcal L+\lambda\sum_jw_j^2.$$

L1 has subgradient $\lambda\operatorname{sign}(w)$ away from zero and an interval at zero; proximal updates apply soft thresholding. L2's gradient is $2\lambda w$. Bayesian MAP with independent Laplace priors yields L1, while Gaussian priors yield L2 under matched likelihood scaling. Standardize features when penalties should treat them comparably, and usually handle the intercept separately. Correlated features can make lasso selection unstable.
@@ 776|linear islp reg prml scikit
Elastic net combines sparse selection with quadratic shrinkage. Lasso can arbitrarily choose one feature from a correlated group; an L2 component encourages more stable sharing of coefficients and can help when features greatly outnumber samples. [1]

$$\min_w\frac1{2n}\|y-Xw\|^2+\lambda\left[\alpha\|w\|_1+\frac{1-\alpha}{2}\|w\|_2^2\right].$$

Tune both total penalty and mixing weight inside the validation procedure. Scale features using training folds only. Elastic net does not identify causal variables, and selecting a whole correlated group can be less interpretable than one representative. Compare predictive error, coefficient stability, sparsity, and domain constraints; choose pure ridge when dense small effects are plausible and sparse selection is unnecessary.
@@ 777|linear optim dl prml islp
You **can** optimize squared error after a sigmoid, but it loses logistic regression's convenient convex log-loss objective and can have flat, nonconvex regions. With $p=\sigma(z)$ and half squared error, [1]

$$\frac{dL}{dz}=(p-y)p(1-p),\quad \frac{d^2L}{dz^2}=p(1-p)\left[p(1-p)+(p-y)(1-2p)\right].$$

For $y=0,p=0.9$, the bracket is $0.09-0.72<0$, proving nonconvexity in $z$. Saturation also multiplies the gradient by a small $p(1-p)$. Bernoulli negative log likelihood instead has gradient $p-y$ and Hessian $p(1-p)\ge0$ in linear logits. MSE can be a legitimate probability scoring rule, but is usually an inferior optimization choice here.
@@ 778|calibration linear metrics prml dl
Logistic regression can be well calibrated when the log-odds model is appropriate, data are representative, and estimation is adequate. It is not calibrated by definition under misspecification, regularization, finite samples, or distribution shift. Naive Bayes can double-count correlated evidence; neural networks can produce overconfident logits. [1]

Platt scaling fits a sigmoid to model scores on held-out data; isotonic regression fits a nondecreasing calibration map with more flexibility and greater overfitting risk. Temperature scaling rescales multiclass logits while preserving argmax predictions.

Use a separate calibration split or cross-validation, reliability plots, log loss, and Brier score. Calibration is population-dependent and can degrade under prior shift. A calibrated but uninformative model may predict the base rate for everyone, so evaluate discrimination separately.
@@ 779|svm linear islp prml optim
Slack variables allow examples inside the margin or on the wrong side of the boundary. The soft-margin primal trades margin size against violation cost. [1]

$$\min_{w,b,\xi}\tfrac12\|w\|^2+C\sum_i\xi_i,\quad y_i(w^Tx_i+b)\ge1-\xi_i,\quad \xi_i\ge0.$$

A larger $C$ penalizes violations more strongly, often fitting training data more tightly with a smaller margin and higher variance. This is a tendency rather than a strict monotonic statement for every dataset. A smaller $C$ tolerates violations in exchange for stronger regularization. Feature scaling changes geometry and the effective penalty. For kernel SVMs, tune kernel parameters together with $C$ and use appropriate class weighting when costs are asymmetric.
@@ 780|svm prml linear pml2 scikit
Many algorithms depend only on inner products. A positive-semidefinite kernel supplies those products in an implicit feature space without explicitly constructing all features. [1]

$$e^{-\gamma\|x-x'\|^2}=e^{-\gamma\|x\|^2}e^{-\gamma\|x'\|^2}\sum_{n=0}^{\infty}\frac{(2\gamma x^Tx')^n}{n!}.$$

Each polynomial term corresponds to finite-degree monomials; all degrees together yield an infinite-dimensional representation. Infinite dimension does not mean infinite computational cost per kernel evaluation, nor automatic overfitting: regularization and $\gamma$ control the function class used. The $N\times N$ Gram matrix can still be expensive in data size. Validate scaling and kernel width, and use approximations when sample count makes exact kernel methods impractical.
@@ 781|scikit svm linear cluster pca
Distance-based methods such as kNN, k-means, and RBF SVMs are sensitive to feature units; large-scale coordinates can dominate distances. PCA on covariance is scale-sensitive. Regularized linear models and neural optimization also depend on scales, though standardization is not a logical requirement for every implementation. [1]

Exact axis-aligned trees depend on feature order and are invariant to strictly increasing transformations in ideal arithmetic. Decreasing transformations can reverse split directions. Non-strict transformations may collapse values, and histogram binning, numerical precision, tie handling, and regularization can break practical equivalence.

Fit scalers on training data only, preserve meaningful binary/count semantics where appropriate, and evaluate the pipeline. “Completely invariant” is too strong for all tree implementations and all monotonic transformations.
@@ 782|cluster prml scikit islp pml2
K-means minimizes squared distances to centroids, favoring roughly spherical similarly scaled clusters. GMMs assign soft memberships to Gaussian components and can represent elliptical clusters. DBSCAN groups density-connected points and marks sparse points as noise, handling nonconvex rings but struggling when densities differ under one global radius. Agglomerative clustering repeatedly merges groups according to a chosen linkage. [1]

$$J_{kmeans}=\sum_i\|x_i-\mu_{c_i}\|^2.$$

Single linkage can follow nonconvex shapes but suffers chaining; complete/Ward linkage favors different geometry. Neither cluster count nor a visually attractive plot proves meaningful structure. Choose distance, scaling, noise policy, and hyperparameters based on the task, then assess stability and domain usefulness. GMM component count need not equal the number of semantic classes.
@@ 783|cluster prml islp scikit pml2
The elbow method looks for diminishing reductions in within-cluster squared error, but there may be no clear elbow. Silhouette compares each point's average within-cluster distance $a$ with its nearest alternative-cluster average $b$. [1]

$$s=(b-a)/\max(a,b).$$

It favors certain geometries and can mislead on nonconvex or highly unequal clusters. GMM AIC/BIC compare penalized likelihoods for probabilistic models; they are not k-means objectives and depend on covariance assumptions. Use multiple initializations, resampling stability, downstream utility, and plausible segment sizes. There may be no uniquely “optimal” $K$: two resolutions can answer different business questions. Avoid choosing $K$ on test outcomes that will later be reported as independent evidence.
@@ 784|pca cluster scikit prml islp
PCA learns a global linear projection maximizing variance and provides a direct transform for new points. t-SNE optimizes neighborhood probability agreement in a low-dimensional layout, emphasizing local structure. UMAP builds and optimizes a neighborhood graph and commonly supports transforming new data with a fitted model. [1] [3]

Independent t-SNE fits have arbitrary rotations, reflections, translations, and nonlinear layouts, so test coordinates cannot be compared with training coordinates as though they share axes. Fit a suitable joint or out-of-sample method instead, without leaking test labels into modeling decisions. Neither t-SNE distances between far clusters nor apparent cluster sizes have a simple original-space meaning. Validate representations on downstream tasks, not just attractive two-dimensional plots.
@@ 785|leak cv feast rules mltest
Five leakage paths are fitting preprocessing/feature selection/SMOTE before splitting; allowing the same patient or user into both sides; random splitting when deployment predicts the future; target encoding with the row's own label; and joining tables whose values were updated after prediction time. [1] [2] [3]

Put learned transformations inside each training fold. Split by the deployment unit and time, enforce event-time/as-of joins, and compute target statistics out of fold. Keep label-generation windows and feature timestamps explicit. A feature available in today's database may not have existed when the historical decision was made.

Re-run suspiciously strong models after removing high-risk features and inspect duplicate overlap. Leakage controls should be executable data tests, not only a checklist in the model report.
@@ 786|catboost linear scikit leak prml
One-hot encoding preserves category identity but grows with vocabulary. Frequency encoding is compact but merges categories with equal frequency. Hashing bounds dimension but introduces collisions. Learned embeddings model category similarity but need data and a policy for unseen values. Target encoding uses outcomes and requires strict leakage control. [1]

$$enc(c)=\frac{n_c\bar y_c+\alpha\mu}{n_c+\alpha}.$$

For training rows, compute this statistic from other allowed folds or earlier observations; validation uses training-only statistics. Rare categories shrink toward the global mean. Choose by cardinality, sample size, estimator, memory, and update frequency. Distinguish a genuinely unseen category from missing data. Temporal/category drift can invalidate frequency or target statistics even if the encoding code is unchanged.
@@ 787|linear forecast scikit d2l prml
Map a periodic value to two coordinates on the unit circle so the endpoints meet. For period $T$, [1]

$$u=\sin(2\pi t/T),\qquad v=\cos(2\pi t/T).$$

For hours, 23 and 0 are separated by one hourly angle rather than numeric distance 23. Both sine and cosine are needed to identify phase unambiguously. A linear model on these two features represents one sinusoidal cycle; add harmonics for more complex periodic patterns or use categories when each hour has unrelated effects. Calendar months have unequal lengths, and daylight-saving/time-zone behavior needs explicit handling. Cyclical encoding does not replace trend, holidays, or interactions such as weekday-by-hour effects.
@@ 788|prml scikit causal islp leak
MCAR means missingness is independent of observed and missing values. MAR allows dependence on observed variables, after conditioning on which missingness does not depend on the unobserved value. MNAR retains dependence on the missing value itself. These are assumptions about the missingness mechanism, not labels identifiable from the observed table alone. [1]

A missingness indicator can be highly predictive when measurement is a decision—for example, a specialist orders a test only for concerning cases. That signal may fail if workflow changes and can leak post-outcome actions. Imputation plus an indicator can help prediction but does not solve MNAR identification. Examine collection timing, subgroup missingness, and sensitivity to plausible unobserved values before interpreting effects.
@@ 789|calibration prml metrics linear scipy
Assume sampling changes only class priors, retaining the same class-conditional feature distributions. Let $p_s$ be the model probability under sampled prior $\pi_s$ and let population prior be $\pi$. Correct the odds by the prior-odds ratio. [1] [2]

$$a=\frac{\pi(1-\pi_s)}{(1-\pi)\pi_s},\qquad p=\frac{ap_s}{1-p_s+ap_s}.$$

If all positives are retained and negatives have retention $r=0.01$, then $a=r$. A sampled-model probability 0.5 becomes $0.01/1.01\approx0.009901$. The correction assumes the sampled-model scores are calibrated and sampling is independent of features within class. Feature-dependent sampling, prior drift, and model misspecification require more than this adjustment. Choose decision thresholds using population costs and prevalence.
@@ 790|cluster calibration ensemble leak metrics
SMOTE interpolates between minority examples in feature space. In high dimensions, nearest neighbors may not be meaningful; interpolation can cross class boundaries or create impossible categorical/structured combinations. It also changes the training class balance, which can distort probability calibration. [1] [2]

XGBoost is not inherently well calibrated, so the question's premise should not be assumed. Compare class weighting, threshold tuning, balanced sampling, and SMOTE using a realistic untouched validation distribution. Apply any resampling only within training folds; synthetic points generated before splitting can leak neighbor information.

Judge precision/recall at operational capacity and probability quality separately. Synthetic sample count is not equivalent to new independent information, and improvement on a balanced test set may disappear at the true deployment prevalence.
@@ 791|metrics calibration prml islp scipy
ROC plots TPR against FPR; precision–recall plots precision against recall. Prevalence enters precision explicitly. [1]

$$precision=\frac{\pi\,TPR}{\pi\,TPR+(1-\pi)FPR},\qquad \Delta FPR=\Delta FP/N_-.$$

With 100 positives and 100,000 negatives, TPR 0.9 and FPR 0.01 yield 90 true positives and 1,000 false positives: precision is only 8.26%. A 1% FPR can therefore be operationally unacceptable. ROC-AUC is not mathematically invalid; it summarizes ranking across thresholds and can hide the operating region of interest. PR curves are prevalence-sensitive. Specify whether “PR-AUC” means trapezoidal area or average precision, and compare models on the same target population.
''')
