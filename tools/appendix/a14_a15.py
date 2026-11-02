from author import write
write(r'''
@@ 918|linear pca prml scipy islp
Duplicate columns or more columns than rows make $X$ rank-deficient, so $X^TX$ has zero eigenvalues and the ordinary inverse does not exist. Least-squares fitted values can still be unique even when coefficients are not; the pseudoinverse selects a minimum-norm solution. [1]

For any nonzero $v$ and $\lambda>0$,

$$v^T(X^TX+\lambda I)v=\|Xv\|^2+\lambda\|v\|^2>0.$$

Thus the fully penalized matrix is positive definite and invertible. If an intercept is exempt, analyze that block separately. Ridge improves conditioning but introduces shrinkage bias. Use a stable linear solve/QR/SVD rather than explicitly constructing the inverse in production numerical code.
@@ 919|linear reg prml islp optim
Under the **constraint** $\beta_1+\beta_2=4$, write $\beta_1=2+a,\beta_2=2-a$. Then the L2 penalty is $8+2a^2$, minimized at $(2,2)$. The L1 penalty equals four for every nonnegative pair summing to four, so it does not prefer a unique split. [1]

For actual penalized regression, the total coefficient is generally shrunk and need not equal four. If $s=\|X_1\|^2$ and loss is squared error plus $\lambda(\beta_1^2+\beta_2^2)$, symmetry gives

$$\beta_1=\beta_2=\frac{4s}{2s+\lambda}.$$

Lasso may select one duplicate due to solver behavior, but any minimizing nonnegative allocation with the same total has equal objective. This illustrates nonuniqueness, not guaranteed selection of one particular feature.
@@ 920|linear reg islp prml scikit
Intercepts are usually unpenalized because they set the baseline level rather than feature sensitivity. Penalizing them makes the fit depend on arbitrary target centering or feature-origin conventions and shrinks the baseline toward zero. [1]

For an intercept-only squared-loss model, an unpenalized intercept equals the sample mean; adding $\lambda b^2$ yields $b=\sum_i y_i/(N+\lambda)$ under the unnormalized loss. That may be inappropriate when zero has no special meaning.

“Never” is too strong: a prior on the intercept can be intentional, especially with sparse data or hierarchical models. State that prior and scaling explicitly. Library defaults vary, and some implementations regularize an intercept through a synthetic feature, so inspect the estimator rather than assuming the convention.
@@ 921|linear prml optim islp reg
If a separating direction makes every signed margin positive, multiplying its weights by larger constants drives all training probabilities toward correct extremes and log loss toward zero. No finite unregularized maximum-likelihood solution is attained under complete separation. [1]

$$L(cw)=\sum_i\log(1+e^{-c y_iw^Tx_i})\to0\quad(c\to\infty).$$

Numerical solvers may report huge coefficients, warnings, or an apparent converged value caused by finite precision. An L2 penalty grows quadratically with scale, producing a finite optimum for penalized directions. Intercept handling and degenerate all-one-class data still need care. Use regularization, inspect separation and sample size, and do not interpret enormous coefficients as precise evidence of an enormous causal effect.
@@ 922|linear ensemble xgb islp prml
The diagonal boundary $x_1+x_2=0$ is a linear separator, so logistic regression can represent it with coefficients $(1,1)$ and zero intercept, up to scale. Axis-aligned trees approximate the diagonal by a staircase of rectangular regions, potentially requiring many leaves. [1] [2]

For same-sign quadrants $x_1x_2>0$, no single linear boundary separates the two positive quadrants from the two negative ones. Adding the interaction feature $z=x_1x_2$ makes a linear threshold in $z$ sufficient. A small tree can represent quadrant logic directly.

Performance still depends on noise, training data, regularization, and the tested range. This comparison illustrates inductive bias, not a universal ranking of trees and linear models.
@@ 923|ensemble catboost lightgbm xgb islp
A high-cardinality feature offers many candidate partitions, increasing the chance of finding an apparently good split by noise. A unique transaction ID can isolate rows perfectly while providing no reusable predictive structure. Impurity-based importance inherits related selection bias. [1]

Gain ratio divides information gain by split information, discouraging some high-branching splits but not eliminating all selection bias. Minimum leaf sizes, depth/leaf limits, penalties, and validation reduce memorization. CatBoost's ordered target handling addresses a different but related leakage/prediction-shift problem; it is not simply a cardinality penalty. [2]

Audit ID-like variables, compare permutation importance on held-out data, and use group/time splits. A random split can make repeated-entity memorization look useful when deployment must generalize to unseen entities.
@@ 924|ensemble xgb linear svm cluster
A strictly increasing transform preserves feature order, so an exact tree splitter can form the same left/right partitions with transformed thresholds. Log on positive values and cube are examples. In ideal arithmetic the fitted partitions are unchanged. [1]

Histogram binning, finite precision, ties, random thresholds, and implementation details can break exact practical invariance. The result also says nothing about changing the **target**.

Linear regression changes its hypothesis class from linear-in-$x$ to linear-in-$f(x)$. kNN distances change, and neural optimization/representations can change. Fit transforms only where mathematically defined and use training-only learned parameters. Tree invariance does not mean preprocessing is irrelevant: missing values, categorical encoding, and leakage still matter.
@@ 925|ensemble linear metrics forecast prml
MSE on raw targets selects arithmetic means in leaves and rewards splits according to raw squared-error reduction. MSE on log targets instead selects mean log values and uses a different error geometry, so both leaf values and split choices can change. [1]

$$\exp\left(\frac1n\sum_i\log y_i\right)$$

is the geometric mean, not generally the conditional arithmetic mean of $Y$. Exponentiating a predicted mean log target therefore introduces retransformation bias for mean forecasting; a distributional correction or smearing estimate may be appropriate. Log transforms require positive targets or an explicitly justified alternative. They emphasize relative errors and downweight large absolute values, which may or may not match the application's costs.
@@ 926|metrics linear prml forecast scipy
At fixed $X=x$, differentiating $E[(Y-a)^2]$ gives $2(a-E[Y])$, so squared loss is minimized by the conditional mean. The absolute-loss subgradient is $P(Y<a)-P(Y>a)$; zero belongs to it at a median. [1]

$$\rho_\tau(u)=u(\tau-\mathbf1[u<0]),\qquad \arg\min_aE[\rho_\tau(Y-a)]\in Q_\tau(Y).$$

Thus pinball loss estimates a conditional quantile. Huber is quadratic near zero and linear in the tails, balancing efficiency and outlier robustness, but generally estimates neither the exact mean nor median without additional assumptions. Choose the target functional according to the decision: inventory underage/overage costs often motivate a quantile rather than a mean.
@@ 927|metrics linear islp prml cv
Test $R^2$ is negative when squared prediction error exceeds the error of predicting that test set's mean. An intercept-free regression can also have negative conventional centered $R^2$ even in sample. A constant target makes the denominator zero, requiring an explicit convention. [1]

$$\bar R^2=1-(1-R^2)\frac{n-1}{n-p-1}$$

is the usual adjusted in-sample formula for $p$ predictors plus an intercept, when the denominator is positive. It penalizes additional predictors but is not a substitute for held-out validation and is not directly appropriate for arbitrary model complexity. Report absolute error, baseline comparison, and split design. A high $R^2$ on a nonstationary or leaked time series can be misleading.
@@ 928|prml cluster scikit scipy pca
The claim needs sampling assumptions; it is not true for arbitrary chosen points or an exponentially growing dataset. For independent uniform points in a unit cube, each squared coordinate difference has mean $1/6$ and finite variance, so the sum has mean $d/6$ and relative fluctuation $O(d^{-1/2})$. [1]

For a fixed finite number of sampled pairs, their distances concentrate near $\sqrt{d/6}$, making the farthest/nearest ratio approach one in probability. This reduces relative distance contrast.

Real data may lie on a low-dimensional manifold or have informative sparse coordinates, so Euclidean distance is not universally meaningless. Feature selection, metric learning, appropriate normalization, and domain-specific representations can restore useful neighborhoods. State sample-size growth and distribution before claiming an asymptotic result.
@@ 929|faiss sbert prml mrl irl
Expand the squared norm:

$$\|u-v\|^2=\|u\|^2+\|v\|^2-2u^Tv=2(1-\cos(u,v))$$

when both norms equal one. Minimizing Euclidean distance then produces the same ordering as maximizing cosine similarity, with ties preserved. [1]

For unnormalized vectors, maximum inner product search also rewards candidate magnitude; cosine removes that magnitude, and Euclidean distance adds the candidate norm term. They can rank items differently. Normalizing vectors is therefore a modeling choice, not merely an implementation optimization. A trained retrieval model may intentionally encode popularity or confidence in norms. Use an index and distance convention matching training, and handle zero vectors explicitly because cosine normalization is undefined there.
@@ 930|pca linear prml scikit islp
Construct balanced classes with $x_1\sim N(0,100)$ independent of class and $x_2=+1+\epsilon$ for one class, $-1+\epsilon$ for the other, with small noise. The largest variance lies along $x_1$, so one-component PCA discards the nearly perfect class signal in $x_2$. [1]

LDA uses labels and maximizes between-class separation relative to within-class scatter:

$$J(w)=\frac{w^TS_Bw}{w^TS_Ww}.$$

Here it favors the vertical direction. PCA optimizes reconstruction variance, not classification. LDA has its own assumptions and can require regularization when within-class covariance is singular. Fit either transform only on training folds; using test labels for LDA would create direct leakage.
@@ 931|svm prml linear islp scikit
In the SVM dual, only examples with nonzero multipliers contribute to $w=\sum_i\alpha_i y_ix_i$. Removing examples known to have zero multipliers can leave the same optimum feasible and optimal under the same objective and constraints. [1]

However, the prompt's “zero change” needs qualifications: changing sample count can change effective regularization in average-loss parameterizations; numerical tolerances and nonunique intercepts can matter; and identifying non-support vectors from an approximate fit can be imperfect. Soft-margin support vectors may lie inside the margin or be misclassified.

Do not remove points merely because a heuristic says they look far away. Verify exact objective conventions and refit behavior, especially when class weights or per-example penalties change with the dataset.
@@ 932|prml irl scikit scipy islp
A zero estimated class-conditional word probability makes the whole multinomial likelihood zero if that word occurs. Additive smoothing reserves positive probability for unseen vocabulary terms. [1]

$$P(w\mid c)=\frac{count(w,c)+\alpha}{\sum_vcount(v,c)+\alpha V},\quad score(c)=\log P(c)+\sum_wcount(w,d)\log P(w\mid c).$$

Log space prevents numerical underflow from multiplying many small probabilities; it does not fix zeros by itself, since $\log0=-\infty$. Smoothing and log arithmetic solve different problems. Define the vocabulary from training data and specify handling of out-of-vocabulary terms. Conditional independence is an approximation; correlated words can make Naive Bayes probabilities overconfident even when classification is useful.
@@ 933|cluster prml scipy islp scikit
K-means uses squared distances, so a far outlier has disproportionately large influence and can pull a centroid or consume a cluster. Its alternating assignment/update procedure can converge to a local optimum depending on initialization. [1]

K-medoids restricts centers to observed points and can use more robust dissimilarities; k-medians minimizes Manhattan distances and updates coordinates to medians. These reduce some outlier influence but do not make arbitrary contamination harmless. Medoid optimization can be more expensive.

Use k-means++ or multiple starts, scale features sensibly, inspect extreme points, and compare objective/stability across seeds. Decide whether anomalies should form a meaningful cluster or be excluded using a justified data-quality policy rather than deleting them solely to improve a clustering score.
@@ 934|scikit cluster prml metrics dl
Isolation Forest scores points by average path length under random feature/threshold partitions: unusual points can often be isolated with fewer splits. One-class SVM learns a support boundary; LOF compares local density to neighbors; autoencoders use reconstruction error, which may also be low for some anomalies. [1]

Isolation Forest's cost is better stated using tree count $T$ and subsample size $\psi$: training is roughly $O(T\psi\log\psi)$ under typical balanced behavior, and scoring $O(NT\log\psi)$. It is not universally $O(N\log N)$ or guaranteed shallow for every anomaly.

Evaluate contamination assumptions, high-dimensional noise, and deployment novelty separately. Thresholds need operational calibration; an anomaly score is not automatically a fraud probability or a reason to take action.
@@ 935|calibration metrics prml linear scipy
Take labels $(1,1,0,0)$ with predictions $(0.02,0.02,0.01,0.01)$. Every positive outranks every negative, so ROC-AUC is exactly one. But the probabilities are severely underconfident for positives. [1]

$$logloss=-\tfrac14[2\log0.02+2\log0.99]\approx1.961.$$

The constant 0.5 predictor has log loss about 0.693, so the perfectly ranking model is worse under this probability score. Calibration checks frequency conditional on predicted probability, not ranking. A monotone calibration map can preserve AUC while improving probability quality. Use separate calibration data and test distribution; perfectly fitting a calibration map to these four observed points would not establish generalization.
@@ 936|irl metrics prml scipy sbert
For relevance sequence $(1,0,1,0,1)$, precision at relevant ranks is $P@1=1$, $P@3=2/3$, and $P@5=3/5$. If there are exactly three relevant documents in the corpus, [1]

$$AP=\frac{1+2/3+3/5}{3}=\frac{34}{45}\approx0.7556.$$

Precision@5 is $3/5=0.6$ and recall@5 is $3/3=1$. If additional relevant documents exist outside the returned list, recall decreases and AP's denominator must reflect the chosen definition. MAP averages AP over queries. “AP@K” conventions vary between total relevant count and a capped denominator; specify yours and do not compare numbers computed with different conventions. Unjudged documents also need a documented policy.
@@ 937|metrics prml scipy snorkel islp
Raw agreement can be high when annotators mostly choose the dominant class. If both independently label 95% of items negative, expected agreement from their marginals is $0.95^2+0.05^2=0.905$, even without useful case-level agreement.

$$\kappa=\frac{p_o-p_e}{1-p_e}.$$

Cohen's kappa corrects relative to that marginal-chance model for two raters. Krippendorff's alpha generalizes disagreement accounting to multiple raters, missing annotations, and distance functions for different label scales. [1]

Neither is a universal annotation-quality score: prevalence, rater bias, and ambiguous guidelines affect them. Report confusion matrices, class-specific agreement, adjudication findings, and label uncertainty. High agreement can still reflect shared misunderstanding rather than correctness.
@@ 938|prml scikit cluster metrics mltest
Uncertainty sampling selects small margins or high predictive entropy. Query-by-committee selects disagreement among models. Core-set/diversity selection seeks coverage of representation space, avoiding a batch full of near-duplicates. [1]

$$H(p)=-\sum_cp_c\log p_c.$$

A model may be uncertain because an example is corrupted, intrinsically ambiguous, or outside the intended task; labeling it may add little useful information. Combine uncertainty with representativeness, diversity, quality filtering, and task-value weighting. Reserve a randomly sampled evaluation set so selection bias does not contaminate performance estimates.

Run iterative labeling rounds, retrain, and compare learning curves against random sampling at equal annotation cost. Account for expert effort and disagreement, not just number of labels purchased.
@@ 939|snorkel prml reg calibration mltest
Weak-supervision labeling functions emit noisy labels or abstain using rules, dictionaries, heuristics, or other models. A label model estimates their reliability and dependencies, then combines votes into probabilistic training labels. [1]

Correlated functions must not be treated as independent evidence: ten variants of the same rule do not equal ten independent experts. Identifiability assumptions and limited gold labels matter; inspect coverage, conflicts, and held-out accuracy.

Label smoothing is different: it mixes a hard target with a specified distribution, such as $(1-\epsilon)y+\epsilon/K$. It regularizes confidence but does not learn which labeling function is trustworthy. Preserve a clean human-reviewed test set and audit slices where weak rules share blind spots before trusting downstream model gains.
@@ 940|forecast scipy prml cv linear
Strict stationarity means every finite-dimensional joint distribution is invariant to a common time shift. Weak stationarity requires finite second moments, constant mean, and covariance depending only on lag. Strict stationarity does not imply finite variance. [1]

ADF tests a unit-root null using an augmented autoregression; KPSS tests a stationarity null around a specified level or trend. Their nulls point in opposite directions. Failure to reject either is not proof of stationarity: short samples, structural breaks, lag choices, and deterministic-trend specifications affect power.

Use plots, domain knowledge, transformations, and rolling diagnostics alongside tests. A stationary-looking training window does not guarantee a stable deployment distribution, and differencing should match the predictive objective rather than merely force a p-value below a threshold.
@@ 941|forecast scipy prml linear cv
Independent random walks can share long smooth excursions by chance. Regressing their levels violates usual stationary-error assumptions, so high $R^2$ and conventional t-statistics can be spurious. Differencing removes a unit root but may discard a meaningful long-run relation if one exists. [1]

Cointegration means a nontrivial linear combination of integrated series is stationary, for example $Y_t-\beta X_t$. Engle–Granger tests residual behavior with appropriate nonstandard critical values; Johansen handles multiple series/relations.

A cointegrating relation is not automatically a profitable pairs strategy. Test stability, estimate using past data only, model execution costs, and monitor breaks. Repeatedly searching many pairs creates selection bias; a favorable residual test on the chosen pair needs independent validation.
@@ 942|forecast scipy prml linear cv
ACF measures correlation at each lag, including indirect effects through intermediate lags. PACF measures lag association after accounting for intervening lags. For ideal stationary models, an AR($p$) has PACF cutoff after $p$, while an MA($q$) has ACF cutoff after $q$; the other function generally tails off. [1]

ARIMA first differences $d$ times, then fits AR and MA structure. Real finite samples, mixed ARMA processes, seasonality, and structural breaks rarely produce perfect cutoffs. Use plots as candidates, then compare information criteria and rolling-origin forecasts, inspecting residual autocorrelation.

Over-differencing can introduce artificial dependence and inflate uncertainty. Select seasonal components and deterministic terms deliberately rather than reading one significant bar as a complete model specification.
@@ 943|forecast prml scipy cv metrics
Returns can have little linear autocorrelation while their magnitudes cluster, meaning conditional variance changes over time. GARCH(1,1) models an innovation as a standardized shock times a dynamic scale. [1]

$$r_t=\mu_t+\epsilon_t,\quad \epsilon_t=\sigma_tz_t,\quad \sigma_t^2=\omega+\alpha\epsilon_{t-1}^2+\beta\sigma_{t-1}^2.$$

With nonnegative parameters and $\alpha+\beta<1$, the usual finite unconditional variance is $\omega/(1-\alpha-\beta)$ under standard assumptions. Near-one persistence gives slow volatility decay. Heavy-tailed shocks and asymmetric leverage effects may require extensions. A conditional-variance forecast is not a direction forecast; evaluate likelihood, interval calibration, and risk forecasts on forward splits rather than claiming volatility predictability implies return predictability.
@@ 944|forecast attention scaling prml cv
Simple linear baselines can excel when benchmarks are dominated by stable trend/seasonality and complex models fail to exploit temporal ordering or are poorly tuned. This is evidence about benchmark/model configurations, not proof that transformers cannot forecast. [1]

Patch-based models encode short windows rather than individual scalar timestamps, exposing local patterns and reducing attention token count. Foundation models add broad pretraining and normalization/tokenization choices; Chronos, TimesFM, and Moirai do not share one identical architecture or objective.

Compare with seasonal naive, linear, and statistical baselines using leak-free rolling evaluation, matched horizons, covariates, and tuning budgets. Check cross-dataset pretraining overlap. Improved tokenization alone does not solve regime shift, missing future covariates, or the mismatch between point-error metrics and business decisions.
@@ 945|forecast codec prml ddpm sentencepiece
Discrete bins let a sequence model predict a categorical distribution over quantized values, enabling token-based likelihood/sampling but introducing resolution, range, and scaling choices. Vector or spherical quantizers encode richer latent patterns, requiring a trained encoder/codebook and reconstruction validation. Continuous projections preserve numeric input precision and use regression or probabilistic heads. [1] [2]

Uniform quantization has step $\Delta$ and bounded rounding error $\Delta/2$ within range; clipping outliers violates that bound. Diffusion/flow heads can represent multimodal futures at additional sampling cost.

Normalize using only available history and evaluate tail events, cross-series scale transfer, interval coverage, and calibration. A discrete vocabulary is not inherently more expressive for financial data; compare uncertainty and decision quality, not only token prediction loss.
@@ 946|leak forecast feast cv rules
Linear interpolation between a missing point and a later observed point uses future information. In a historical backtest, that future value was unavailable when the prediction would have been made. A centered rolling window creates a similar leak. [1]

A trailing 20-day z-score is valid only if every included observation was available at decision time; predicting before today's close means shifting the window to exclude today's closing value. Forward fill may be causal but can create stale values, so include age/missingness indicators and expiration rules.

Reconstruct data as-of timestamps, account for publication/revision delays, and test preprocessing with a streaming replay. Running a causal model on noncausal features still produces an invalid backtest.
@@ 947|cv forecast leak causal metrics
Using today's index constituents for historical training conditions on future survival and selection. It excludes delisted or failed firms and includes companies before they qualified for the index, changing both the investable universe and the distribution of outcomes. [1]

Use point-in-time membership, historical identifiers, delisting returns, corporate actions, and availability timestamps. A survivorship-free price file alone is insufficient if features or universe filters still use current information. Rebalances and liquidity constraints must match what was tradable then.

The bias often inflates apparent performance but its exact direction depends on strategy. Validate on forward periods with a frozen decision process and include transaction costs. State universe construction explicitly so a reported metric can be interpreted and reproduced.
@@ 948|forecast metrics cv scaling prml
Prediction accuracy ignores whether an expected move exceeds trading costs before the signal decays. Net value subtracts spread, fees, impact, slippage, and latency losses from gross forecast benefit. [1]

$$\alpha_{net}\approx\alpha_{gross}-cost_{spread}-cost_{fees}-cost_{impact}-cost_{delay}.$$

The square-root impact expression $impact\propto\sigma\sqrt{Q/V}$ is an empirical approximation with regime/market dependence, not a universal law. High turnover repeatedly pays costs; passive orders avoid immediate spread crossing but face fill uncertainty and adverse selection.

Backtest with realistic timestamps, available liquidity, queue/fill assumptions, and capacity. Evaluate net performance and risk at different turnover constraints. A better five-minute forecast can be economically worse if it requires faster, larger, or more frequent trades than the baseline.
@@ 949|linear pca prml scipy forecast
Let $F$ contain cross-sectional factor exposures and $a$ the raw alpha vector. Regress alpha on exposures and keep the residual, using an intercept and weights where appropriate. [1]

$$\hat\beta=(F^TWF)^{-1}F^TWa,\qquad a_\perp=a-F\hat\beta,\qquad F^TWa_\perp=0.$$

Use QR/SVD for numerical stability; Gram–Schmidt expresses the same projection idea with an orthonormal basis. This removes linear exposure to the included factors under the chosen weighting, not every nonlinear or future risk. Rank deficiency, changing exposures, constraints, and transaction costs matter.

Neutralizing predictions is different from constructing a factor-neutral portfolio: portfolio weights and risk estimates must also satisfy exposure constraints. Validate whether the removed component was unwanted risk or legitimate predictive signal.
@@ 950|metrics forecast linear prml optim
Squared loss gives large residuals quadratic influence, so extreme returns can dominate training. Huber becomes linear beyond a threshold, reducing outlier influence while remaining smooth near zero. Correlation losses target cross-sectional ordering/association, and Sharpe-inspired objectives target return relative to volatility. [1]

$$L_\delta(e)=\begin{cases}e^2/2,&|e|\le\delta\\\delta(|e|-\delta/2),&|e|>\delta.\end{cases}$$

These objectives optimize different quantities. Correlation ignores absolute calibration and can be unstable in tiny batches; Sharpe losses can exploit noisy denominator estimates or ignore costs/tail risk. Differentiable rank approximations introduce their own smoothing bias. Evaluate net portfolio outcomes, robustness, and uncertainty with temporal validation; a training loss aligned with one backtest statistic is not a guarantee of live performance.
''')
