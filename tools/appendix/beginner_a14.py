from expand import write
write(r'''
@@ 918
**Duplicate columns create different coefficient vectors with identical predictions.** If columns one and two are equal, predictions depend on their coefficient sum. Weights (1,3), (2,2), and (4,0) all produce the same fitted contribution. Data alone cannot distinguish those choices. [1]

Algebraically, the vector v=(1,-1) lies in the null space: Xv=0. Consequently $X^TXv=0$, so the matrix has a zero eigenvalue and cannot be inverted. When p exceeds n, the column rank is at most n, again leaving unidentified coefficient directions. Least-squares predictions can still be defined through a pseudoinverse; failure of the ordinary inverse is not failure of the entire regression problem.

**Ridge assigns a cost to every nonzero direction.** For lambda greater than zero, $v^T(X^TX+\lambda I)v=\|Xv\|^2+\lambda\|v\|^2>0$. Even a direction invisible to X now has positive curvature. Thus the fully penalized matrix is positive definite and invertible.

This changes the estimation problem by preferring smaller coefficients; it is not free recovery of information absent from the data. Exempting an intercept needs separate treatment. In software, solve the system using stable numerical routines instead of explicitly forming an inverse, especially with nearly dependent columns.
@@ 919
**When features are identical, the prediction sees only their sum; the penalty decides how to split it.** First impose the question's constraint that the sum equals four. Write the two coefficients as 2+a and 2-a. Their squared sum is $8+2a^2$, minimized uniquely at a=0. Ridge therefore prefers an equal split among equivalent directions under that constraint. [1]

For nonnegative coefficients summing to four, the L1 penalty is always four. The allocations (4,0), (3,1), and (2,2) are equally penalized. A lasso solver may return a sparse endpoint, but the objective does not uniquely demand it or identify which duplicate deserves the weight.

**Actual penalized regression also shrinks the total.** Let $s=\|X_1\|^2$ and let the total coefficient be t. The residual sum of squares is $s(4-t)^2$. The smallest L2 penalty for that total is $\lambda t^2/2$. Differentiating gives $2s(t-4)+\lambda t=0$, so $t=8s/(2s+\lambda)$ and each coefficient is $4s/(2s+\lambda)$.

Thus “ridge gives exactly two and two” is correct for the fixed-sum illustration, not for arbitrary positive regularization in the unconstrained fit. Loss normalization changes the numerical interpretation of lambda, so state the objective before comparing coefficients.
@@ 920
**An intercept represents the baseline, while slopes represent sensitivity to features.** If every target is shifted upward by 100, an unpenalized intercept can shift upward by 100 without changing the fitted slopes. Penalizing it makes that arbitrary choice of target origin affect the amount of shrinkage. [1]

For an intercept-only model, minimize $\sum_i(y_i-b)^2$. Differentiation gives b equal to the sample mean. Adding $\lambda b^2$ yields $b=\sum_i y_i/(N+\lambda)$, pulling the baseline toward zero. If the targets are temperatures expressed in different units, zero may have no privileged statistical meaning.

This is why many regression implementations leave the intercept unpenalized and regularize feature coefficients after appropriate centering or scaling. It also helps separate baseline prevalence from the effects of predictors in logistic models.

**“Never” is too strong.** A Bayesian or hierarchical model may deliberately put a prior on the intercept, especially when observations are sparse. That can be useful if the prior's center and strength have a defensible interpretation. Check the library convention: some estimators represent the intercept through a synthetic feature that is penalized. The rule is to make baseline shrinkage intentional rather than accidental.
@@ 921
**Perfect separation lets likelihood improve forever by increasing confidence.** Suppose a vector w gives every training example a positive signed margin $y_iw^Tx_i$, with labels encoded as -1 and +1. Multiplying w by c preserves all classifications while pushing predicted probabilities closer to the observed zero/one labels. [1]

Each logistic loss term is $\log(1+e^{-c y_iw^Tx_i})$. Since every signed margin is positive, every term approaches zero as c grows. For any finite c, a larger value improves the objective. There is an infimum but no finite unregularized solution attaining it under complete separation.

**Classification accuracy can therefore hide estimation failure.** Accuracy is already 100%, yet coefficients continue growing. A numerical solver may stop because of tolerances or overflow, not because the finite maximum-likelihood estimate exists. Very large coefficients in a small dataset can reflect separation rather than strong, precisely estimated effects.

An L2 penalty grows quadratically with coefficient magnitude, balancing the decreasing data loss and producing a finite optimum in penalized directions. Check intercept treatment and one-class degeneracies as well. Regularization controls the confidence and coefficient scale; it does not prove the separating pattern will generalize to new observations.
@@ 922
**Model families make different shapes easy to represent.** The boundary $x_1+x_2=0$ is one straight diagonal line. A logistic model with weights (1,1) and zero intercept places its 0.5 threshold exactly on that line. An axis-aligned tree can split only on one coordinate at a time, so it approximates the diagonal with rectangular steps. [1] [2]

With limited leaves, those steps create approximation error near the diagonal. More leaves improve the approximation but require more data to estimate reliably. This is a reason to prefer a simple linear model when the real structure is close to linear, not a theorem about every finite noisy dataset.

**Now reverse the geometry.** For the same-sign rule $x_1x_2>0$, the upper-right and lower-left quadrants are positive while the other two are negative. One line cannot separate both positive regions from both negative regions. A tree can split at x1=0 and then x2=0 to express the rule with four leaves.

Adding the feature $z=x_1x_2$ makes the same problem linearly separable in z. Feature engineering can therefore change what a simple model represents. Compare the input representation and inductive bias together rather than treating model names as a universal performance ranking.
@@ 923
**Trying more possible splits creates more opportunities to fit noise.** A high-cardinality categorical feature may offer many candidate partitions. A unique transaction ID can even isolate individual records, creating pure training leaves without learning a pattern that transfers to future transactions. [1]

Imagine ten randomly labeled examples, each with a unique ID. A split into ten ID branches can achieve zero training impurity regardless of whether the IDs contain real predictive information. The apparent information gain reflects memorization, not necessarily useful signal.

Gain ratio divides information gain by a measure of how much the split fragments the data, discouraging some high-branching choices. Minimum leaf size, limited depth or leaves, and regularization similarly restrict unsupported complexity. They reduce particular failure modes rather than guaranteeing unbiased feature selection.

**Separate cardinality from target leakage.** CatBoost's ordered target statistics address leakage and prediction shift in categorical encoding; they are not simply a generic cardinality penalty. Validate ID-like features with time or unseen-entity splits that match deployment. Held-out permutation importance can reveal whether an apparently dominant feature actually contributes outside training. A random split containing the same entities on both sides may reward memorization that will fail on new entities.
@@ 924
**An exact tree split depends on ordering, not the numerical spacing between values.** If positive values are 1, 2, and 100, a threshold separating the first two from the last creates the same partition after applying log or cube. A strictly increasing transform preserves every comparison, so the same candidate left/right partitions remain available. [1]

The transformed threshold need not numerically match the old threshold, but it represents the same partition. Under ideal exact splitting with the same tie handling, split gains and predictions can therefore remain unchanged.

**Other models use more than order.** KNN uses distances, so compressing 100 with a logarithm changes its relative proximity. Linear regression on log(x) describes a curved relationship in original x, not the original linear hypothesis class. Neural-network representations and optimization can also change.

Practical tree implementations may use approximate histogram bins, random thresholds, finite precision, or tie conventions that break exact invariance. Missing values and categorical encodings add separate considerations. Most importantly, this argument concerns a feature transformation, not transforming the target. Fit learned preprocessing only on training data and use transforms only where they are defined. Tree scale invariance is a useful property with assumptions, not permission to ignore all preprocessing.
@@ 925
**Changing the target changes what errors are expensive.** A tree trained with raw squared error treats a miss of 100 as equally costly whether the true value is 200 or 20,000. Squared error on log targets instead emphasizes multiplicative differences. Doubling and halving values become comparable changes in log space. [1]

Consider one leaf containing targets 1 and 9. Its raw-MSE optimum is the arithmetic mean, 5. In log space, the optimum is $(\log1+\log9)/2=\log3$, whose exponent is 3: the geometric mean. These are different predictions for the same data.

**Splits change too.** The tree chooses partitions by reduction in its loss. A split that isolates one huge raw target may be highly valuable under raw MSE and less dominant after logarithmic compression. Thus feature-order invariance provides no invariance guarantee for target transformation.

Exponentiating an estimated conditional mean of log(Y) generally does not produce the conditional arithmetic mean of Y. If mean forecasting is required, a suitable retransformation or distributional correction may be needed and must be validated. Logs also require positive values or a justified alternative transformation. Choose the target scale according to the application's error costs rather than merely making a distribution look more symmetric.
@@ 926
**A loss function chooses which summary of the conditional outcome distribution you learn.** Fix X=x and predict a constant a for that conditional distribution. Expanding squared loss gives $E[Y^2]-2aE[Y]+a^2$. Differentiating in a yields $2(a-E[Y])$, so the optimum is the conditional mean. [1]

For absolute loss, moving a slightly right increases the distance to observations on its left and decreases the distance to observations on its right. The expected slope is $P(Y<a)-P(Y>a)$. A median balances these sides; with point masses, the subgradient formulation handles nonunique medians.

**A small example shows the difference.** For outcomes 0, 0, and 9, the mean is 3 while the median is 0. Squared loss spreads the influence of the large outcome across the prediction; absolute loss resists that pull. Neither answer is universally more correct: they answer different decision problems.

Pinball loss weights underprediction by tau and overprediction by 1-tau. Its expected slope is $F(a)-\tau$ at continuous points, so the optimum satisfies $F(a)=\tau$. This gives the desired quantile. Huber transitions from quadratic to linear error, providing a robustness compromise but not generally the exact mean or median. Select the loss according to the quantity and costs the application needs.
@@ 927
**R-squared compares your squared error with a particular constant baseline.** Its denominator is the squared error from predicting the evaluation set's mean. A negative value means the model is worse than that baseline; it is mathematically valid. [1]

Take test targets (0,1,2). Their mean is 1 and total squared deviation is 2. Predictions (3,3,3) have squared error 9+4+1=14, giving $R^2=1-14/2=-6$. A large negative number is possible when predictions are very poor or the target varies very little.

For ordinary training-set least squares with an intercept, the fitted model can do at least as well as its intercept-only special case, so conventional training R-squared is nonnegative. That argument does not apply automatically to held-out data or to a model without an intercept.

**Adjusted R-squared penalizes extra predictors through degrees of freedom.** It can decrease when a new feature improves fit too little to justify the added parameter. However, it remains an in-sample formula under specific assumptions, not a substitute for validation or a universal complexity measure for arbitrary ML models. Report absolute error and baseline results, and handle a constant target explicitly because its denominator is zero.
@@ 928
**Many independent noise coordinates make distances look alike relative to their size.** The statement needs a sampling model: take independent uniformly distributed points in the unit cube and keep the number of compared points fixed as dimension grows. It is not true for arbitrary hand-chosen points. [1]

For independent U,V uniform on [0,1], $E[(U-V)^2]=1/6$. Across d independent coordinates, squared distance is a sum with mean d/6. Its standard deviation grows like the square root of d, so its relative fluctuation shrinks like $1/\sqrt d$. Dividing squared distance by d therefore concentrates it near 1/6.

With finitely many comparisons, all those distances concentrate near $\sqrt{d/6}$ together. The maximum divided by the minimum then approaches one in probability. Nearest and farthest points become harder to distinguish by relative Euclidean distance, which can make noisy high-dimensional neighborhoods uninformative.

**The assumptions matter.** An exponentially growing dataset, dependent coordinates, or data concentrated on a low-dimensional structure can behave differently. A few informative features can also be overwhelmed by many irrelevant ones. Feature selection, metric learning, and useful embeddings can recover meaningful neighborhoods. “Distance becomes meaningless” is too absolute; the proof explains one concentration mechanism and why representation choice matters.
@@ 929
**For unit vectors, distance contains exactly the same ranking information as angle.** Expand the squared norm: $\|u-v\|^2=u^Tu+v^Tv-2u^Tv$. Unit lengths make the first two terms equal one, and their dot product equals cosine similarity. Hence squared distance is $2-2\cos(u,v)$. [1]

A cosine similarity of 1 gives distance zero; similarity zero gives squared distance two; similarity -1 gives squared distance four. Because the relationship decreases strictly with cosine similarity, nearest Euclidean neighbors and largest-cosine neighbors have the same ordering when normalization assumptions hold.

**Without normalization, magnitude changes the answer.** Let the query be u=(1,0), and candidates a=(100,1) and b=(1,0). Inner product strongly favors a: 100 versus 1. Cosine favors b because it aligns perfectly, and Euclidean distance also favors b because it equals the query. Maximum inner product can reward vector length intentionally.

Normalizing therefore changes the model's scoring rule if norms encode useful information such as popularity or confidence. Match the retrieval index's metric to training and evaluation. Zero vectors need explicit handling because dividing by their norm is undefined. The equivalence is an exact algebraic identity under unit norms, not a general interchangeability of all vector-search metrics.
@@ 930
**PCA preserves variation, even when that variation has nothing to do with the label.** Construct two classes spread widely left-to-right. For both, let x1 have variance 100. Put one class near x2=+1 and the other near x2=-1, with very small vertical noise. The classes are easy to separate vertically but look identical horizontally. [1]

The largest total variance is along x1, so one-component PCA retains the horizontal direction and discards x2. In that one-dimensional representation, the class distributions overlap completely. PCA did exactly what it was designed to do: preserve the direction of greatest variance and reduce reconstruction error. It was not optimizing classification.

**LDA uses the labels to choose a different objective.** It seeks large separation of class means relative to within-class spread. Horizontal means are similar and within-class spread is large; vertical means differ and within-class spread is small. The vertical direction therefore has a strong discriminant ratio.

LDA has limits too: estimates can be unstable with little data or singular within-class covariance, and one linear projection may miss nonlinear class structure. Fit PCA or LDA inside the training fold; fitting supervised LDA using test labels leaks the answer. The example shows why dimensionality reduction must match the downstream objective.
@@ 931
**Support vectors are the observations whose constraints actively shape the optimum.** In the SVM dual representation, $w=\sum_i\alpha_i y_i x_i$. Points with zero multiplier contribute nothing to that expression. Points on or inside a soft margin may have nonzero multipliers, including misclassified points. [1]

Under an unchanged objective and exact optimal solution, removing constraints with zero multipliers can preserve the same optimum. The original dual solution remains available after deleting zero entries, and the same primal solution satisfies the remaining constraints. This explains why many distant, inactive points need not affect the fitted hyperplane.

**“Remove 90% and get exactly zero change” requires care.** If the implementation averages loss over sample count, removing points can change effective regularization unless the penalty is adjusted. Class weights may also depend on counts. Approximate solver tolerances and nonunique intercepts can alter the reported solution even when the ideal argument applies.

Do not label points inactive merely because they look distant in a plot; use the fitted margins and dual coefficients under the actual objective. A removed point might also matter after a later change in hyperparameters or features. Support-vector sparsity describes one fitted optimization problem, not a permanent guarantee that the rest of the dataset is unnecessary.
@@ 932
**A product treats one zero as decisive.** In multinomial Naive Bayes, a document's class score multiplies probabilities for its words. If a word never appeared in the spam class, a naive frequency estimate assigns it probability zero, making the entire document likelihood zero for spam regardless of all other words. [1]

Additive smoothing gives every vocabulary term some probability mass. If a class contains 100 word occurrences, vocabulary size is 20, and alpha=1, an unseen word gets probability $1/(100+20)=1/120$. A word seen nine times gets $(9+1)/120$. The denominator increases too, so the probabilities still sum to one.

**Log space fixes a different problem.** Multiplying hundreds of small positive probabilities can underflow to numerical zero. Adding their logarithms preserves useful relative scores. But log(0) is negative infinity, so switching to logs alone does not solve the unseen-word problem. You need both smoothing and stable arithmetic.

Build the vocabulary from training data and specify treatment of truly out-of-vocabulary terms. Conditional independence is a simplifying assumption: related words can count overlapping evidence repeatedly, making probabilities overconfident. Naive Bayes can still rank or classify effectively, but probability calibration should be assessed separately.
@@ 933
**Squared distance gives extreme points disproportionate influence.** For one-dimensional data (0,0,0,100) and one cluster, the k-means centroid is 25. The three ordinary points are now far from their center because the single outlier pulled the mean. A median-based center is 0, which better represents the majority under absolute distance. [1]

K-medians minimizes Manhattan distance and updates coordinates with medians. K-medoids restricts each center to an actual observed point and can use suitable dissimilarities. Both can reduce some outlier sensitivity, but neither makes arbitrary contamination harmless, and medoid search can be more expensive.

**Initialization is a separate issue.** With several clusters, alternating assignment and centroid updates decreases the k-means objective but can settle at a local optimum. Different starting centers may divide the same data differently. K-means++ initialization and multiple restarts help explore better solutions.

Scale features deliberately, inspect cluster stability, and decide what an unusual point means. A rare but legitimate population may deserve a cluster; a sensor error may not. Removing every extreme point merely because it worsens the objective can erase the phenomena you intended to discover. Choose distance and robustness according to the data-generating task.
@@ 934
**Different anomaly detectors define “unusual” differently.** Isolation Forest asks how quickly random splits separate a point. An isolated extreme value can often be cut off with a short path, while a point in a dense group requires more splits. One-class SVM estimates a boundary around supported data. LOF compares a point's local density with its neighbors. An autoencoder treats large reconstruction error as suspicious. [1]

Consider a sparse small cluster beside a dense large cluster. A global density rule may call the entire small cluster anomalous, while a local comparison can recognize that its members resemble one another. Conversely, a powerful autoencoder might reconstruct some anomalies well, so low error does not prove normality.

**Complexity depends on the implementation.** Isolation Forest commonly builds T trees on subsamples of size psi. Its training cost is then roughly $O(T\psi\log\psi)$ under typical balanced behavior, while scoring N points costs about $O(NT\log\psi)$. Calling it universally O(N log N) hides those choices.

Select thresholds using operational capacity and known validation cases. Check high-dimensional noise, drift, and contamination. An anomaly score is a ranking of unusualness under a model, not automatically a calibrated fraud probability or evidence of wrongdoing. Compare detectors on the kinds of novelty the application actually needs to catch.
@@ 935
**Perfect ordering can coexist with very wrong probabilities.** Give two positive examples scores 0.02 and 0.02 and two negatives scores 0.01 and 0.01. Every positive outranks every negative, so the probability that a random positive beats a random negative is one: ROC-AUC=1. [1]

Yet the model claims that cases scored 0.02 succeed only about 2% of the time, while both examples in that group are positive. Its probability estimates are badly mismatched to this observed sample.

**Compute the loss.** Each positive contributes $-\log0.02\approx3.912$, and each negative contributes $-\log0.99\approx0.01005$. Averaging gives about 1.961. A constant prediction of 0.5 has log loss about 0.693, so the perfectly ranking model is much worse under log loss.

A monotonic calibration map can change probability scale while preserving ranking. However, fitting a map perfectly to these four points does not establish calibration on future data. Use separate calibration and test sets with representative prevalence. Choose metrics according to use: ranking candidates, estimating expected value, and making probability-threshold decisions impose different requirements. AUC alone cannot certify the latter two.
@@ 936
**Average precision rewards placing relevant documents early, not merely finding them somewhere in the list.** For relevance labels (1,0,1,0,1), the relevant positions are ranks one, three, and five. Precision at those points is 1/1, 2/3, and 3/5. [1]

If these are all three relevant documents in the corpus, AP is their average: $(1+2/3+3/5)/3=34/45\approx0.7556$. Precision at five is only 3/5=0.6 because it counts all five returned positions equally. Recall at five is 3/3=1 because all relevant documents were found.

**Why not average precision at every rank?** AP accumulates credit when a relevant document is retrieved; irrelevant positions affect later precision by increasing the denominator. This ties the score to the quality of the ranking as recall increases.

If two more relevant documents exist outside the list, recall at five becomes 3/5. A truncated AP using total relevant count would also divide the accumulated precision by five rather than three. Other AP@K conventions cap that denominator, so name the convention. MAP averages per-query AP, giving each query a defined weight. Also document how unjudged documents are treated; missing judgments are not automatically reliable negative labels.
@@ 937
**High agreement can occur because almost everyone chooses the majority class.** If two annotators independently label 95% of examples negative, their marginal distributions predict agreement of $0.95^2+0.05^2=0.905$. They can agree over 90% of the time without much item-specific shared understanding. [1]

Cohen's kappa compares observed agreement with that marginal baseline: $(p_o-p_e)/(1-p_e)$. If observed agreement is 0.92 and expected agreement is 0.905, kappa is approximately 0.158, far less impressive than the raw 92% suggests. The correction depends on a particular chance model, not on a universal definition of quality.

Krippendorff's alpha works through observed versus expected disagreement and supports multiple annotators, missing ratings, and distance functions for different label types. An ordinal rating can therefore distinguish a one-level disagreement from a much larger one when the chosen distance reflects that scale.

**Agreement is not correctness.** Annotators may share the same mistaken guideline. Class prevalence also affects these statistics, sometimes producing surprising values. Inspect confusion matrices, class-specific disagreement, examples requiring adjudication, and whether labels are inherently ambiguous. Improve the labeling instructions using actual disagreements, and retain a trusted evaluation process rather than using one coefficient as a complete quality certificate.
@@ 938
**Spend labels where they are likely to improve the model, not simply where it feels confused.** Uncertainty sampling selects small prediction margins or high entropy. Query-by-committee selects examples where different models disagree. Core-set or diversity methods seek coverage so the batch is not filled with nearly identical cases. [1]

If a binary model predicts 0.5 on a blurry, corrupted image, uncertainty is high, but even an expert may be unable to label it. Another uncertain image near a real decision boundary may be highly informative. The same score can represent reducible model ignorance or irreducible ambiguity.

**Use rounds instead of spending the whole budget blindly.** Start with representative labels, train a model, select a diverse batch of informative and labelable cases, then retrain. With 5000 available labels, repeated smaller rounds allow later choices to reflect what earlier labels taught. Include quality filters and account for annotation effort, not only example count.

Reserve a randomly sampled evaluation set that the acquisition strategy does not influence. Otherwise the selected training distribution can distort performance measurement. Compare learning curves against random sampling at equal cost, and audit rare groups. Pure uncertainty sampling can repeatedly chase outliers; combining uncertainty with representativeness and diversity better connects the budget to useful coverage.
@@ 939
**Weak supervision combines imperfect rules instead of pretending each rule is a gold label.** A labeling function might mark a review positive when it contains a phrase, use a dictionary, or consult another model. It can also abstain when it has no relevant evidence. Several functions may disagree on the same example. [1]

A label model estimates how trustworthy and dependent these signals are, then produces a probabilistic label. If three independent, reliable sources agree, confidence may increase. But ten near-identical rules derived from the same keyword are not ten independent votes; counting them that way would exaggerate certainty.

Inspect coverage, conflict, and correlations, and use clean labeled data to assess the combined output. Reliability estimation requires assumptions and can be hard to identify from unlabeled agreement alone. A shared blind spot can survive every majority vote.

**Label smoothing is a different operation.** It replaces a hard target with a mixture containing some probability mass on alternatives, discouraging extreme confidence. It does not learn which labeling function is reliable or repair a systematically wrong rule. Train with weak labels if useful, but retain an independently reviewed test set and evaluate slices where heuristics are likely to fail. More weak labels do not automatically compensate for a biased labeling mechanism.
''')
