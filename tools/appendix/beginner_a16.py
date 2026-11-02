from expand import write
write(r'''
@@ 951
**Hidden neurons need a reason to learn different features.** If two neurons start with identical incoming weights, biases, and outgoing connections, they compute the same activation and receive the same update. Gradient descent preserves their equality, so adding those neurons provides copies rather than distinct learned directions. [1]

With all-zero multilayer weights, the problem can be worse: a layer's gradient contains weights from later layers, so multiplying by zero can block learning in earlier layers. Random, appropriately scaled weights break this symmetry and let units respond differently.

**Biases are different.** Setting biases to zero is usually compatible with distinct random weight vectors; the neurons already receive different weighted inputs. Logistic regression also lacks a layer of interchangeable hidden units. At zero weights its sigmoid predicts 0.5, and its gradient is proportional to $X^T(0.5-y)$, which is generally nonzero.

Thus zero is not intrinsically forbidden for trainable parameters. The issue is whether initialization leaves useful learning directions indistinguishable or blocks gradients. Match the initialization scale to the activation and layer dimensions so symmetry breaking does not create exploding or vanishing signals.
@@ 952
**Replacing zero with the same small constant does not break symmetry.** If every hidden neuron has the same weight vector and bias, each sees the same weighted sum. If its outgoing connections are also identical, backpropagation gives each the same update. The neurons remain duplicates after every deterministic step. [1]

For example, four neurons with identical weights do not initially detect four different directions in input space. They all detect one direction four times. Training has no asymmetric signal telling one to specialize differently.

A common positive constant can also accumulate many same-sign inputs, producing activation scales that grow poorly with layer width. Well-scaled random initialization addresses both diversity and signal variance.

**Randomness is a practical method, not the underlying requirement.** A deliberately designed asymmetric deterministic initialization can also work. The essential conditions are distinguishable units and suitable numerical scale. Likewise, constant biases are often fine when weights already differ. Do not infer that every parameter tensor must be random; inspect the symmetry of the whole computation and how gradients reach each unit.
@@ 953
**Mini-batch gradients should reflect the training population instead of a repetitive ordering artifact.** Suppose data is stored with all negative examples followed by all positives. Early batches push the model one way, later batches push it another, and momentum carries those order-dependent effects across updates. [1]

Shuffling mixes examples so batches are usually more representative and changes which examples interact in each update. It also prevents every epoch from repeating the same sequence of gradient errors. Sampling without replacement within an epoch is not exactly IID sampling, but it is a useful practical approximation for many objectives.

**Keep structure that the task needs.** Shuffle independent examples, not tokens inside a sentence or frames inside a sequence. Preserve grouping when labels or hidden state depend on order. In distributed training, update the sampler's epoch seed so workers do not repeat one permutation indefinitely.

Shuffling cannot fix an unrepresentative dataset or a leaked validation split. Use a reproducible seed when debugging, then distinguish order sensitivity from genuine model quality through appropriate validation.
@@ 954
**Ask which ordering carries information before deciding what to shuffle.** In a stateful RNN, the hidden state from one chunk may initialize the next chunk of the same sequence. Randomly placing an unrelated chunk next would attach the wrong history to its inputs. [1]

An online-learning simulation also needs chronological processing: the model at time t must not have learned from observations arriving later. A deliberate curriculum has an intended order that arbitrary shuffling would undo.

**Independent time-series windows are different.** If each example contains its own complete historical window, labels are correctly aligned, and hidden state resets between examples, training windows can often be shuffled safely. The order inside each window and the chronological train/test boundary still matter.

Separate three decisions: order within an example, order across stateful batches, and order among independent examples. “Never shuffle time series” confuses them. Test the data pipeline with known sequence IDs and timestamps so hidden-state carryover or future availability does not become an invisible source of leakage.
@@ 955
**K controls how local the prediction is.** With K=1, one nearby example determines the answer. This can follow a complex boundary but is sensitive to noise or one mislabeled point. With a larger neighborhood, individual accidents average out, while small local patterns may be blurred. [1]

Imagine a narrow island of positive examples inside a mostly negative region. A small K can preserve that island. A very large K may include so many surrounding negatives that the island disappears from predictions.

At K=N, every query uses the entire training set. Regression returns the same global mean; ordinary classification returns the global majority decision. Variance from local sample fluctuations is low, but bias can be enormous if the target varies with x.

These are broad trends rather than strict monotonic guarantees on every finite dataset. Distance weighting, feature scaling, and density affect the result. Tune K with an appropriate split and inspect rare classes or sparse regions, not just aggregate accuracy. A stable predictor is useful only if it retains the structure the task needs.
@@ 956
**KNN stores examples and postpones most modeling work until a query arrives.** A query is compared with stored points, the closest K are selected, and their labels or values are aggregated. This is why it is called lazy learning. [1]

If there are N stored points with d features, a brute-force query computes roughly Nd coordinate comparisons. Selecting neighbors adds work, and storage is O(Nd). Basic fitting can be little more than validating and storing data, though building a search index can be substantial.

**Nonparametric does not mean parameter-free.** The representation grows with the dataset instead of being summarized by a fixed-size coefficient vector. K, the distance metric, feature scaling, weighting, and index settings are still important choices.

For a large embedding catalog, vectorized brute force may be efficient on accelerators; tree or approximate indexes offer different build, memory, and recall tradeoffs. Benchmark the complete query path, including preprocessing. Keeping every training point also creates update and data-management requirements that a small fixed model may avoid.
@@ 957
**Exact search can skip a region if it proves that region cannot contain a better neighbor.** A KD-tree partitions coordinate space into boxes; a ball tree groups points inside metric balls. The query's minimum possible distance to a region gives a pruning bound. [1]

Suppose the current nearest point is two units away and every point in a region must be at least five units away. That entire region can be skipped without sacrificing exactness. In favorable low-dimensional data, many regions can be ruled out cheaply.

**High-dimensional geometry often weakens those bounds.** Distances become less distinctive and regions overlap or remain close under their bounds, forcing the search to visit many branches. Traversal overhead can then make a tree slower than an optimized scan.

There is no universal failure threshold at dimension 20. Intrinsic dimension, sample distribution, metric, and query workload matter. Compare exact trees, vectorized brute force, and approximate methods at the required recall and latency. An index is useful because it avoids measured work, not because its name implies sublinear performance in every dataset.
@@ 958
**The distinction concerns how model capacity grows as data grows.** A parametric model has a fixed-size parameterization once its architecture is specified. Linear regression with d features has d slopes and an intercept whether trained on 100 or a million examples. [1]

Logistic regression and a Gaussian mixture with a fixed number of components are further parametric examples. A fixed neural-network architecture is also parametric despite having many weights.

KNN retains a growing set of examples. Kernel density estimation typically places a contribution at each observation. An unrestricted decision tree can add more regions as more data arrives. These are common nonparametric examples.

**Neither category is assumption-free.** KNN assumes the chosen distance makes nearby examples informative; density estimation depends on bandwidth; trees depend on partitioning rules. A fixed leaf cap also restricts a tree's capacity. Conversely, a very large parametric model can be extremely flexible. Choose based on sample efficiency, computational cost, and inductive bias rather than interpreting “nonparametric” as “no parameters” or “always more general.”
@@ 959
**A generative classifier explains how each class could produce the input.** It models a class prior and a class-conditional input distribution, then applies Bayes' rule. Naive Bayes uses simplifying conditional independence; a class-conditional Gaussian mixture models richer distributions. [1]

A discriminative classifier focuses on the label given the input or on a separating boundary. Logistic regression directly models conditional class probability, while an SVM optimizes a margin-based decision rule.

**Consider spam detection.** Naive Bayes estimates how often words occur in spam and ordinary mail, then compares the resulting likelihoods. Logistic regression learns weights that distinguish the classes without needing a full model of how emails are generated.

Strong generative assumptions can help when data are limited or when missing inputs and simulation matter. Incorrect assumptions about the input distribution can also introduce lasting bias. Discriminative models concentrate on the classification task but still need suitable features and regularization. An SVM margin is not automatically a probability. Compare learning curves and deployment needs instead of assuming one family always dominates.
@@ 960
**The useful lesson is a bias–sample-efficiency tradeoff, not a universal logarithmic sample rule.** Naive Bayes imposes a strong structure on how features relate within a class. That structure can make its estimates stabilize quickly with limited data, but it can also constrain the best boundary it will ever reach. [1]

Logistic regression directly fits the conditional decision problem. It may require more examples to estimate its boundary well while eventually achieving a lower limiting error when the generative assumptions are wrong.

**Think of two learning curves.** One descends quickly and levels off at a higher error; another descends more slowly and eventually passes it. This illustrates the result's intuition without claiming that every dataset follows those curves. If Naive Bayes assumptions are well matched or the discriminative model is misspecified, the comparison can differ.

The Ng–Jordan analysis concerns particular model pairs and assumptions, with feature dimension and sample count playing different roles. Reusing n for both obscures the claim. Read the stated theorem before quoting asymptotic rates, and validate the practical choice with learning curves under consistent features, regularization, and held-out evaluation.
@@ 961
**Exponentials can overflow even when the final normalized answer is ordinary.** For logits (1000,1001), direct exponentiation exceeds ordinary floating-point ranges. Softmax then attempts ratios involving infinities, potentially producing NaNs. Yet only the difference between the logits matters. [1]

Subtract the maximum, giving (-1,0). Their exponentials are about (0.3679,1), so the normalized probabilities are approximately (0.2689,0.7311). Multiplying numerator and denominator by the same factor preserves the result while keeping intermediate values safe.

For log-sum-exp, factor out the maximum explicitly: $\log\sum_i e^{x_i}=m+\log\sum_i e^{x_i-m}$. The shifted exponentials are at most one for finite inputs.

**Stable arithmetic still needs valid inputs.** If every masked logit is negative infinity, subtracting the maximum creates an undefined infinity difference. NaNs in the input also remain a problem. Define a policy for fully masked rows and use stable library log-softmax or cross-entropy implementations. Taking the log of an already underflowed probability cannot recover the lost information.
@@ 962
**Training needs feedback about how to improve, not just whether the current answer is wrong.** Zero-one loss is flat while a prediction stays on the same side of a decision boundary. A tiny parameter change usually leaves it unchanged, so its local gradient gives no useful direction. [1]

Cross-entropy distinguishes a correct answer at probability 0.51 from one at 0.99, and a wrong answer at 0.49 from one at 0.001. For a binary logit z, its derivative is $\sigma(z)-y$. A confidently wrong positive example gives a strong negative derivative, pushing the logit upward.

**The surrogate is easier to optimize, but it is not the final product metric.** Logistic loss is convex for linear logits; adding a neural network generally removes that convexity. Exact zero-one optimization difficulty also depends on the hypothesis class, so an unqualified complexity slogan is misleading.

Train with a suitable differentiable objective, then choose thresholds and evaluate costs, accuracy, or ranking on independent data. A well-optimized log loss does not automatically select the threshold needed for an asymmetric decision problem.
@@ 963
**Hinge loss asks for a sufficient margin, then stops asking for more.** With labels -1 or +1, the signed margin is y times the score. Hinge loss is one minus that margin when the margin is below one, and zero otherwise. [1]

A margin of -0.5 is wrong and incurs loss 1.5. A margin of 0.2 is correctly classified but still incurs 0.8. A margin of 2 incurs zero: the example is comfortably on the correct side.

Logistic loss continues decreasing at margin 2 and beyond, although its gradient becomes small. It therefore keeps rewarding increased probabilistic confidence instead of becoming exactly flat at a fixed margin.

**A score above one is not a probability above one.** The margin lives on an unrestricted score scale. Regularization still influences parameters even when some examples have zero hinge data loss. Compare margin robustness, probability calibration, and decision costs separately. Hinge's flat region explains support-vector sparsity, while log loss supplies a likelihood-based probabilistic interpretation.
@@ 964
**Hard voting discards confidence before combining models; soft voting combines confidence first.** Suppose three binary models predict positive probabilities 0.51, 0.51, and 0.01. Hard voting chooses positive because two models cross 0.5. Soft voting averages to about 0.343 and chooses negative. [1]

Neither result is automatically correct. The third model might have decisive evidence, or it might be badly overconfident. Soft voting benefits from comparable, useful probabilities; hard voting can be less vulnerable to extreme scales but throws away meaningful differences in confidence.

Weighted variants give different influence to each model. Estimate weights and calibration using held-out or out-of-fold predictions, not the final test labels. Align class ordering before averaging so one model's class-zero column is not mistaken for another's class-one column.

Diversity matters: models making the same errors add little independent information. For cost-sensitive decisions or abstention, validate a threshold on the aggregate score rather than assuming 0.5 is universally appropriate.
@@ 965
**Averaging cancels independent fluctuations but preserves shared mistakes.** Write the variance of the mean as one over M squared times the sum of all variances and covariances. There are M diagonal terms and M(M-1) off-diagonal terms. [1]

If each model's variation has variance sigma squared and every pair has correlation rho, the result is $\rho\sigma^2+(1-\rho)\sigma^2/M$. For rho=0 and M=10, variance becomes 0.1 sigma squared. For rho=0.5 and M=10, it is 0.55 sigma squared: much less improvement.

**The limit exposes the bottleneck.** As M grows, the independent component vanishes but the correlated component remains. Training another near-copy of the same model therefore gives diminishing returns.

This derivation concerns prediction or error variation under a specified ensemble-generating process. It does not directly promise the same reduction in classification error, and shared bias is not removed. Compare the quality gain with inference cost, and seek useful diversity through data, features, or modeling choices rather than adding weak models solely to lower correlation.
@@ 966
**Bootstrap samples alone can still produce very similar trees.** If one feature is overwhelmingly attractive, almost every bagged tree may choose it at the root despite seeing different sampled rows. Their structures and errors remain correlated. [1]

A random forest also restricts which features are considered at each split. Sometimes the dominant feature is unavailable, forcing a tree to learn another useful partition. Across many trees, this can diversify errors and make averaging more effective.

**There is a strength-versus-diversity tradeoff.** Too few candidate features can produce weak individual trees. Too many can make them nearly identical. A square-root feature count is a common classification setting, not a defining rule that applies to every task or library.

Tune feature subsampling together with leaf size, depth, and tree count. Use validation or appropriate OOB estimates and examine important slices. Low correlation is valuable only when the trees still carry predictive signal; averaging many uninformative independent guesses is not the objective. The forest works by combining competent but imperfectly correlated predictors.
@@ 967
**Each bootstrap tree gets its own temporary holdout.** A bootstrap sample draws N times with replacement from N rows. On any draw, one particular row is missed with probability 1-1/N. Missing it on every draw has probability $(1-1/N)^N$. [1]

Take logarithms: $\log[(1-1/N)^N]=N\log(1-1/N)$. Since log(1-z) is approximately -z for small z, this approaches -1, so the probability approaches $e^{-1}\approx0.368$.

For a given training row, aggregate predictions only from trees that did not sample it. Those are its out-of-bag predictions. Different rows use different tree subsets; there is not one permanent 36.8% holdout shared by the forest.

**Respect dependence and model selection.** OOB can be useful for ordinary independently sampled data, but it does not simulate forecasting into the future or predicting unseen groups when related examples remain in training. Repeatedly tuning to OOB results can also overfit that estimate. Keep a deployment-appropriate final evaluation split for the whole modeling process.
@@ 968
**AdaBoost deliberately shifts attention toward examples the current ensemble gets wrong.** A weak learner with weighted error epsilon below one half receives weight $\alpha=\tfrac12\log((1-\epsilon)/\epsilon)$. Better learners receive more influence in the final vote. [1]

If epsilon=0.2, alpha is about 0.693. A misclassified example's weight is multiplied by $e^{0.693}\approx2$, while a correctly classified example's weight is multiplied by $e^{-0.693}\approx0.5$, before normalization. The mistake gains four times as much relative weight as the correct example.

**That focus can be harmful when the label is wrong.** A persistently mislabeled point keeps attracting effort even if no meaningful rule can satisfy it. The associated exponential margin loss grows rapidly on such cases, making noise influential.

Shrinkage, early stopping, robust variants, and label review can reduce the problem. Random forests do not repeatedly amplify the same errors through this boosting rule, though they are not immune to noisy labels. Handle perfect or unusable weak learners explicitly rather than blindly evaluating infinite or invalid coefficient formulas.
@@ 969
**The meta-learner must see the kind of base-model predictions it will receive at deployment.** A highly flexible base model may predict its own training labels nearly perfectly while generalizing poorly. If those in-sample predictions train the stacker, the stacker learns to trust an unrealistic signal. [1]

Split training data into folds. For each fold, fit each base model on the other folds and predict the held-out rows. Collect those out-of-fold predictions into a new feature table and fit the meta-model there. Every meta-training row now contains predictions from models that did not train on that row.

Then refit base models on the permitted full training set for serving, while evaluating the entire stack on an independent holdout. Preprocessing and hyperparameter choices must respect the same fold boundaries.

**The failure is leakage, not a guaranteed 100% weight outcome.** A stacker might distribute weights in many ways, but in-sample inputs bias its evidence. Use time-aware or group-aware folds when deployment requires them. Otherwise closely related records can leak even though the row itself was technically held out.
@@ 970
**Early stopping limits how far optimization follows the training data.** A model often learns broad signal before fitting idiosyncratic noise. Select a checkpoint using validation performance and stop when further updates cease to improve that criterion meaningfully. [1]

In quadratic linear regression, gradient descent learns high-curvature directions faster than low-curvature directions. Stopping after finitely many steps leaves some directions more strongly shrunk. Ridge also shrinks directions according to the design matrix's spectrum, which explains the regularization analogy.

**The analogy is not exact equality.** With eigenvalue lambda and step size eta, a finite-step filter involves $1-(1-\eta\lambda)^t$, while ridge uses a factor like $\lambda/(\lambda+\alpha)$. One scalar ridge penalty generally cannot match the early-stopping filter at every eigenvalue.

For neural networks the relationship is less direct. Save the best validation checkpoint rather than simply the last step after patience expires. Keep the final test set untouched, and account for noisy validation metrics. Early stopping is a useful control on optimization and generalization, not a theorem that every stopped model equals an L2-penalized fit.
@@ 971
**Mixup teaches intermediate inputs to have intermediate targets.** If lambda=0.7, combine 70% of one input with 30% of another and use the same mixture of their target vectors. The model is encouraged to vary smoothly between examples instead of making arbitrarily sharp changes. [1]

For a cat and dog image, the target may become 0.7 cat and 0.3 dog. This is a training construction, not a claim that the mixed image contains a literal fractional animal. CutMix instead pastes a spatial region from one image into another and commonly weights labels by the replaced area.

**The transformation must fit the task.** A small pasted patch may contain the only object, so area is an imperfect proxy for semantic contribution. Detection and segmentation need boxes or masks transformed consistently. Mixing arbitrary tabular identifiers or text tokens may create meaningless examples.

Validate strength, task accuracy, and calibration. Augmentation introduces assumptions about which intermediate or altered examples should have which labels. It helps when those assumptions regularize useful behavior and can hurt when they contradict the real task.
@@ 972
**Test-time augmentation averages a model's views of the same example.** Predict from the original image and several valid transformations, then combine the predictions. This can reduce sensitivity to one crop or orientation, behaving like a small ensemble without training separate networks. [1]

If five views are processed, inference work may approach five forward passes, though batching and shared overhead affect actual latency. That can be acceptable for offline analysis and too expensive for a strict interactive budget.

**A valid transformation must preserve the label.** Horizontal flipping can change text, directional signs, or laterality. For segmentation, map each predicted mask back into original coordinates before averaging; otherwise corresponding pixels do not align.

Benchmark the full latency distribution and quality on deployment data. A competition gain may depend on test images unlike production traffic. TTA is used in some deployed systems when its benefit justifies the cost, so “never used in production” is inaccurate. Choose the number and type of views from a measured quality–latency tradeoff.
@@ 973
**Multiclass asks for one answer; multilabel allows several answers at once.** An animal image classified as exactly one of cat, dog, or bird uses a mutually exclusive class model. A photo tagged “outdoor,” “night,” and “vehicle” can legitimately have several positive labels. [1]

Softmax makes multiclass probabilities sum to one, so increasing one class's share reduces the others. Independent sigmoid outputs each lie between zero and one without a shared sum constraint. Binary cross-entropy then scores each label separately.

**Factorized outputs do not mean the real labels are unrelated.** A shared network can learn that night and headlights often co-occur, even with separate sigmoid heads. However, hard hierarchical or exclusion constraints may need additional structured modeling.

Choose thresholds per label based on prevalence and costs rather than assuming all should use 0.5. Treat missing annotations carefully: an unrecorded tag may be unknown, not negative. Report macro and micro metrics and rare-label performance. Using softmax for a genuinely multilabel task forces valid labels to compete and encodes the wrong output contract.
@@ 974
**Binary classifiers can be combined in two common ways.** One-vs-rest trains one classifier for each class against all others, giving K models. One-vs-one trains a classifier for every unordered pair, giving $K(K-1)/2$ models. [1]

With four classes, OvR needs four classifiers and OvO needs six. Each OvO classifier sees only examples from its two classes, then a voting or score-combination rule produces the final class. OvR compares the K class-versus-rest scores.

**More models does not always mean more training work.** Pairwise datasets are smaller, which can benefit expensive kernel methods. But prediction may require evaluating many classifiers. OvR faces potentially imbalanced one-versus-many problems, and its scores may have different scales.

Specify how scores or votes are combined and how probabilities are calibrated if needed. Neither construction automatically respects class hierarchy or gives a coherent multiclass probability distribution. A native multiclass model such as softmax is another option. Compare total training cost, inference cost, imbalance, and validation quality under the actual class distribution.
@@ 975
**Ordinal labels carry an order that ordinary class names do not.** Predicting five stars instead of four is usually a smaller mistake than predicting five instead of one. A flat multiclass loss does not automatically encode that distinction. [1]

For five categories, predict four cumulative probabilities: above one, above two, above three, and above four. Valid probabilities must decrease as the threshold rises. If they are (0.9,0.7,0.4,0.1), category probabilities are (0.1,0.2,0.3,0.3,0.1), obtained by successive differences with boundary values one and zero.

**Independent unconstrained heads can violate this structure.** Predicting a greater probability of exceeding four than exceeding three is inconsistent and can produce negative category probabilities after differencing. Ordered cutpoints with a shared score, or another explicit monotonicity mechanism, prevent that problem.

Evaluate an error measure that reflects ordering as well as probability calibration. The proportional or shared-score assumptions of a simple ordinal model may be restrictive, so compare flexible alternatives when evidence supports them. Respecting order does not require pretending the distance between every adjacent category is numerically identical.
''')
