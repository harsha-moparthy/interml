from expand import write
write(r'''
@@ 976
**Separate the decision to spend from the amount spent after that decision.** A single continuous regression can struggle when 80% of outcomes are exactly zero and the rest follow a long positive tail. A hurdle model uses a gate for positive outcome and a second model for positive magnitude. [1]

If purchase probability is 0.2 and mean positive spend is 50, expected spend is 10. The two components can depend on different features: one may predict willingness to buy, another basket size among buyers.

**Zero-inflated and hurdle models are not identical.** In a hurdle model, the positive component cannot generate zero. In a zero-inflated count model, a structural-zero component is mixed with a count distribution that can also produce zero. Choose according to how the data arise.

Evaluate gate calibration and positive-tail fit separately before evaluating aggregate value. For a log-normal positive component, exponentiating its mean log spend is not its arithmetic mean. Also check observation windows: a customer who has not purchased yet may be censored rather than a permanent zero.
@@ 977
**Follow-up length carries information.** A machine still working after two days and one still working after two years are both “not failed,” but they provide very different evidence about lifetime. Binary classification can lose that distinction if both are labeled as permanent negatives. [1]

Right censoring means the event time exceeds the last observed follow-up, not that the event will never happen. Survival analysis includes that partial information. Kaplan–Meier estimates the fraction surviving beyond each time while adjusting the at-risk set. Cox models relative hazard through covariates and a baseline hazard.

**Hazard is an instantaneous event rate among those still at risk, not a cumulative probability.** A Cox multiplier of two means twice the modeled hazard under its proportional-hazards assumptions, not necessarily twice the probability of failure by every horizon.

Check censoring assumptions, changing effects over time, and competing events. Evaluate survival probabilities at useful horizons and inspect calibration, not only ranking. Survival methods are valuable when event timing and censoring matter; they are not automatically superior if the application truly has complete labels for one fixed horizon.
@@ 978
**Power transformations change scale to make relationships or variability easier to model.** For positive x, Box–Cox uses $(x^\lambda-1)/\lambda$, with log(x) as the continuous limit at lambda=0. Lambda=1 is essentially a shifted identity; other values compress or expand different ranges. [1]

A logarithm turns multiplicative differences into additive ones. Values 1, 10, and 100 are extremely uneven on the raw scale but equally spaced in base-ten log units. This can help a model whose errors are more naturally relative than absolute.

Yeo–Johnson uses a related piecewise family that permits zero and negative values. A Bessel function is a different mathematical object and does not belong in the question's list of standard power transforms.

**Do not transform solely to make a histogram look Gaussian.** Many models do not require normally distributed features. Residual behavior and the modeled relationship may matter more. Fit any transformation parameter using training data only, preserve a valid inverse when needed, and assess held-out performance. Trees may gain little from a strictly monotonic feature transform, while linear models can change substantially.
@@ 979
**Clipping changes an extreme value; trimming removes its row.** If a chosen upper boundary is 100, winsorization replaces 1000 with 100. Trimming discards the observation entirely. The first keeps the sample count but alters the tail; the second changes which population is represented. [1]

Neither operation establishes that the original value was erroneous. A very large transaction or rare failure may be the event the model most needs to understand. First inspect units, collection bugs, and domain meaning.

**Learn boundaries without looking into validation or the future.** Estimating the 99th percentile from the whole dataset lets later observations influence training preprocessing. Fit it on the training subset and reuse that fixed rule on held-out data. In a rolling system, update it only with information available then.

Consider retaining a clipping indicator and the original value for audit. Evaluate tail cases separately, since aggregate accuracy may improve while rare-event detection worsens. A 1st/99th percentile rule is a tunable modeling choice, not a universal data-cleaning law. Distribution drift can also make a previously sensible boundary clip too much new data.
@@ 980
**A displacement is unusual relative to the data's usual scale and direction.** A change of ten units in a feature whose standard deviation is 100 is small; the same change in a feature whose standard deviation is one is large. Euclidean distance ignores that unless features are transformed. [1]

Mahalanobis distance accounts for the covariance matrix. With diagonal covariance, it divides squared coordinate differences by their variances. With correlated features, it also recognizes that movement along a common trend is less surprising than movement across a tight relationship.

Imagine height and weight are strongly correlated. A point moving along the tall-and-heavy direction can be ordinary even with a large raw distance from the mean. A mismatched height-weight combination can be unusual despite a smaller raw distance.

**The covariance estimate is part of the model.** Few samples, high dimension, or outliers can make its inverse unstable or misleading. Use shrinkage or a justified lower-dimensional representation and stable factorization-based solves. One ellipsoid also cannot describe every multimodal distribution. Mahalanobis distance improves the metric when its covariance assumptions match the data, not by universally replacing Euclidean distance.
@@ 981
**Cosine annealing gradually reduces the step size within a cycle.** It starts near a chosen maximum, moves smoothly toward a minimum, and slows refinement near the end. A warm restart raises the learning rate again while retaining learned weights. [1]

At the start of a cycle, the cosine term is one and the schedule returns the maximum. Halfway through, it returns the midpoint of maximum and minimum. At the end, it reaches the minimum. Longer later cycles can give progressively more time for refinement.

**Restarting the rate is not restarting the model.** The network does not return to random weights. Optimizer-state treatment depends on the chosen implementation and should be documented. The higher-rate phase can explore beyond the current local region, but it can also disturb a good solution.

Compare against a simpler decay schedule at equal optimizer updates. Express the cycle lengths in the units actually advanced by the scheduler, especially with gradient accumulation. Select checkpoints using independent validation rather than assuming the last cycle endpoint is always best.
@@ 982
**The one-cycle policy changes the learning rate to suit different phases of training.** It begins relatively low, rises to a high peak, then falls to a much smaller final rate. Momentum is often varied in the opposite direction. [1]

The early increase allows a larger exploratory phase; the late decrease permits precise refinement. In favorable experiments, the high-rate phase provides useful regularization and reaches good performance with fewer updates, motivating the term super-convergence.

**A high peak is a risk to tune, not a free acceleration.** If it exceeds the stable range, loss can diverge or useful weights can be damaged. Batch size, weight decay, augmentation, and total updates interact with that choice.

Use a bounded range-finding experiment and a validation comparison to choose the peak and duration. Advance the scheduler once per intended optimizer update, not once per microbatch if updates are accumulated. Report the complete schedule so “one cycle” is reproducible. A successful recipe on one dataset does not guarantee the same speedup on another architecture or training budget.
@@ 983
**Weight averaging smooths a sequence of nearby parameter estimates.** EMA updates a running model toward the current weights with a small fraction of each step. With beta=0.99, the new average is 99% old average and 1% current weights. Recent iterates influence it gradually instead of replacing it abruptly. [1]

Polyak averaging takes an ordinary average of iterates; SWA averages selected later checkpoints, commonly under a schedule that explores a useful region. These are weight-space operations, unlike averaging predictions from separately executed models.

**Compatibility matters.** Nearby iterates from one trajectory may average well, while unrelated checkpoints can have mismatched representations. BatchNorm running statistics may need recalculation for the averaged weights. Keep accumulation precision, start time, decay, and checkpoint state explicit.

EMA is common in diffusion training, but the claim that LLMs almost always serve EMA weights is not a reliable universal rule. Evaluate the averaged model against the ordinary checkpoint on the actual task. Smoother weights can improve generalization, but they may also lag behind a rapidly improving or adapting model.
@@ 984
**Checkpointing trades recomputation for fewer stored activations.** Ordinary backpropagation keeps intermediate values from the forward pass so gradients can use them later. Instead, save selected boundaries and recompute missing values when backward reaches each segment. [1]

For L equal layers split into segments of length s, save roughly L/s boundaries and at most s internal activations while recomputing one segment. The idealized activation count is proportional to L/s+s. Differentiating the continuous approximation gives $-L/s^2+1=0$, so s is about the square root of L.

With L=100, segments of ten suggest about ten boundaries plus ten active internal values rather than storing all hundred, ignoring constants and endpoint details. The price is executing some forward computations again.

**This is not a square-root bound on all GPU memory.** Parameters, gradients, optimizer states, and temporary attention buffers remain. Unequal layer sizes and branching graphs change the best schedule. Preserve randomness and mutable state during replay so recomputation matches the intended forward computation. Measure actual peak memory and step time; the formula explains one simple schedule under explicit assumptions.
@@ 985
**Module behavior and gradient recording are separate controls.** `model.eval()` changes how modules such as dropout and BatchNorm behave. It does not tell autograd to stop constructing a graph. `torch.no_grad()` disables gradient recording in its scope but leaves the module's training flag unchanged. [1]

For ordinary inference, you typically need both: deterministic evaluation behavior and no unnecessary gradient graph. If you use only no-grad while the model remains in training mode, dropout can still randomly zero activations and BatchNorm can still update running statistics.

**Sometimes the separation is useful.** Gradient-based input attribution needs evaluation-mode predictions while retaining gradients with respect to the input. Conversely, a statistics-collection pass may intentionally use training behavior without parameter gradients.

Choose the two settings according to the operation, and restore the previous mode after temporary validation. Confusing them can create excess memory use or inconsistent outputs without producing an obvious error. Inference mode offers another optimized context with additional restrictions; it should be chosen deliberately rather than treated as identical to every use of no-grad.
@@ 986
**A tensor's shape is not its physical storage order.** Strides specify how far to move in storage when advancing along each dimension. Transposing a matrix usually changes those strides without rearranging its underlying elements. [1]

For a contiguous 2-by-3 matrix, transposition yields a 3-by-2 view whose logical row order no longer matches a simple contiguous traversal. Flattening that transposed layout as a no-copy view may be impossible with the requested strides.

`view()` requires the requested shape to be compatible with existing storage and strides. `reshape()` uses a view when possible and otherwise may copy. Calling `contiguous()` explicitly requests an appropriate contiguous copy when one is needed.

**Do not assume every reshape aliases or every noncontiguous tensor forbids every view.** Some reshapes remain compatible, and others allocate. Check shape, stride, and aliasing when debugging in-place changes or performance. A costly layout copy can dominate a small operation, while a mistaken axis interpretation can silently compute the wrong result even when all dimensions are legal.
@@ 987
**Stopping gradients and copying data solve different problems.** `detach()` disconnects a tensor from its current autograd history but generally shares the underlying storage. `clone()` allocates copied data while preserving a differentiable connection to the original. [1]

If x requires gradients and y=x.clone(), a loss on y can still propagate to x. If z=x.detach(), ordinary gradient tracking through that connection stops, but changing shared storage can affect x as well. Detachment is not immutability.

**Use `detach().clone()` for an independent frozen snapshot.** This is useful when a target or saved diagnostic should neither backpropagate nor change when the original tensor is modified later. Moving it to CPU is a third, separate decision when logging or retaining it outside GPU memory.

In-place edits can trigger autograd version checks if saved values are needed for backward, and careless shared-storage mutation can be surprising. Decide explicitly whether you need graph separation, storage separation, or device transfer. The methods are related only because practical workflows sometimes need more than one of those operations together.
@@ 988
**The CPU can enqueue GPU work before that work has finished.** This asynchronous execution allows launch and preparation to overlap device computation. Requesting a Python number with `.item()` requires the device result to be available on the host, so the CPU may have to wait. [1]

Imagine each step queues kernels and immediately reads its loss scalar. That read creates a repeated synchronization point, reducing the opportunity to overlap work across the pipeline. Printing actual device values can create a similar need to retrieve them.

**The effect depends on frequency and workload.** One occasional scalar read does not universally destroy throughput. A repeated read in a tight small-kernel loop may be much more costly. Accumulate metrics on-device and transfer summaries at planned intervals when appropriate.

Use CUDA events or properly synchronized timing for benchmarks. Measuring only the time to enqueue an operation can make GPU work look falsely cheap, while a later `.item()` appears to bear the cost of all preceding work. A timeline helps attribute the delay to the real computation and synchronization boundary.
@@ 989
**Parameters are optimized weights; buffers are tracked model state with other update rules.** `nn.Parameter` registers a tensor for parameter iteration and ordinary optimizer use. `register_buffer` makes non-parameter tensors follow module device moves and, by default, serialization. [1]

BatchNorm running mean is a buffer because a running-statistics rule updates it rather than gradient descent. A fixed positional table can also be a buffer so it moves with the model instead of remaining accidentally on the CPU.

**Persistence is a choice.** A reproducible RoPE sine/cosine cache may be registered as nonpersistent or regenerated instead of being saved in the checkpoint. Thus it is too strong to say every such table must appear in `state_dict`. Buffers can still be mutated by application code.

A plain tensor attribute does not automatically receive the same device and serialization handling. Choose registration, dtype, persistence, and optimizer membership deliberately. This avoids both training constants unintentionally and forgetting state required to reproduce inference after loading a checkpoint.
@@ 990
**A distributed collective is a coordinated operation, not a local function call.** If rank zero enters a barrier while rank one skips it, rank zero waits for a participant that never arrives. Similar mismatches can occur in DDP operations that trigger communication internally. [1]

A branch such as “only the worker with this condition runs the collective” is unsafe unless the protocol explicitly accounts for every rank. Even if all workers eventually communicate, calling different collectives or calling them in different orders can mismatch the sequence.

**An earlier exception can look like a network failure elsewhere.** One worker may crash while reading data; the others later time out in synchronization. Inspect the earliest rank-local error and step identifiers before blaming the collective implementation.

Keep distributed decisions consistent, broadcast decisions when needed, and match iteration counts or use supported uneven-input mechanisms. Record tensor shapes and collective order for diagnosis. Increasing the timeout does not fix missing participation; it only lengthens the wait. Recovery should return all workers to a coherent state rather than letting some continue with stale progress.
@@ 991
**Teacher forcing trains each next-token prediction with the correct preceding tokens.** For a target sentence, the model predicts token two after the true token one, token three after the true first two, and so on. With causal masking, those losses can be computed efficiently over the known sequence. [1]

During generation, the model instead sees its own outputs. If it chooses a wrong early token, later predictions condition on a history that may differ from common training prefixes. Errors can then compound. This training–generation discrepancy is called exposure bias.

**A translation example makes the distinction clear.** Training always supplies the correct opening phrase before predicting the verb. At inference, an incorrect opening may imply a different grammatical structure, and the model must recover from that self-created context.

Evaluate generated sequences and deliberately perturbed prefixes, not just teacher-forced token loss. Sequence-level objectives or suitable augmentation can improve robustness, but naive scheduled sampling changes the learning objective and has caveats. EOS decisions, search strategy, and model quality also affect generation; not every sequence error is explained by exposure bias alone.
@@ 992
**Extending a fixed prefix multiplies its probability by another number at most one.** Log probabilities therefore accumulate nonpositive terms. A prefix with probability 0.4 followed by a token of conditional probability 0.5 has joint probability 0.2. Raw beam scores can consequently favor early termination when the model assigns EOS enough probability. [1]

Length normalization divides log probability by a function of output length, or uses another explicit penalty. This changes the search objective so longer candidates are not penalized in exactly the same way as raw summed log likelihood.

**It is a decoding heuristic, not a probability identity.** A normalized score need not be the log probability of a valid distribution over complete sequences. Too much compensation can favor rambling outputs; too little can produce truncated ones.

Specify whether EOS counts toward length and exclude prompt tokens consistently. Stopping rules must respect the normalized scoring objective, otherwise an unfinished candidate that could later win may be discarded. Tune on task quality and completeness rather than assuming that longer text is better. The model's learned length distribution also matters, not just multiplication by probabilities.
@@ 993
**Greedy decoding repeatedly follows the locally most likely continuation.** If the generated phrase makes repeating that phrase even more likely, the model can enter a self-reinforcing loop. Low temperature sharpens the same preference and may reduce opportunities to escape it. [1]

A repetition penalty lowers scores for previously generated tokens. This can discourage loops, but repeated tokens are sometimes necessary in code, names, lists, or exact quotations. Penalizing them indiscriminately can damage the answer.

**Min-p sampling uses a relative probability cutoff.** If the most probable token has probability 0.5 and min-p is 0.1, keep tokens with probability at least 0.05, then renormalize. The threshold adapts to how peaked the distribution is. If only the loop token survives, min-p does not solve the loop.

Neither technique guarantees factuality or nonrepetition. Compare decoding settings on realistic prompts, and inspect training-data or model issues if repetition is systematic. A sampling control changes how the model's distribution is used; it does not necessarily repair the distribution itself.
@@ 994
**Perplexity summarizes how much probability the model assigns to the actual text.** Average the negative log probability of each observed token, then exponentiate. Lower values mean the model gave the test sequence higher probability per prediction unit. [1]

PPL=10 means average log loss equals that of choosing uniformly among ten alternatives. It does not mean there were exactly ten equally likely tokens at every position. Some tokens may be nearly certain and others very surprising.

**The unit matters.** One tokenizer might encode a word as one token and another as three. Their per-token losses are measured over different events, so raw token perplexities cannot be interpreted as directly comparable model quality. Normalize total log likelihood by a common byte or character count when making a suitable cross-tokenizer comparison.

Document text normalization, encoding, context windows, and BOS/EOS treatment. Common-unit likelihood comparisons still require compatible scoring conventions. Finally, a model can predict ordinary text well yet follow instructions poorly. Perplexity measures a distributional prediction task on a specific corpus, not every useful language-model capability.
@@ 995
**Overlap metrics and semantic metrics notice different kinds of agreement.** BLEU uses clipped n-gram precision with a brevity penalty, rewarding generated phrases that appear in references. ROUGE variants emphasize overlap through recall, subsequences, or related formulas. [1] [3]

A valid paraphrase can receive low lexical overlap. BERTScore compares contextual token representations, and MoverScore uses a transport-style alignment, making them more tolerant of some paraphrases.

**Semantic closeness is not factual equivalence.** “The shipment contains 15 units” and “the shipment contains 50 units” are similar sentences with a crucial numerical disagreement. Negation and unsupported details can be missed by both broad semantic matching and aggregate overlap scores.

Use metrics as complementary diagnostics, with exact variants and tokenization specified. Corpus-level BLEU is not interchangeable with a single sentence's score. For summarization, check claims against the source and assess missing important information as well as invented content. Multiple references can recognize legitimate wording variation, but no one automatic score establishes faithfulness or usefulness by itself.
@@ 996
**Learning a new task changes parameters that an old task may still need.** A model trained first on one image category set and then only on another can shift its features and decision boundaries until earlier categories perform poorly. This is catastrophic interference or forgetting. [1]

The problem is not simply that old examples are absent from storage. Even with an old checkpoint available, the currently deployed model may no longer express the old behavior. Shared features can enable transfer, but conflicting gradients can also overwrite useful structure.

**Different remedies preserve different things.** Replay keeps old examples in training. Distillation preserves selected old outputs. Importance penalties restrict changes to estimated critical parameters. Adapters or task-specific capacity reduce forced sharing but require routing or storage choices.

Evaluate both old and new tasks throughout the sequence. In class-incremental settings, the model may not know which task a test example belongs to, making evaluation harder than selecting a task-specific head with an oracle. Track calibration and rare classes as well as average accuracy. Retention and adaptation are a tradeoff to measure, not something guaranteed by one regularizer.
@@ 997
**Keeping raw records on a device does not make every transmitted update private.** FedAvg trains local models and aggregates their updates at a server. Those updates can still encode information about local records, especially under a strong observer or repeated access. [1] [2]

Differential privacy instead limits how much the distribution of released outputs can change when one protected contribution is added or removed. Epsilon controls the multiplicative bound and delta allows a specified slack. The neighboring-dataset definition must say whether the protected unit is one record or an entire user.

**Clipping bounds influence; noise obscures it.** DP-SGD limits contribution norms, adds calibrated noise, and accounts for cumulative privacy loss over steps. Clipping without noise is not the same guarantee, and noise without correct sensitivity accounting is insufficient.

DP does not promise that no familiar string can ever appear or that no population-level fact can be inferred. Its guarantee depends on the budget, access model, and correct implementation, with a utility tradeoff. Federated learning, secure aggregation, and differential privacy address different aspects of the system and should not be treated as interchangeable labels.
@@ 998
**Membership inference asks whether one particular record helped train the model.** It does not necessarily recover the record itself or the model's weights. An attacker may exploit lower loss or higher confidence on training examples than on otherwise similar unseen examples. [1]

Suppose a model gives unusually confident answers on a rare sentence it memorized. An attacker with that sentence and query access may compare its behavior with a reference distribution and infer membership. Attack strength depends on outputs available, auxiliary data, and the training process.

**A fair evaluation must match member and nonmember distributions.** If members come from one domain and nonmembers from another, the attack may merely recognize the domain. Use realistic candidate records and report false-positive rates, especially for rare or sensitive subgroups.

Regularization and output restrictions can reduce some attacks but do not certify privacy. Proper differential privacy provides a formal influence bound under stated assumptions. Distinguish the privacy question from ordinary generalization: good average test accuracy does not rule out memorization of a small sensitive subset.
@@ 999
**An adversarial perturbation is chosen to increase model error, not sampled as ordinary noise.** FGSM linearizes the loss around the current input and takes a step in the sign of its gradient. Under an infinity-norm constraint, this sign choice maximizes that first-order loss increase. [1]

For gradient components (2,-3), an illustrative allowed perturbation is (epsilon,-epsilon). Every coordinate uses its permitted magnitude in the direction that increases the loss locally. Clip the resulting input to the valid data range as well.

PGD repeats smaller gradient steps and projects back into the allowed perturbation set, exploring a stronger local attack. Define targeted versus untargeted objectives, pixel scale, norm, budget, and model access before comparing results.

**Robustness must survive a competent evaluation.** Broken or masked gradients can make a weak attack fail without making the model robust. Test suitable adaptive attacks and ordinary corruptions. Mathematical closeness in one norm does not automatically represent a physically plausible sensor change. Adversarial training can improve resistance to a specified threat model, with compute and clean-quality tradeoffs.
@@ 1000
**Calibration changes confidence while trying to preserve useful discrimination.** Platt scaling fits a sigmoid with learned slope and intercept to binary scores. Temperature scaling fits one positive scalar T to multiclass logits using a held-out calibration set. [1]

With logits (2,1,0), T=1 gives approximate probabilities (0.665,0.245,0.090). T=2 yields roughly (0.506,0.307,0.186): the distribution is softer, but class one still has the largest probability. Dividing every logit by the same positive number preserves their order, and softmax preserves that order too.

**Therefore top-one predictions remain unchanged**, apart from numerical or tie-handling details. A smaller temperature sharpens confidence; a larger one increases uncertainty. This can improve probability quality without correcting any mistaken class ranking.

One scalar cannot repair arbitrary class-specific errors or distribution shift. Platt scaling is originally binary, so a multiclass use requires an explicit extension. Fit calibration separately from model training and final evaluation, then monitor it under the deployment distribution. A calibration transform is a probability adjustment, not a substitute for a better classifier.
''')
