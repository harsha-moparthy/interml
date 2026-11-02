from expand import write
write(r'''
@@ 1026
**A chunk should contain enough evidence to answer a question without burying that evidence in unrelated material.** Fixed token windows are predictable but can split a definition from its qualifier. Overlap reduces boundary loss while duplicating text and increasing index size. [1] [2]

Recursive splitting respects paragraphs and headings until a size limit. Structure-aware splitting keeps tables or sections intact with parent metadata. Semantic splitting groups related sentences, but similarity alone may still separate a rule from a necessary exception.

**Consider a policy paragraph followed by “except for temporary staff.”** Retrieving only the first paragraph can produce a confident wrong answer. Parent expansion or contextual metadata may be more useful than simply increasing every chunk size.

Contextual retrieval adds a short document-specific description before indexing. Preserve the original passage and provenance so generated context does not become unsupported evidence. Compare answer-bearing recall, citation correctness, final grounded accuracy, latency, and storage. Tune chunking with reranking and context budget; a fact lookup and a cross-section synthesis need different evidence granularity.
@@ 1027
**Top-k similar passages are good at local lookup but can miss a corpus-wide pattern.** Asking “What date did project X launch?” needs a few precise passages. Asking “What themes recur across all projects?” requires coverage across many documents, including ones that are not nearest neighbors of one query embedding. [1]

GraphRAG extracts entities and relationships, groups the graph into communities, and summarizes those communities. The summaries provide a route to broader synthesis instead of repeatedly retrieving from the same small neighborhood.

**Every derived layer can introduce errors.** A mistaken entity merge or invented relation can affect a community summary and then the final answer. Link derived claims back to original passages and evaluate diversity of evidence as well as factuality.

Use ordinary hybrid retrieval for exact local questions and graph/community retrieval where global structure helps. Include extraction cost, update latency, and access control on derived summaries. A summary built from restricted documents can itself reveal restricted information. The graph is a retrieval aid, not proof that its relationships or conclusions are true.
@@ 1028
**A follow-up often omits slots supplied by the conversation.** After “What was Apple's Q3 revenue?”, “What about Microsoft?” changes the entity while retaining the metric and intended period. Searching only the literal second sentence loses those constraints. [1]

Create a standalone query using structured conversation state: entity, metric, year, period, currency, and comparison intent. Preserve the original utterance and retrieve with both forms when useful.

**Inherited context can itself be ambiguous.** Companies may have different fiscal quarters. A comparison of each company's fiscal Q3 is not necessarily a comparison over the same calendar months. Resolve the intended period or state the interpretation clearly, and cite the underlying reports.

Do not silently invent a year or change revenue to profit. Test pronouns, entity swaps, topic changes, and corrections separately from retrieval quality. A rewriting model can produce a fluent but wrong query; validate important slots before treating the rewrite as the user's request. Conversational retrieval succeeds when omitted context is restored accurately, not merely when the query becomes longer.
@@ 1029
**Check claims against evidence one at a time.** An intrinsic hallucination contradicts the supplied source. An extrinsic hallucination adds something the source does not support, even if it might be true elsewhere. A single response can contain both supported and unsupported claims. [1] [2]

If a source says revenue was 10 million, “revenue was 20 million” contradicts it for the same period. “Profit was 2 million” is unsupported if the source says nothing about profit. Treating both as generic semantic mismatch loses a useful distinction.

Split the answer into atomic claims, retrieve relevant evidence windows, and classify support, contradiction, or insufficient evidence. Dates, negation, arithmetic, and entity alignment need explicit checks because broad text similarity can miss them.

**Latency needs a measured budget.** A small NLI model, batching, and cached evidence can help, but a universal sub-20 ms promise is unjustified without hardware and workload assumptions. Measure end-to-end p99 and error rates. When evidence is missing, abstain or retrieve more rather than interpreting absence of contradiction as support.
@@ 1030
**Embedding migration is a consistency problem as well as a computation problem.** Query vectors from a new encoder may not be compatible with old item vectors. Switching only one side can silently destroy retrieval even if dimensions match. [1] [2]

Build a versioned replacement index beside the old one. Backfill in checkpointed jobs, dual-write new updates, and reconcile deletions during the migration. Keep stable document IDs so results and evaluations remain comparable.

**Switch the encoder and index together.** Evaluate judged relevance and slice recall, canary traffic, and retain a rollback path. Account for temporary double storage and embedding throughput across 500 million records.

A learned alignment can sometimes bridge spaces related approximately by rotation, but different semantic geometry cannot be repaired by assuming such a map. Low average alignment error also does not guarantee the same nearest neighbors. Validate retrieval directly. Treat the bridge as a measured migration option, not a substitute for versioning, freshness, and end-to-end consistency.
@@ 1031
**Orthogonal alignment rotates or reflects one coordinate system without changing its internal distances.** With paired row embeddings B from one encoder and A from another, seek an orthogonal matrix Omega making B times Omega close to A. This is appropriate only if their spaces differ approximately by that kind of transformation. [1]

Expand the squared Frobenius objective. Orthogonality makes the squared-norm terms constant, so minimizing error is equivalent to maximizing $\operatorname{tr}(\Omega^TB^TA)$. Take the SVD $B^TA=U\Sigma V^T$ and choose $\Omega=UV^T$.

**Why this is optimal.** Let $C=U^T\Omega V$, also orthogonal. The trace becomes $\sum_i\sigma_i C_{ii}$, at most the sum of singular values because each diagonal entry is at most one. Choosing C=I reaches that bound.

Center the paired data if translation must be removed, and restore the corresponding means at inference. Requiring a rotation without reflection needs a determinant correction. Test held-out retrieval: preserving distances within B does not prove that arbitrary changes in the encoders' semantics can be aligned.
@@ 1032
**Positive-pair agreement alone admits a useless solution: make every embedding identical.** If the objective only pulls two views of the same example together, a constant vector gives zero distance for every pair. Low training loss then says nothing about whether different examples remain distinguishable. [1] [2]

Contrastive methods introduce negatives, so the matched example must be preferred over alternatives. A constant representation cannot accomplish that discrimination.

**BYOL uses a different asymmetry.** An online network with a predictor matches a slowly updated target network, and gradients do not flow through the target branch. Its full training dynamics can learn useful features without explicit negative pairs. Stop-gradient by itself is not a general theorem preventing collapse; predictor, normalization, augmentation, and optimization choices matter.

Monitor per-dimension variance, covariance or effective rank, and downstream performance. A representation can look numerically nonconstant yet still preserve little useful information. Positive alignment and discrimination or feature diversity must be assessed together rather than declaring success because an alignment loss became small.
@@ 1033
**Temperature controls how sharply score differences become probabilities.** InfoNCE divides similarities by tau before softmax. A small tau makes a slight similarity advantage dominate, focusing the loss on high-scoring competitors. [1] [2]

For one positive score 0.8 and one negative score 0.7, tau=1 gives positive probability about 0.525; tau=0.1 gives about 0.731. If the negative instead scores 0.9, the low-temperature loss reacts strongly to that mistake.

The derivative with respect to score j is $(p_j-\mathbf1[j=+])/\tau$. This shows both rescaling and concentration. However, if the positive already wins by a fixed margin, wrong-candidate probabilities vanish exponentially as tau approaches zero, and gradients can vanish despite the reciprocal-temperature factor.

**Very low temperature is not free hard-negative mining.** False negatives become especially damaging, numerical stability matters, and ties have different limiting behavior. Use stable log-sum-exp and explicit multiple-positive handling. Tune temperature through retrieval performance and embedding diagnostics rather than assuming sharper probabilities always create better representations.
@@ 1034
**A different batch row is not necessarily a different meaning.** In a large contrastive batch, two captions can describe the same item, two queries can be paraphrases, or two images can show equivalent relevant content. Treating all off-diagonal pairs as negatives can push genuine matches apart. [1] [2]

Suppose “wireless headphones” and “Bluetooth headphones” are both valid descriptions of the same product. An identity-matrix target rewards one while penalizing the other solely because of row placement.

**Make the positive relation explicit.** Deduplicate exact and near duplicates, mask known related items, or assign multiple positives. One objective rewards the total softmax mass over all valid positives. Averaging each positive's log probability is another objective with different behavior, so select deliberately.

Hard-negative mining increases the need for careful labels because semantically close candidates are also more likely to be valid answers. Evaluate on independent relevance judgments and inspect mined examples. Larger batches provide more comparison diversity, but they also increase opportunities for false-negative collisions; batch size alone does not establish better supervision.
@@ 1035
**A visual adapter must learn how to express image features in a space the language model can use.** If both large pretrained components change rapidly from the first step, the alignment target moves while being learned. Freezing them initially and training the projection can make this first stage more stable. [1] [2]

After alignment, selectively unfreeze layers or adapters for instruction tuning, often with smaller learning rates for pretrained weights. Include diverse image-language tasks and language-only evaluation to detect destructive adaptation.

**Verify that the image matters.** Swap in a different image while keeping the question fixed. If the answer barely changes when it should, the system may be exploiting textual priors rather than visual evidence. Test OCR, spatial relations, and counterfactual details directly.

The staged recipe is a useful baseline, not mandatory for every architecture or data scale. Some domains need substantial visual adaptation; others benefit from keeping most weights frozen. A low projection loss can coexist with shortcut learning. Select the training schedule from grounded task quality and retention, not simply successful convergence of the adapter objective.
@@ 1036
**Adding new classes changes the competition among all output scores.** Training a 50-class model only on ten new classes pushes features and logits toward the new labels, potentially making old examples look like new classes. Forgetting can happen even if old output weights remain present. [1] [2]

Replay a bounded, representative set of old examples while learning the new ones. Distillation can encourage the new model to preserve selected old predictions or features. The buffer must cover important variation, not just easy prototypes.

**Cosine classifiers control one source of imbalance.** Normalizing class weights and features removes logit changes caused purely by vector norm, using a shared scale. It does not automatically correct prior imbalance or biased replay sampling.

Evaluate old-class retention, new-class learning, and calibration on a balanced validation design. Respect deletion and privacy requirements when storing exemplars. When replay is unavailable, compare constrained updates or other retention methods, recognizing their limitations. A single overall accuracy can hide that the model learned all new classes by sacrificing an old minority class.
@@ 1037
**Federated clients may be optimizing different local problems.** One device may contain mostly one label or user behavior, another a very different distribution. Several local steps can move their models far apart before averaging, producing client drift. [1] [2] [3]

FedAvg combines updates, often weighted by client sample count. FedProx adds a penalty for moving away from the current global weights, limiting how far local optimization departs. SCAFFOLD uses control variates to correct systematic differences between local and global update directions.

**Stronger correction can also limit useful personalization.** Tune local steps, learning rate, participation rate, and proximal strength under realistic heterogeneous data and unreliable devices. Compare communication rounds and local compute as well as global accuracy.

Inspect worst-client and subgroup performance: an average can improve while a small group deteriorates. Secure aggregation and differential privacy address information exposure, not statistical heterogeneity. Likewise, solving drift does not establish privacy. Treat optimization, participation, fairness, and privacy as interacting but distinct design questions.
@@ 1038
**An update can reveal constraints on the data that produced it.** Even without raw examples, a shared gradient depends on inputs, labels, model weights, and batch composition. Under favorable conditions for an observer, those constraints can reveal sensitive information, which is why federated storage alone is not a complete privacy argument. [1] [2]

Small batches and detailed update access can be especially revealing. The actual threat depends on what the observer knows and whether they see individual clients or only an aggregate.

**Use defenses with explicit guarantees and assumptions.** Secure aggregation hides individual contributions within a sufficiently populated aggregate under its protocol assumptions. Differential privacy clips contributions, adds calibrated noise, and accounts for repeated releases. Access controls and limited retention reduce unnecessary exposure.

Larger batches or compressed updates may complicate some reconstruction attempts but are not formal privacy guarantees. Evaluate leakage only in an authorized environment with realistic attacker access. Distinguish preventing direct observation of one client's update from bounding the information contained in the released aggregate; those are different protections.
@@ 1039
**A watermark adds a small statistical preference that a detector knows how to look for.** A secret, context-dependent rule divides candidate tokens into favored and unfavored groups. Generation slightly prefers the favored group, leaving a detectable excess over many tokens. [1]

In a simplified null model with favored fraction gamma, T tokens have expected favored count gamma times T and standard deviation $\sqrt{T\gamma(1-\gamma)}$. The standardized excess is a z-score. Longer text can provide more statistical power under the model assumptions.

**Text is not a sequence of perfectly independent coin flips.** Repetition, low-entropy passages, editing, and mixed authorship affect the test. Paraphrasing can weaken the signal, and repeated testing changes false-positive considerations.

Calibrate thresholds on relevant human and generated text. A positive result is evidence for a particular keyed watermark mechanism, not universal proof of AI authorship. An absent watermark does not establish human origin. Protect the key and communicate uncertainty; a detector's statistical output should not be treated as an infallible identity or provenance certificate.
@@ 1040
**Extraction monitoring looks for suspicious use patterns, but ordinary users can produce similar patterns.** Broad systematic probing, repeated boundary-focused queries, or coordinated accounts may indicate attempts to imitate a model. They may also reflect legitimate benchmarking or a large integration. [1] [2]

Combine behavioral signals with authenticated usage context and documented expectations. Quotas, credential controls, anomaly review, and appropriate output minimization can reduce unnecessary exposure. Avoid returning internal details that the product does not need.

**Evaluate false positives and business impact.** Blocking every high-volume client can harm legitimate use. Provide an escalation path and retain enough evidence for review without overcollecting sensitive content.

Watermarks or fingerprints may add evidence under their assumptions but are not perfect ownership tests. If an API intentionally exposes useful input-output behavior, some degree of imitation may remain possible. The realistic objective is managing exposure and detecting misuse with defensible evidence, not claiming that monitoring can make functional replication impossible.
@@ 1041
**A backdoor can be narrow enough to evade ordinary validation.** The model may behave normally on common inputs but fail under a particular condition. Therefore high average accuracy after training or mitigation does not establish that all conditional behavior is clean. [1]

The number “500 poisoned examples” alone says little about success. Dataset size, duplication, influence, model capacity, and training procedure all matter. Controlled research demonstrates possible persistence; it does not prove that every unusual response is evidence of deliberate poisoning.

**Defensive work starts with provenance and controlled evaluation.** Restrict dataset changes, record source and version lineage, and investigate unexpected clusters or label changes. Use trusted held-out data and targeted evaluation slices in an isolated environment.

After mitigation, retest both ordinary utility and the suspected conditional failure, and retain rollback options. A generic safety-tuning pass is not proof of removal. Conversely, avoid accusing a source based on one anomalous output: distinguish reproducible conditional behavior, accidental model error, and evidence of malicious data modification.
@@ 1042
**Automated red teaming is a search for failures, and the evaluator can itself be wrong.** A generator proposes diverse tests against an authorized isolated system; a judge identifies candidate failures; reviewers verify whether the observed behavior actually violated the intended requirement. [1] [2]

Optimizing only an attack-success score can cause the generator to repeat one pattern or exploit weaknesses in the judge. Include coverage and diversity, and preserve examples with their model, tool, and environment versions.

**Separate discovery from final evaluation.** If defenses are repeatedly tuned on the same prompts, success on those prompts may reflect memorization. Maintain independent tests and regression cases for previously verified failures.

Measure severity, reproducibility, false positives, and impact on benign tasks. Findings should lead to concrete changes such as tool restrictions, validation, or better handling of untrusted content, followed by retesting. A dashboard of adversarial prompts is not an improved system until verified failures have been addressed. Keep tools and data controlled so testing cannot affect unrelated users or external systems.
@@ 1043
**A confident softmax can simply mean one class score is much larger than the others.** It does not directly say whether the input resembles training data or whether another plausible model would disagree. A model can be confidently wrong on an unfamiliar input. [1] [3]

Deep ensembles train multiple models and compare their predictions. MC dropout samples predictions with stochastic dropout active. If members confidently choose different classes, their average can reveal disagreement that one model's softmax hides.

**Agreement is not proof of knowledge.** Models can share architecture, data, and blind spots, so every member may make the same unsupported prediction. Evidential approaches parameterize uncertainty over class probabilities but also depend on training assumptions and can fail under shift.

Evaluate calibration, out-of-distribution behavior, and risk–coverage curves: as uncertain cases are deferred, does the error rate on retained cases fall? Compare entropy of the average with average member entropy to separate some disagreement from shared ambiguity under the chosen approximation. Treat uncertainty estimates as signals to validate, not automatic certificates of reliability.
@@ 1044
**Some uncertainty comes from variable outcomes; some comes from limited knowledge.** With the same observed features, repeated outcomes may differ because of noise or unmeasured factors. That is aleatoric uncertainty relative to those features. Epistemic uncertainty concerns the fitted model or unknown parameters. [1]

A noisy sensor can create variability even after observing many examples. A region of input space with almost no training examples creates uncertainty that more representative data may reduce.

**The variance identity separates these contributions inside a model.** Average the conditional outcome variance over parameter uncertainty, then add the variance of the conditional means. The first term describes within-model variability; the second describes disagreement among plausible models.

The distinction depends on available information. Adding a better sensor or relevant feature can reduce uncertainty previously treated as irreducible. A misspecified model can also underestimate both components. Validate intervals and behavior under shift rather than assuming a mathematically tidy decomposition is empirically accurate. More data help most when they address the missing knowledge rather than simply repeat the same limited observations.
@@ 1045
**Let the model predict both the expected value and how noisy that prediction is.** For a Gaussian output, predict mean mu and log variance s. Exponentiating s gives a positive variance without requiring a constrained raw output. [1]

Ignoring a constant, the loss is half of $e^{-s}(y-\mu)^2+s$. The first term penalizes residuals relative to predicted variance. The second prevents the model from avoiding all error penalties by declaring infinite uncertainty.

**Follow the gradients.** The mean gradient is $e^{-s}(\mu-y)$. The log-variance gradient is $\tfrac12[1-e^{-s}(y-\mu)^2]$. With residual 2 and variance 1, the latter is -1.5, so gradient descent raises predicted log variance. With a tiny residual relative to variance, it pushes variance downward.

Use stable scale limits and inspect interval coverage by input region. The model can learn larger spread for genuinely noisy inputs, but a single Gaussian cannot represent two separated plausible outcomes well. Predicting uncertainty is useful only if the distributional family and calibration match the task.
@@ 1046
**A mean can lie where almost no real outcome occurs.** If a target is equally likely near -3 and +3, squared-error regression predicts near zero. That minimizes expected squared error but hides the two plausible alternatives. [1] [2]

A mixture density network predicts component probabilities, means, and scales conditional on x. Softmax makes mixture weights sum to one; a positive parameterization ensures valid scales. The resulting density is the weighted sum of component densities.

**Train the density, not only its mean.** Evaluate negative log likelihood using log-sum-exp for stability. At inference, report mode-specific probabilities, sample outcomes, or derive an appropriate decision from the full distribution. Averaging the modes back to zero can discard the benefit.

Guard against components collapsing to tiny variance around individual observations, unused components, and interchangeable component labels. More components increase flexibility but require more evidence. Check held-out likelihood and predictive calibration, including tails. A multimodal model is justified when the conditional outcomes really need it, not simply because a more complicated density can fit training data better.
@@ 1047
**Physics-informed learning adds equations as training evidence.** A network approximates a field such as temperature over space and time. Automatic differentiation computes derivatives, and the loss penalizes violations of a governing PDE plus mismatches to observations and boundary conditions. [1]

For diffusion, a residual might be $\partial_tu-\nu\partial_{xx}u$. Evaluating it at collocation points encourages the learned field to satisfy the equation there.

**Soft penalties and hard constraints differ.** A small sampled residual does not prove the PDE holds everywhere. A parameterization such as $u(x)=g(x)+x(1-x)f_\theta(x)$ preserves g's endpoint values on [0,1] exactly, while leaving the interior learnable.

Softplus can enforce a positive scalar output, but does not automatically make a function monotone or convex. Those require suitable architectures or parameterizations. Balance loss scales so one residual does not dominate merely because of units. Validate against independent numerical solutions, boundary checks, and conservation errors across the domain, including regions not used as collocation points.
@@ 1048
**A neural operator learns a rule from one function to another.** Instead of predicting a single value from a fixed vector, it might map a spatial material-coefficient field to the solution field of a PDE. Training examples are different input functions and their corresponding solutions. [1] [2]

FNO mixes selected Fourier modes of a field and returns to physical space with local processing and nonlinearities. DeepONet combines a branch representation of the input function with a trunk representation of the output coordinate.

**Shared structure can support new sampling grids.** The learned map is not necessarily tied to one output coordinate list in the same way as an ordinary fixed-size regression head. But this does not establish accurate prediction at arbitrary resolution or outside the training physics.

Retained modes, sampling density, aliasing, boundary conditions, and parameter ranges limit generalization. Test grid refinement, unseen coefficient fields, conservation, and comparison with a numerical solver. Include preprocessing and discretization in runtime comparisons. A fast prediction is useful only at an explicitly measured accuracy and within a stated domain of validity.
@@ 1049
**Interleaving compares rankers within a shared result impression.** Team-draft interleaving alternates selections from two ranked lists, randomizes the starting side, and avoids duplicates. Clicks are attributed to the ranker that contributed each item, yielding a preference signal. [1]

Sharing one impression can reduce some noise from comparing separate user groups. It is especially useful when the question is which ranking users prefer. However, attribution rules, position effects, and ranker similarity influence sensitivity.

**Shadow traffic answers a different question.** A candidate system runs on real requests without showing its results. This reveals latency, errors, and output differences, but users cannot react to an output they never see, so it does not estimate its causal engagement effect.

Use shadowing for operational readiness, interleaving for suitable relative ranking comparisons, and randomized A/B tests for downstream business outcomes and guardrails. There is no universal ten-to-fifty-times speedup. Match the experiment to the claim you want to make rather than treating all online evaluations as interchangeable.
@@ 1050
**An experiment needs a decision rule before the results are visible.** Define the primary outcome, diagnostic metrics, guardrails, and integrity checks. The primary outcome answers the main product question; diagnostics explain mechanisms; guardrails limit unacceptable harm; invariants check whether the experiment ran correctly. [1] [2]

For a recommender, qualified engagement might be primary, coverage and clicks diagnostic, complaints and latency guardrails, and assignment/logging consistency invariants. Specify exposure unit, duration, useful effect size, and stopping rules.

**A positive p-value result is not an automatic launch decision.** A statistically detectable gain can be too small to matter, accompanied by a failed guardrail, or invalidated by sample-ratio mismatch. Repeated looks and many tested metrics also affect inference.

Inspect confidence intervals and practical tradeoffs. Investigate instrumentation or assignment failures before rerunning, and do not discard inconvenient metrics after seeing them. Ship when the evidence supports the intended causal claim and the complete product outcome is acceptable. The experiment's job is to inform that decision, not merely produce one favorable number.
''')
