from expand import write
write(r'''
@@ 903
**The funnel spends expensive computation only after cheap stages have narrowed the choice.** Scoring ten million items with a rich user-item interaction model is usually too costly for each request. Retrieval finds a broad shortlist using precomputed embeddings, inverted indexes, or rules. A lightweight pre-ranker reduces that shortlist before a heavy ranker evaluates detailed interactions. Re-ranking arranges the final list under diversity and eligibility constraints. [1] [2]

Imagine retrieval returns 2000 items, pre-ranking retains 300, heavy ranking selects 50, and re-ranking fills 20 feed slots. If an excellent item is discarded at the first stage, no later model can recover it. Candidate recall is therefore a first-class metric, not just final ranking accuracy.

**Latency is a shared budget.** A hypothetical 100 ms service budget might reserve 15 ms for retrieval, 10 for pre-ranking, 40 for heavy ranking, 10 for re-ranking, and 25 for features and network overhead. Those numbers illustrate allocation, not universal targets. Parallel candidate sources and feature fetches may overlap, while one slow dependency can dominate p99.

Choose stage complexity by measured marginal quality and tail latency under load. Cache candidate-independent work and retain valid fallback paths. Evaluate filters separately: policy eligibility, inventory availability, and a learned low score have different meanings. The best heavy ranker is ineffective if upstream filtering consistently removes the useful candidates.
@@ 904
**Feature interactions explain why the same item appeals differently in different contexts.** A user liking hiking gear and an item being waterproof may be more predictive together than either feature alone. Recommendation architectures differ in how they represent such combinations. [1] [2] [3]

Wide & Deep combines explicitly designed linear crosses with learned neural representations. The wide part can memorize a useful rule, while the deep part can generalize through embeddings. DeepFM adds factorization-machine pairwise interactions, using shared embeddings to learn which categorical values work together without storing every pair separately.

DCN explicitly constructs feature crosses across layers; DCN-v2 makes those interactions more expressive. DLRM embeds categorical inputs, processes dense inputs, and combines them through interaction operations before prediction. DHEN combines heterogeneous interaction modules so different forms of interaction can contribute at different depths.

**Compare the interaction mechanism, not just model size.** If a feature never arrives at serving time or its label leaks the outcome, no interaction network fixes the problem. Use the same data, features, and validation split for a fair comparison. Include calibration, embedding memory, feature-fetch latency, and p99 serving cost. A small log-loss gain may not justify a much heavier model in a stage that scores hundreds of candidates per request.
@@ 905
**Two-tower retrieval is fast because the item representation does not depend on the current user.** Compute every item's vector in advance, index those vectors, and compute one query/user vector per request. A dot product then gives a score compatible with efficient nearest-neighbor search. [1]

Now add “this user's historical CTR on this author.” The same author has a different value for each user, and the same user has a different value for each author. It is inherently a property of the pair. Putting it into the item tower would require recomputing that item's vector for each user, defeating the shared catalog index. Putting one candidate-specific value into the user tower would require a different query vector for each candidate.

**Use the cross-feature once the candidate set is small.** A pre-ranker or heavy ranker can fetch that user-author statistic for hundreds of retrieved items and evaluate rich interactions directly. Retrieval can still approximate the preference using separate user-interest and author/content embeddings.

The restriction is about arbitrary joint features, not every mathematical interaction. A structured feature that factorizes into independently computed terms may fit the two-tower score. Check that factorization explicitly. Also keep query and item encoder versions compatible: even correctly separated towers can fail if the index contains vectors from an incompatible model version.
@@ 906
**A long history is useful only if you can find the parts relevant to this candidate.** A user may have 10,000 clicks spanning cooking, software, and travel. Averaging them all can blur current interests, while attending to every event for every candidate is expensive. [1] [2] [3]

DIN makes history attention target-aware: when ranking a camera, photography-related events receive more weight than cooking events. DIEN additionally models how interests evolve through the sequence, helping distinguish old and recent patterns. Both become costly if applied naively to an unbounded history for hundreds of candidates.

**SIM separates finding evidence from interpreting it deeply.** A cheap first stage searches a relevant subsequence, perhaps by category or learned similarity. A richer second stage attends over that bounded subset. If 10,000 events are reduced to 100 plausible matches, the expensive interaction stage has far less work, though retrieval can miss useful older evidence.

Maintain compact session state, indexed long-term history, and cached candidate-independent summaries. Batch candidate scoring and count feature access and serialization in the 20 ms target; a warm model-only benchmark is insufficient. Evaluate quality as the subsequence budget changes, including users with diverse interests and useful old events. Recency is one clue, not proof that an old interest disappeared.
@@ 907
**Generative retrieval predicts an address instead of comparing a query with every stored vector.** An item's identifier is expressed as a sequence of tokens. Conditioned on the user or query, the model generates that sequence, and a lookup maps the completed identifier to an item. [1]

A hierarchical semantic ID might first choose a broad region of item space and then progressively finer codes. TIGER builds such codes from quantized item representations. The probability of a complete ID is the product of the probabilities of its successive tokens given the preceding tokens. Beam search keeps several promising partial IDs to produce multiple candidates.

**This changes where the index lives.** In vector retrieval, updating an item's embedding and index entry can directly change its searchable representation. In generative retrieval, successful prediction also depends on the model learning the relevant ID sequences and their relation to queries. New items, code collisions, and invalid sequences therefore need explicit handling.

Constrain generation to valid IDs when appropriate, resolve collisions, and apply eligibility checks before returning items. Compare recall, latency, freshness, and tail coverage against a two-tower/vector-index baseline. Generating an ID does not eliminate catalog management or downstream ranking; it replaces one candidate-generation mechanism with another whose update and search tradeoffs must be measured.
@@ 908
**Sharing helps when tasks need similar features and hurts when their updates conflict.** Click prediction may favor attention-grabbing content, while satisfaction prediction may penalize that same content. If both tasks must use exactly the same representation, improving one can damage the other: negative transfer. [1]

A simple gradient picture helps. If task A wants a shared parameter to increase and task B wants it to decrease, their updates can cancel or one can dominate. Differences in label frequency and loss scale can make the larger task win even when the smaller task is more important.

**MMoE gives tasks different mixtures of shared experts.** Each task has its own gate, so click and satisfaction models can emphasize different expert outputs. PLE makes shared and task-specific paths more explicit across layers, reducing compulsory sharing where tasks diverge. Neither architecture guarantees compatibility or fixes biased labels automatically.

Compare each task against a separate-model baseline, and inspect missing labels and gradient/loss scales. A weighted average can hide unacceptable deterioration in a priority task. Use explicit constraints or a Pareto comparison to decide the tradeoff. Architecture is one control; better label definitions, task weighting, and training populations may matter just as much.
@@ 909
**An auction needs comparable expected value per opportunity, not just the largest number printed on a bid.** For a cost-per-action bid, payment occurs only when the relevant action happens. Expected value per impression therefore multiplies the action value by the probability of reaching that action. [1]

Suppose ad A bids 20 currency units per conversion, has CTR 0.02, and post-click CVR 0.10. Its expected value per impression is $20\times0.02\times0.10=0.04$, or 40 per thousand impressions. Ad B bids 10, with CTR 0.05 and CVR 0.12, giving 0.06 per impression, or 60 per thousand. B has a lower bid but greater expected value under this simplified model.

**Units determine the formula.** A CPC bid needs only click probability; a CPM bid is already per thousand impressions. The question's product without the factor 1000 is expected value per impression, not literally per mille. Conversion probability must also use the intended attribution definition and window.

Allocation scores do not fully specify payments. GSP, VCG, and first-price rules have different pricing and incentive properties. Real systems add relevance, quality, budget, pacing, reserves, and eligibility. Work through the actual mechanism rather than assuming the expected-value score alone describes the whole auction.
@@ 910
**Ranking and probability accuracy answer different questions.** AUC asks whether positives tend to score above negatives. Calibration asks whether cases predicted at 10% actually succeed about 10% of the time. If a probability enters expected revenue, budget planning, or a threshold, its numerical scale matters. [1]

But the question overstates the contrast: organic ranking can also combine probabilities with value weights or thresholds, so correct order alone is not always sufficient there. Conversely, a common positive rescaling can leave a particular auction's allocation unchanged.

**Work through a quality-adjusted CPC GSP example.** Let A bid 2 with predicted CTR 0.10, giving score 0.20. B bids 3 with CTR 0.05, giving score 0.15. A wins, and a simplified next-score-over-own-CTR price is 0.15/0.10=1.5 per click. Double both CTRs: scores become 0.40 and 0.30, and the price is 0.30/0.20=1.5 again.

That cancellation can fail with reserves, clipping, heterogeneous distortions, or other payment rules. VCG analysis must specify welfare units and how probabilities enter externality payments. Calibration remains important for interpretable expected outcomes, but “doubling every probability doubles every price” is false in general. Audit calibration by campaign and time, then analyze the exact allocation and payment equations.
@@ 911
**A calibration plot compares promised frequency with observed frequency.** Group examples with similar predicted probabilities. If 1000 examples average 0.20 but only 100 are positive, that bin has a 0.10 probability gap. ECE averages absolute bin gaps weighted by bin size. [1] [2]

Brier score instead averages $(p-y)^2$ per example. Predicting 0.9 for a positive incurs 0.01; predicting 0.9 for a negative incurs 0.81. It evaluates probability quality without choosing bins, though its decomposition includes more than calibration alone.

**Normalized cross-entropy supplies a baseline comparison.** Under the convention used here, divide model log loss by the log loss of a constant predictor using the reference prevalence. If baseline loss is 0.20 and model loss is 0.15, NE=0.75: the model has 25% lower loss under that baseline definition. Other naming conventions exist, so state yours.

ECE can hide errors inside wide bins or opposing subgroup errors. Log loss strongly penalizes confidently wrong predictions. Evaluate reliability curves, slices, ranking, and online utility together. Published use of a metric does not establish a company's universal current “north-star” metric; the mathematical reason to use normalized loss is its reference to a specified baseline, not an appeal to a company name.
@@ 912
**Clicked impressions are a selected population.** People click ads that attracted their attention, so their features need not represent all displayed ads. A CVR model trained only on clicks has sparse labels and may generalize poorly when used across impression space. [1]

ESMM uses the identity $P(click,conversion\mid x)=P(click\mid x)P(conversion\mid click,x)$. One component predicts CTR, another predicts conditional CVR, and their product predicts the joint click-and-conversion event. CTR and the joint event can both be supervised over all impressions.

**A counting example.** Out of 10,000 similar impressions, suppose 200 clicks and 20 attributed post-click conversions occur. CTR is 0.02, conditional CVR is 0.10, and joint probability is 0.002. The product connects the sparse conversion task to a much denser impression-level task while shared representations can transfer information.

An unclicked impression is negative for the joint event; that is not the same as observing what would happen if its user were forced to click. ESMM does not automatically identify that causal counterfactual or remove all confounding. Delayed labels, tiny CTR probabilities, and misspecified models still matter. Evaluate mature outcomes, joint calibration, and performance on the clicked subset instead of treating the chain-rule identity as a guarantee of unbiased causal prediction.
@@ 913
**“No conversion yet” is not the same label as “will not convert within the target window.”** If the target is purchase within 30 days, a click from ten minutes ago has barely been observed. Marking it permanently negative creates a bias toward fast conversions and against fresh cohorts. [1] [2]

Suppose 100 clicks will eventually produce 10 purchases, but only two occur on day one. A day-one snapshot interpreted as final labels suggests 2% conversion instead of the eventual 10%. Waiting for maturity solves part of the labeling problem but makes training data less fresh.

**Censoring-aware methods use the observation time.** Survival modeling distinguishes an observed event from a case that has survived without an event only up to its current age. A delay model can estimate how much of the eventual outcome is likely to be visible. Importance weighting can correct some observation biases when the delay and sampling assumptions are valid.

Keep event time, ingestion time, and attribution time separate, and revise labels as outcomes arrive. Evaluate on fully matured cohorts to avoid rewarding a model for reproducing immature labels. Check delay drift: a promotion, payment change, or seasonal pattern can alter how long purchases take. Weight clipping and uncertainty matter when correction weights become large. A fresh-data metric should not silently redefine a 30-day target as a one-day target.
@@ 914
**Cold start means missing evidence, not evidence of dislike.** A new user has no interaction history, while a new item has no learned engagement record. The system needs useful initial representations and a controlled way to gather feedback. [1] [2]

For users, use current session actions, stated interests, context, and a diverse quality prior. A first search for hiking boots is more informative than treating the user as an average of all established users. For items, content embeddings from text, images, or audio can place a new upload near related items before its ID embedding is well trained.

**Exploration creates the missing evidence.** A contextual bandit balances estimated utility with uncertainty. UCB adds an uncertainty bonus; Thompson sampling samples plausible reward parameters and acts under that sampled model. Both require a meaningful uncertainty model and careful feedback logging. Exploration should remain within quality and eligibility constraints.

Meta-learning may help when many related adaptation tasks and few-shot assumptions match deployment, but it is not required for a strong baseline. Track cold-start cohorts separately, including useful first exposure and subsequent satisfaction. A global score dominated by established traffic can conceal total failure on new users or creators. Do not equate unexposed items with observed negative feedback.
@@ 915
**A separate vector for every ID grows linearly with the catalog.** With one billion IDs, 32 dimensions, and four bytes per value, raw embedding weights alone use 128 billion bytes. Optimizer states, keys, and metadata can add much more during training. [1]

Hashing caps the table size by mapping several IDs to the same slot. That saves memory but forces collisions to share parameters. Quotient-remainder representations use two smaller tables: for ID i and divisor m, one table represents the quotient and another the remainder, and their embeddings are combined. Compositional schemes generalize the idea to combinations of learned code vectors.

**Allocate capacity where evidence supports it.** Frequency-adaptive dimensions give common IDs richer vectors and rare IDs smaller ones. Eviction removes stale entries but needs a content or unknown-ID fallback. An item absent for 30 days may be seasonal rather than irrelevant, so retention should reflect the product instead of a universal timeout.

Measure quality for frequent and rare IDs, collision interference, cache behavior, and update throughput. More component lookups can save storage while increasing random memory access. Keep training and serving mappings synchronized: an ID assigned to different slots across the two systems yields incorrect predictions even if both tables fit comfortably in memory.
@@ 916
**Query understanding should preserve what the user meant while making it easier to retrieve.** Consider “blu trail shoes under 80.” A pipeline may suggest “blue,” recognize a product-search intent, identify color and category, and extract a price ceiling. Those pieces should not all be treated as interchangeable text tokens. [1]

Spell correction generates plausible alternatives; tokenization and normalization must match the index. Intent classification can distinguish navigating to a known brand from exploring information or shopping. Entity extraction identifies brands, materials, sizes, colors, and numeric constraints. Expansion can add useful synonyms, such as a recognized product-category equivalent.

**The order is not rigid.** An unusual brand may look like a typo, so entity recognition can protect it before correction. Exact SKUs or code queries often need minimal normalization. A price cap or “without peanuts” is a hard constraint, while a broad semantic synonym is usually an optional retrieval hint.

Keep the original query and log rewrites so failures are diagnosable. Retrieve from original and expanded forms when appropriate, then rerank with the actual intent and constraints. Evaluate exact identifiers, negation, rare brands, and zero-result cases separately. More retrieved documents are not an improvement if the rewrite changed the user's request.
@@ 917
**Zero results can mean several different failures.** The index may contain no matching text, inventory may be unavailable, or valid candidates may all violate the user's filters. Define where in the pipeline ZRR is measured so those cases are not mixed together. [1]

For 1000 eligible searches with 70 empty result sets, ZRR is 7%. Returning one unrelated item for every search would reduce it to zero while making the product worse. The metric needs a relevance and usefulness counterpart.

**Relax interpretation gradually.** Correct a clear typo, consider synonyms, and broaden optional descriptive words. If a user asks for “waterproof trail shoes under 80,” related phrasing for trail shoes may be appropriate; silently dropping waterproofing or the price cap is not an equivalent answer. If no exact matches remain, label alternatives and explain which condition differs.

Use relevance thresholds and permit an honest empty result when available alternatives are poor. Analyze queries that users reformulate, inspect inventory limitations, and track useful-result rate and satisfaction. Keep original and relaxed queries for comparison. The goal is helping the user find an acceptable item, not optimizing an empty-set statistic independently of meaning.
''')
