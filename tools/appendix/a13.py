from author import write
write(r'''
@@ 903|logq din dlrm irl scaling
Retrieval uses precomputed vectors, inverted indexes, and simple rules to reduce millions of items to thousands. Pre-ranking applies a cheap scorer to reduce thousands to hundreds. Heavy ranking uses richer interactions and histories on that shortlist. Re-ranking enforces diversity, eligibility, and business constraints. [1] [2]

An **illustrative** 100ms server budget might allocate 15ms retrieval, 10ms pre-ranking, 40ms heavy ranking, 10ms re-ranking, and 25ms for feature/network overhead; these are design assumptions, not industry constants. Stages may overlap.

Measure candidate recall after every filter and p99 under load. Later stages cannot recover excluded items. Choose model complexity by candidate count, feature-fetch cost, and measured marginal quality, not just by nominal FLOPs. Preserve policy/eligibility checks even in fallback paths.
@@ 904|deepfm dcn dlrm din prml
Wide & Deep combines a linear component over explicit crosses with a neural component. DeepFM shares embeddings between factorization-machine pairwise interactions and a DNN. DCN explicitly builds feature crosses; DCN-v2 increases expressiveness and supports low-rank/mixture variants. DLRM combines dense features with categorical embeddings and interaction operations. [1] [2] [3]

DHEN ensembles heterogeneous interaction modules across depth to represent different interaction patterns; more modules add capacity and serving cost. The architectures address how features interact, not missing data quality or biased labels.

Compare identical input features, training samples, calibration, and latency. Embedding memory and feature retrieval can dominate the interaction network. A model that improves log loss may still be unsuitable if it exceeds the ranking stage's tail-latency budget.
@@ 905|logq sbert din colbert dlrm
A standard two-tower score factorizes as $f(user)^Tg(item)$ so item vectors can be computed once and indexed. A feature depending jointly on this user and this candidate cannot generally be evaluated inside either independently precomputed tower without breaking that separation. [1]

For example, “this user's historical CTR on this author” changes with both sides. Introduce it after candidate retrieval in pre-ranking/heavy ranking, or approximate it through separately encoded user-author preferences. Some structured interactions can be algebraically factorized or stored in user features, so the prohibition is not absolute for every cross-feature.

Verify index freshness and query/item encoder version compatibility. Adding arbitrary pairwise features to retrieval would require scoring the whole catalog or a different indexing strategy, defeating the intended scalability.
@@ 906|din dien sim logq scaling
DIN attends to user-history items conditioned on the current candidate. DIEN models evolving interests with sequence dynamics and candidate-aware extraction. Applying either naively to 10,000 events for every candidate can exceed a 20ms budget. SIM first retrieves a relevant subsequence cheaply, then applies richer attention. [1] [2] [3]

Store long-term history in an indexed representation, maintain compact recent/session state, and cache candidate-independent summaries. Retrieve a bounded number of category/semantic matches per candidate or candidate group, then batch fine scoring.

The 20ms target must include feature retrieval, serialization, and tail load, not only a warm GPU forward. Evaluate whether truncation/search misses older useful interests, and distinguish event recency from genuine preference changes. Measure quality versus history-search budget explicitly.
@@ 907|tiger logq faiss sbert irl
Generative retrieval predicts an item's identifier token sequence conditioned on the query/user context. DSI learns document IDs as model outputs; TIGER builds hierarchical semantic IDs using residual-quantized item representations and trains a sequence model to generate them. [1]

$$p(item\mid x)=\prod_{t=1}^Kp(code_t\mid x,code_{<t}).$$

This replaces explicit nearest-neighbor scoring with constrained sequence generation and an ID-to-item lookup. Beam search returns candidates, but invalid IDs, collisions, catalog updates, and exposure of new items need handling. The “index” is partly encoded in model parameters/ID structure, so freshness may be harder than updating a vector index. Compare retrieval recall, latency, update cost, and tail-item coverage; generative IDs do not remove the need for eligibility filtering and reranking.
@@ 908|mmoe din deepfm prml optim
Negative transfer occurs when shared parameters receive conflicting task gradients or must represent incompatible patterns. Improving click prediction can harm conversion or satisfaction, especially when labels and populations differ. [1]

MMoE uses shared experts with task-specific gates, allowing each task a different mixture. PLE separates shared and task-specific experts across layers, restricting which information is passed where. These designs reduce forced sharing but do not guarantee positive transfer or solve label bias.

Measure per-task gradient alignment, data availability, loss scales, and performance against separate-model baselines. Tune task weighting and missing-label handling. A single weighted average metric can hide a harmed high-priority task; use explicit constraints or a Pareto analysis before selecting the final model.
@@ 909|esmm calibration metrics ope prml
Rank by expected value per impression under the bid's payment basis. For a cost-per-action bid, expected value is $bid\times P(click)\times P(conversion\mid click)$; multiply by 1,000 to express expected value per thousand impressions. For CPC bids, use bid times CTR; CPM bids already have an impression-based unit. [1]

Highest bid alone ignores conversion likelihood; highest CTR alone ignores value. Real auctions also include quality, relevance, reserve, budget, pacing, and policy constraints. Predicted probabilities must correspond to the auction's outcome definitions and attribution window.

Expected-value ranking does not fully specify pricing: GSP, VCG, and first-price mechanisms differ. Analyze allocation, payment, incentives, and user impact separately instead of treating the eCPM score as the whole auction.
@@ 910|calibration esmm metrics ope experiments
Probability accuracy matters when scores enter expected-value bids, budgets, or thresholds. AUC captures ordering only and cannot detect a uniform probability distortion. Organic ranking also often uses probability values for multi-objective scoring and thresholds, so AUC alone is not generally sufficient there either. [1]

If every predicted CTR is multiplied by the same positive constant, pure bid×CTR ordering is unchanged. In a quality-adjusted CPC GSP price that divides the next score by the winner's CTR, that common factor may cancel. Reserves, clipping, budgets, heterogeneous scaling, or other pricing rules can prevent cancellation. VCG effects depend on how modeled welfare and payments are defined.

Therefore doubling probabilities does not universally double prices. Work through the exact mechanism and units, while monitoring calibration by campaign, slice, and time.
@@ 911|calibration metrics esmm prml experiments
ECE bins predicted probabilities and averages absolute confidence-versus-frequency gaps, weighted by bin size. Brier score is mean squared probability error. Normalized cross-entropy divides model log loss by a specified baseline entropy/log loss. [1] [2]

$$ECE=\sum_b\frac{n_b}{N}|acc_b-conf_b|,\quad BS=N^{-1}\sum_i(p_i-y_i)^2,\quad NE=\frac{logloss}{-\pi\log\pi-(1-\pi)\log(1-\pi)}.$$

For this NE convention, values below one beat the constant-prevalence predictor. State the convention because names vary. ECE depends on binning and can hide subgroup errors. Log loss strongly penalizes confident mistakes and is useful for ads probabilities, but no unsupported claim about a company's universal current “north-star” metric is needed. Report ranking, calibration, and online utility together.
@@ 912|esmm ope calibration logq causal
Post-click conversion labels exist on the clicked subset, whose distribution differs from all impressions. ESMM shares representations and trains CTR and click-and-conversion predictions across impression space using the probability chain rule. [1]

$$P(click,conversion\mid x)=P(click\mid x)P(conversion\mid click,x).$$

The product tower receives joint-outcome supervision even for unclicked impressions, while CTR has its own labels. This can reduce sparsity and improve generalization, but does not magically identify conversion under arbitrary forced clicks or remove all confounding. The conditional CVR target and causal conversion effect are different quantities. Delayed conversions, tiny CTR denominators, and model misspecification remain concerns. Evaluate mature labels, calibration, and click-conditioned as well as impression-level metrics.
@@ 913|esmm causal prml forecast metrics
A recent click with no observed purchase is right-censored, not necessarily a true negative for a 30-day conversion target. Treating it as negative systematically underestimates conversion for fresh cohorts. [1] [2]

Model the eventual-conversion probability together with delay, or use survival/hazard methods that contribute a survival probability for censored observations and event density for observed conversions. Importance weighting can correct sampling/observation bias under a valid delay model, but unstable weights and incorrect assumptions can worsen estimates.

Track event, ingestion, and attribution timestamps; revise labels as outcomes mature. Evaluate on fully matured cohorts and monitor delay-distribution drift. Short-window proxy metrics can support operations but should not silently replace the stated long-window conversion objective.
@@ 914|logq clip rl ope prml
For new users, use session/context features, stated preferences, locale, and safe popularity/diversity priors, then update from early interactions. For new items, encode text/image/audio and metadata so retrieval does not depend solely on an untrained ID embedding. [1] [2]

Contextual bandits allocate bounded exploration according to uncertainty and expected utility; record actual propensities for evaluation. Meta-learning can accelerate adaptation when tasks and few-shot assumptions fit, but is not a prerequisite or guaranteed improvement.

Separate lack of evidence from negative preference. Protect users from low-quality or unsafe exploratory items and enforce creator/tenant eligibility. Evaluate cold-start cohorts explicitly, including time-to-first-useful-exposure and long-term quality; a global metric dominated by established users/items can hide failure.
@@ 915|dlrm logq scaling prml sre
ID embeddings cost roughly $Ndb$ bytes before optimizer state. Hashing bounds $N$ but makes different IDs share parameters, creating collisions. Quotient–remainder schemes encode an ID through two smaller tables; compositional schemes combine several code embeddings, trading uniqueness/capacity against storage. [1]

Frequency-adaptive dimensions allocate more capacity to frequent IDs. Eviction removes stale IDs, but needs an OOV/content fallback, consistent training/serving maps, and safeguards for seasonal reappearance. Optimizer state and index metadata can exceed weight storage during training.

Measure tail quality, collision effects, cache hit rates, and update throughput. A 30-day policy is a product assumption, not a universally correct retention period. Compression that saves RAM but sharply increases random accesses can hurt serving latency.
@@ 916|irl sentencepiece sbert splade rag
Normalize carefully, preserve the original query, and generate rather than blindly replace spelling alternatives. Tokenize using index-compatible rules; infer intent and extract entities/constraints such as brand, color, and price. Expand with controlled synonyms or semantic rewrites, then retrieve and rerank. [1]

Ordering is not rigid: entity recognition can prevent a rare brand from being “corrected” into a common word, and code/SKU queries should often bypass aggressive normalization. Keep hard constraints separate from optional semantic hints. Log the rewrite and offer fallback to the original.

Evaluate query slices, zero-result rate, exact-identifier accuracy, and downstream satisfaction. A rewrite that increases recall but changes “without sugar” to “sugar” is a semantic failure despite returning many results.
@@ 917|irl splade sbert metrics experiments
Zero-result rate is the fraction of eligible search requests returning no results under a defined filtering stage. Distinguish no lexical matches, no available inventory, and all candidates removed by policy or user constraints. [1]

Relax in stages: fix clear typos, add synonyms, broaden optional modifiers, then offer related categories with explicit labeling. Preserve hard constraints such as safety requirements, availability, and price caps unless the user is shown and accepts the change. Use relevance thresholds and abstain when alternatives are poor.

Measure useful-result rate and reformulation/conversion alongside ZRR. Returning irrelevant spam can reduce ZRR to zero while making search worse. Audit rare entities, negation, and query rewrites, and retain the original query for comparison.
''')
