from expand import write
write(r'''
@@ 707
**Low-bit quantization replaces a fine numerical ruler with a coarse one.** The goal is to store and multiply model values with fewer bits while preserving useful behavior. Rounding and clipping create errors; calibration and scaling decide where those errors land. Post-training quantization starts from an already-trained model rather than training the whole model in the low-bit format. [1]

**The three methods emphasize different information.** GPTQ uses approximate second-order information to choose weight quantization while compensating for reconstruction error. AWQ uses activation information to identify weight channels whose errors matter especially to outputs, then chooses scaling/protection accordingly. SmoothQuant transfers some activation-channel scale into weights through a function-preserving rescaling before quantization, making activation ranges easier to represent. [1] [2] [3]

**Why activation outliers are troublesome.** Suppose most activation values lie between -1 and 1, but one channel reaches 100. A single scale covering that range leaves few quantization levels for the ordinary values. Activations also vary with the input, so a calibration set that misses unusual requests can underestimate the required range. Weights are fixed after training and easier to inspect exhaustively.

A mathematically equivalent rescaling in full precision need not have equivalent rounding error after quantization; that is precisely why choosing scales can help. Assess model quality, calibration coverage, supported kernels, and actual throughput. “Four bits” describes storage precision, not a universal speed multiplier or a guarantee that every layer should use the same precision.
@@ 708
**KV-cache quantization compresses the model's stored attention history.** Weight quantization compresses the reusable model parameters; KV quantization compresses request-specific keys and values that grow with sequence length. The latter can be especially valuable when long contexts or many concurrent requests dominate memory. [1]

**Why errors can affect attention differently.** A key error changes query–key scores and therefore which positions receive attention. A value error changes the information mixed after those weights are chosen. Because softmax is nonlinear, small score changes can matter more when several candidates have similar relevance or when a rare important token must be retrieved.

**A practical tradeoff.** Reducing two-byte cache values to one-byte values roughly halves their raw payload, before scales and metadata. Four-bit storage can reduce it further, but requires suitable packing and kernels. Per-channel or per-group scales, retained higher-precision recent tokens, and calibration choices can change quality substantially.

Long context does not imply a simple fixed amount of perplexity degradation per token. Attention patterns, scale estimation, outliers, and task demands matter. A low average perplexity change can still hide a failure to retrieve one critical fact far back in the prompt.

Evaluate short and long contexts, different token positions, retrieval/copying tasks, and end-to-end generation quality. Include dequantization overhead and bandwidth effects in performance measurements. An aggressively compressed cache that fits more requests may be a good throughput tradeoff for one workload and unacceptable for another requiring precise long-document reasoning.
@@ 709
**Prefix caching reuses computation for identical earlier tokens.** If thousands of requests begin with the same system instructions and examples, recomputing their prefill every time wastes work. A compatible cached prefix can supply the existing KV state, after which the model processes only the new suffix. [1]

**Prompt order affects reuse.** Put stable shared instructions before request-specific text when that preserves the intended meaning. A timestamp at the beginning changes the prefix immediately; the same timestamp at the end allows earlier stable blocks to match. Semantic similarity is insufficient: the relevant token sequence and execution context must be compatible.

**An agent-loop example.** At turn one, process instructions and the initial request. At turn two, append a tool result and continue from reusable earlier blocks. If the system prompt is edited, states after that edit may change because causal representations depend on preceding context. Reuse stops at the valid prefix boundary rather than pretending the rest is unchanged.

Cache identity should include model/version and any relevant adapter, multimodal input, position, or isolation context. Shared caches must respect tenant and data-access boundaries. Eviction and worker routing matter too: a matching prefix on another replica does not help unless the system can route or transfer it efficiently.

Measure reused token fraction and time-to-first-token rather than merely the number of cache lookups. Prefix caching saves prompt work; it does not skip generating new answer tokens. Avoid promising a hit rate without observing actual prompt repetition, block alignment, and cache capacity under production traffic.
@@ 710
**Training memory contains more than the model weights.** AdamW needs gradients and two moving-average states for every trainable parameter. Mixed-precision training may additionally retain a higher-precision master copy of weights. Activations saved for backward, temporary buffers, communication buckets, and allocator overhead add further memory beyond these persistent states. [1]

**Count a common configuration.** A low-precision weight uses two bytes, a low-precision gradient two, an FP32 master weight four, and each of Adam's two FP32 moments four. Total: 16 bytes per parameter. For seven billion parameters that is 112 billion bytes, about 104.3 GiB, before activations and overhead.

**Why “exactly” needs a configuration.** Some implementations keep gradients in FP32, avoid a separate master copy, shard optimizer states, use lower-precision optimizer states, or keep only selected parameters trainable. BF16's wider exponent range can change numerical choices, but does not by itself determine every state dtype. Count what the actual optimizer/runtime stores.

Activations depend on batch size, sequence length, architecture, and checkpointing. Increasing context length can therefore exceed memory even when parameter-state estimates are unchanged. Frozen-base adapter training also has a different accounting: the base may not need trainable gradients or Adam states, while forward/backward activations can still be substantial.

Use this calculation to identify the dominant term before selecting a technique. ZeRO/FSDP shards persistent states, checkpointing reduces saved activations, and quantization changes representation precision. They address different memory components and can be combined, but their communication and compute costs must also fit the training budget.
@@ 711
**Parallelism splits different dimensions of the training problem.** The same word “distributed” can mean replicating a model across data batches, splitting its tensors, splitting its layers, or splitting its sequence work. The right choice depends on what fails to fit and which communication the hardware can sustain. [1] [2]

**Data parallelism:** each replica holds a model, processes different examples, and synchronizes gradients. It increases throughput but ordinary DDP does not reduce each replica's full model-state memory. ZeRO progressively partitions optimizer states, then gradients, then parameters; FSDP-style full sharding corresponds broadly to partitioning all three, with implementation-specific gathering behavior.

**Tensor parallelism:** divide a large layer's matrix operations across devices. A column-sharded projection produces different output slices; a row-sharded reduction can require combining partial outputs. Communication happens frequently within the forward/backward path, making interconnect quality crucial.

**Pipeline parallelism:** place different layer groups on different devices and send activations between them. Microbatches help keep stages busy, but startup/drain bubbles and imbalanced stage times waste capacity. **Sequence/context parallelism:** divide work or activations along the token dimension, with communication appropriate to normalization or attention across partitions.

**Choose by the limiting resource.** If weights and optimizer states do not fit, state sharding may help. If one layer is too large or needs more compute, tensor splitting may help. If long sequences dominate activations, sequence/context methods become relevant. Hybrid systems use several techniques, but multiplying every parallelism factor blindly can create excessive communication and tiny inefficient local operations.
@@ 712
**The frequency of communication matters as much as its size.** Tensor-parallel devices often exchange results inside many layers of every forward and backward pass. These collectives are on a tightly coupled critical path: the next operation may need their result immediately. A slow interconnect can erase the compute benefit of splitting the matrix. [1]

**Compare two arrangements.** Devices connected through a fast intra-node fabric can exchange partial layer outputs with relatively low latency and high bandwidth. Across nodes, communication commonly has higher latency and lower effective bandwidth, especially under contention. Repeating a collective at every layer amplifies that difference.

Pipeline parallelism communicates mainly at stage boundaries, while data parallelism can aggregate gradients in larger buckets and overlap some synchronization with computation. This often makes them more suitable for slower cross-node links. “More suitable” is a workload judgment, not a rule that cross-node tensor parallelism is impossible.

For example, extremely large layers may require tensor parallelism across nodes simply to fit. A cluster with unusually strong networking may also support it well. Conversely, a poorly configured intra-node topology may underperform expectations. Count communicated bytes, collective startup time, overlap, and local matrix sizes for the actual layout.

Benchmark the complete iteration and inspect all ranks. An apparently communication-bound rank may actually be waiting for another rank's slow compute or input pipeline. Topology-aware placement and balanced work are as important as choosing a familiar parallelism name.
@@ 713
**Backward needs intermediate values from forward.** Saving every activation makes gradient computation convenient but consumes memory proportional to the retained computation graph. Activation checkpointing saves only selected boundaries and recomputes missing intermediate values when backward reaches that region. [1]

**Think of storing milestones instead of every step.** For a chain of 100 layers, saving only segment boundaries reduces persistent activation storage. During backward through one segment, rerun its forward computation to recreate the internal values, calculate gradients, then discard them before moving to the previous segment.

A simple equal-cost chain with segment length s needs roughly $L/s$ saved boundaries plus s temporary internal activations. Balancing the terms gives s near the square root of L, illustrating how substantial memory reduction can arise. Real Transformers have uneven tensor sizes and fused kernels, so this is an explanatory model rather than an exact universal memory formula.

**Why the compute percentage is not fixed.** Extra work depends on which operations are recomputed, how expensive forward is relative to backward, and whether kernels can be reused efficiently. The common “about 33%” figure describes a particular accounting regime, not every implementation.

Recomputation must reproduce the relevant forward behavior, including random-number handling for dropout and other stateful operations. Check gradients against an uncheckpointed reference and test training/eval conditions. The method trades compute for activation memory; it does not reduce model weights or Adam states, so it helps only if activations are a meaningful part of the memory bottleneck.
@@ 714
**LoRA starts as a no-op so adaptation begins from the pretrained model.** Its update is a product BA. Setting B to zero makes BA exactly zero, while random A supplies nonzero directions through which B can receive a gradient. The initial forward behavior is unchanged, but learning can begin. [1]

**Differentiate the product to see why.** If G is the gradient with respect to the effective weight update, the factor gradients have the form $\nabla_B L=GA^T$ and $\nabla_A L=B^TG$, with the adapter scale included as appropriate. At B=0 and random A, the first expression can be nonzero while the second is initially zero. Once B updates, A can receive gradients too.

If both A and B are zero, both gradients are zero under this simple factorization, so the adapter cannot start learning through that path. Randomizing both generally creates a nonzero initial update and therefore perturbs the pretrained function before any task learning has occurred.

**Why zero initialization is safe here but not everywhere.** A single zero-initialized ordinary linear layer can still learn because its gradient depends on its nonzero input. The problem with two zero factors is their multiplicative dependence: each factor's gradient is multiplied by the other zero factor. Initialization rules must be justified for the actual computation graph.

Test initial output equality with the frozen base and inspect first-step gradients of both factors. A parameter being trainable does not imply its first gradient must be nonzero; the intended staged gradient flow is part of this initialization design.
@@ 715
**Adapter placement controls which transformations can change.** Adding LoRA only to query and value projections is a compact starting point: queries influence what attention looks for, and values influence what information it passes onward. But a task may also require changes to keys, output projections, or the feed-forward transformations that process each token. [1]

**Use a task-based intuition.** A modest style adaptation may work with a small set of attention adapters. A substantial domain or reasoning adaptation may benefit from broader coverage, including MLP layers. This is a hypothesis to test, not a universal mapping from task name to optimal layer set.

For a fixed rank, adding more target matrices increases trainable parameter count, optimizer state, and adapter computation. Comparing Q/V-only rank 64 with all-linear rank 64 is therefore not a matched-budget experiment. Consider either equal parameter budget or explicitly report the extra capacity.

**A useful experiment sequence.** Establish the frozen/SFT baseline, try a small targeted configuration, broaden coverage while measuring held-out gains, and inspect regressions on general tasks. Track peak training memory and inference overhead in merged and unmerged modes. Rank, adapter scale, dropout, learning rate, and data quality interact with placement.

“Most benefit” should mean benefit per relevant resource under your workload, not whichever configuration has the highest raw score after using more parameters. Also verify module names in the actual implementation; model families package fused projections differently, and a pattern-based adapter injector can silently target fewer or different layers than intended.
@@ 716
**QLoRA combines a compressed frozen base with small trainable adapters.** The large pretrained weights are stored in a low-bit representation, while LoRA parameters learn the task-specific update. Computation uses suitable higher-precision representations where needed. This saves memory mainly by avoiding full-precision storage and optimizer states for a fully trainable base. [1]

**NF4 is a quantization codebook choice.** NormalFloat4 allocates representable values in a way motivated by approximately normally distributed weights. It is not the same as simply rounding every weight to one of 16 uniformly spaced numbers. Real weight distributions and block scaling still affect quantization error.

**Double quantization compresses the metadata too.** Blockwise quantization stores scales or related constants for many groups. Quantizing those constants reduces their additional memory footprint. The saving comes from a second layer of compression, not from applying the same rounding step twice to each weight.

Paged optimizers use memory-management mechanisms to handle optimizer-state pressure and transient memory spikes. Moving data between memory tiers can avoid out-of-memory failures but can also add transfer overhead; it is not free GPU memory.

**Know what remains expensive.** Activations, LoRA gradients/states, temporary dequantized values, and attention caches still consume memory. The base weights are frozen, but gradients may still propagate through their computations to train adapters. Validate quality against an appropriate higher-precision baseline and use supported kernels; a small stored model is not automatically a fast training configuration.
@@ 717
**These methods restrict adaptation in different places.** LoRA changes effective linear weights through a low-rank update. DoRA separates weight magnitude and direction, adapting directional structure with a low-rank mechanism while treating magnitude separately. Prefix tuning learns additional attention-context vectors. Bottleneck adapters insert small trainable modules into the network. [1] [2] [3]

**A geometric analogy.** LoRA adds a restricted set of directions to a transformation. DoRA explicitly distinguishes how far a weight vector points from which direction it points. Prefix tuning changes the learned context the model attends to, while inserted adapters provide an extra transformation along the computation path. These are related parameter-efficient approaches, not interchangeable file formats.

**Serving consequences differ.** A suitable LoRA update can be merged into a dense weight for inference, though quantized-base details require care. Prefix methods add attention-state/context overhead. Inserted adapters add operations unless a particular structure permits fusion. DoRA's deployment behavior depends on how its decomposition is materialized.

Choose using held-out quality, trainable parameter count, memory, latency, and how many task variants must coexist. A system serving many adapters over one shared base may value easy switching more than the lowest single-adapter latency.

Compare matched data and tuning effort. “Fewer trainable parameters” does not mean “less total memory” in every configuration, and a more expressive adaptation can still overfit poor data. Verify compatibility with the actual base model, projection layout, and inference runtime before treating a research method as a drop-in production option.
@@ 718
**Classic PPO-based RLHF assigns four distinct jobs.** The policy generates answers. The reward model scores their desirability. The critic predicts expected future reward to reduce variance in policy-gradient updates. The frozen reference model defines a behavior anchor, commonly the supervised-fine-tuned starting policy. [1]

**Walk through one rollout.** Give the policy a prompt and sample a completion. Score it with the reward model, compute a penalty for drifting from the reference, and estimate advantages using the critic. PPO then updates the policy with a constrained/clipped objective; the critic learns to predict returns more accurately. The reference and reward model are usually fixed during this particular policy-optimization phase.

**Why the critic is not the reward model.** The reward model judges the sampled answer according to learned preferences. The critic predicts expected return from states or prefixes to provide a baseline. One supplies the objective signal; the other helps estimate how surprisingly good or bad a sampled trajectory was.

**Memory depends on execution choices.** Four logical roles do not always mean four fully separate equal-size models simultaneously resident with identical training state. Components may share structure, run in separate phases/devices, be smaller, or be offloaded. Trainable policy/critic states and rollout activations differ from frozen inference-only states.

Evaluate reward quality and policy behavior independently. A policy can improve its learned reward while becoming verbose, evasive, or otherwise exploiting the scorer. The four-model diagram explains the computation, but does not establish that the reward accurately captures what users want.
@@ 719
**The reward model is an imperfect proxy for the real goal.** It learns from a finite set of preference judgments. Optimizing it aggressively can expose patterns that earn high scores without actually improving helpfulness or correctness. This is reward hacking: the model learns to satisfy the measurement rather than the intended behavior. [1]

**A simple example.** If annotators often prefer detailed answers, a reward model may overvalue length. A policy can then produce unnecessarily long responses to increase reward, even when a short direct answer would be better. Other shortcuts can involve confident tone, repeated phrases, or exploiting weaknesses in the judge.

A KL penalty against a reference policy discourages large distributional drift. It says, roughly, “seek higher reward, but pay for moving too far from a useful baseline.” The coefficient controls a tradeoff: too weak can permit harmful drift; too strong can prevent useful learning.

**Why KL is not a correctness guarantee.** A nearby policy can still produce wrong or harmful answers, and the reference may already contain undesirable behavior. The penalty constrains change, not truth. Conversely, some valuable improvements may require meaningful departures from the reference.

Use independent evaluations, held-out human judgments, and targeted failure tests rather than the training reward alone. Track reward, KL, length, diversity, and task-specific quality together. When these diverge, inspect actual outputs before concluding that optimization succeeded. Regularization makes proxy optimization more controlled; it does not remove the need to validate the proxy itself.
@@ 720
**DPO uses preference pairs to update the policy directly.** In classic RLHF, one first fits a reward model and then runs reinforcement learning against that reward. DPO exploits the mathematical relationship between a KL-regularized optimal policy and its reward, allowing the preference likelihood to be written in terms of policy/reference log probabilities. [1]

**The key relationship.** For a fixed prompt, the KL-regularized optimum has the form $\pi^*(y|x)\propto\pi_{ref}(y|x)e^{r(x,y)/\beta}$ under the relevant convention. Rearranging expresses reward as a scaled log ratio between the optimal policy and reference, plus a prompt-dependent normalizing term.

Preference models compare rewards for two completions of the same prompt. That prompt-only term cancels in the difference. Substituting the policy/reference log ratios into a logistic preference model produces the DPO loss without a separately trained reward network in that objective.

**Intuition through relative odds.** If the policy increases the chosen answer's odds relative to the rejected answer more than the reference did, it improves the preference margin. The objective does not merely maximize the chosen answer's likelihood independently; it compares chosen and rejected changes against the baseline.

This simplifies the training pipeline, but shifts responsibility to preference-data quality and objective assumptions. It does not mean every reward-learning issue disappears or that offline preference optimization can explore outcomes absent from its data. Check masking, sequence log-probability sums, reference consistency, and held-out behavior before interpreting lower loss as broad alignment.
@@ 721
**A relative preference objective can improve a ratio while lowering both probabilities.** Suppose a chosen response's probability falls from 0.2 to 0.1, while the rejected response falls from 0.1 to 0.01. The chosen-to-rejected ratio improves from two to ten, even though the chosen response itself became less likely. This illustrates why a favorable preference margin is not identical to preserving absolute likelihood. [1]

Other failures can arise from noisy preference labels, overfitting, length effects, distribution shift, and a mismatch between the reference and training data. Inspect generations and separate metrics rather than treating one training objective as a complete quality measure.

**The alternatives change different assumptions.** IPO uses a different preference objective motivated by controlling overfitting behavior. KTO works with desirable/undesirable feedback rather than requiring every label to be a matched preference pair. SimPO uses a reference-free formulation with length-normalized sequence scoring and a target margin. These methods are not one common “fix DPO” switch. [2] [3] [4]

**Choose by the data and failure.** If only binary accept/reject labels exist, a pairwise method may require extra construction assumptions. If length dominates preferences, inspect scoring and normalization. If chosen likelihood declines, measure it directly along with task quality and rejected likelihood.

Run matched comparisons on representative held-out prompts and report both strengths and regressions. A replacement objective may reduce one issue while introducing another. Preference optimization remains statistical learning from imperfect signals; objective names should not substitute for understanding what feedback is available and what behavior is being rewarded.
@@ 722
**GRPO uses comparisons within a group of sampled answers.** For one prompt, sample several completions, score them, and compare each reward with the group's average. An answer better than its peers gets positive relative advantage; one worse than its peers gets negative relative advantage. This supplies a baseline without training a separate full critic network for that role. [1]

**Work a tiny group.** Suppose rewards are $(0,0,1,1)$. The mean is 0.5, so the centered rewards are $(-0.5,-0.5,0.5,0.5)$. Dividing by a nonzero group standard deviation puts the advantages on a more comparable scale. Policy optimization then increases the relative likelihood of better-scoring samples under its clipping/KL conventions.

**What is saved and what is not.** Removing a critic can reduce model-state memory and critic-training work. But sampling several completions per prompt still costs substantial generation, and reward computation may also be expensive. The total training economics depend on those costs, not just the number of named networks.

If every answer gets the same reward, centered advantages are zero, yielding little or no within-group learning signal. Sparse rewards, noisy graders, normalization choices, and completion length can therefore influence training strongly. Use numerical safeguards for zero reward variance.

A group-relative baseline does not make the reward trustworthy. For math or code, verifiable outcomes can help, but tests may be incomplete or exploitable. Evaluate generalization and reasoning behavior on independent tasks rather than assuming higher training reward proves better reasoning.
@@ 723
**Outcome rewards judge the destination; process rewards judge intermediate work.** An outcome reward model scores a final answer or completed task. A process reward model evaluates steps along the way, potentially identifying where a solution first becomes invalid. The choice changes how detailed the training or search feedback can be. [1]

**A math example.** Two solutions both output the correct number. One uses valid algebra; the other makes two canceling errors. A final-answer checker rewards both, while a capable process evaluator can distinguish them. Conversely, a valid unusual solution may confuse a process evaluator trained mainly on conventional step patterns.

**Why dense feedback can help.** If the final answer is wrong after twenty steps, an outcome-only signal does not identify which earlier decision caused the problem. Step-level evaluation can guide search, reject bad branches sooner, or provide more informative training targets. It also requires reliable step segmentation and labels, which can be expensive or ambiguous.

Process scores are not formal proofs. A model may reward polished-looking explanations or fail on compressed/novel reasoning. If it only observes written steps, it may not capture the actual computation that led to the answer. Outcome verification also has limits when unit tests or numeric checks miss important cases.

Use a combination where appropriate: executable or mathematical outcome checks, process-based search guidance, and independent human/automated validation. Measure final task success and robustness, not merely agreement with the process scorer. The system should reward correct reasoning because it helps solve the task, not because it matches one preferred writing style.
@@ 724
**Choose where the missing capability belongs.** RAG supplies external information at request time. Fine-tuning changes learned behavior or representations. Long-context prompting places a larger amount of relevant material directly in the request. They can be combined; the decision is not a contest with one universal winner. [1]

**Three concrete needs.** Frequently changing company policies favor retrieval from a versioned, access-controlled source. Consistent output format or domain-specific task behavior may favor supervised adaptation. Comparing a few supplied documents that fit comfortably in context may favor direct long-context prompting, avoiding a separate retrieval failure point.

**Ask five questions before choosing.** How often does the information change? Must answers cite exact sources? Is access different for each user? Is the main problem missing knowledge or incorrect behavior? What latency, cost, and data-labeling budget are available? These determine the tradeoffs more reliably than model popularity.

Fine-tuning is not a convenient transactional database: updates and deletions are difficult to localize, and memorized facts may be recalled unreliably. RAG does not automatically teach the model a new reasoning skill, and a relevant passage can still be misinterpreted. Long context can be expensive and does not guarantee every detail is used correctly.

Start with a measured baseline on real tasks. Compare answer accuracy, evidence support, permissions, freshness, latency, and operating complexity. A hybrid can retrieve current facts while a tuned model follows a specialized workflow. Keep the reason for each component explicit so extra infrastructure solves an observed failure rather than merely making the architecture more elaborate.
@@ 725
**An index trades search work against storage and approximation.** A flat index compares the query with every stored vector, giving an exact baseline under the selected distance. Approximate methods avoid some comparisons or compress representations, accepting a possibility of missing true nearest neighbors. [1]

**IVF narrows where to look.** Cluster the vector space, assign items to coarse lists, and search only selected lists near the query. More probed lists usually improve recall while increasing work. Training the coarse partition and handling distribution changes are part of the cost.

**PQ compresses what is stored.** Split vectors into subparts and encode each with a small codebook index. Distances can be approximated from those codes, greatly reducing vector storage. Compression error can change neighbor ordering, so reranking shortlisted candidates with fuller representations may help.

**HNSW navigates a graph.** Connect vectors to neighbors at several levels, use coarse levels to move rapidly through the space, then refine near the query. It can offer strong recall/latency, but graph edges consume memory and construction/update choices matter. It is not simply a compressed flat array.

Benchmark recall against exact ground truth on representative queries, including filters and new items. Report build time, index memory including metadata, update/delete behavior, and tail latency. A method that wins on unfiltered static vectors may be unsuitable for a catalog with strict access filters and frequent changes. “Nearest in embedding space” also remains separate from human relevance.
@@ 726
**Exact identifiers carry information that semantic similarity can blur.** “AB-1207” and “AB-1208” look nearly identical as text but may refer to incompatible parts. A dense encoder trained for semantic similarity may place them close together or fail to preserve the decisive character difference. Lexical retrieval can require the exact identifier. [1]

**Dense retrieval solves a different weakness.** A query for “shoes for jogging” should find “running footwear” even without identical words. Embeddings can bridge that paraphrase gap, while a basic lexical scorer may miss it. Neither strength eliminates the other method's value.

BM25 uses term occurrence, rarity, saturation, and document length. SPLADE learns sparse vocabulary-based representations that can expand or weight terms. A hybrid combines lexical/sparse and dense candidate lists, then optionally reranks them with a richer model. [2] [3]

**Do not add incompatible scores blindly.** BM25 scores and cosine similarities have different scales and distributions. Use calibrated score fusion or a rank-based method such as reciprocal-rank fusion, with evaluated candidate budgets. Preserve hard exact-match constraints when the task requires them rather than allowing a semantic score to override an incompatible SKU.

Evaluate identifiers, acronyms, rare names, paraphrases, and ordinary natural-language queries separately. A strong average score can hide systematic failure on exact lookup tasks. Tokenization and normalization deserve attention too: stripping a meaningful hyphen or changing case in a case-sensitive identifier can break the lexical route before ranking begins.
@@ 727
**The key difference is when query and document interact.** A bi-encoder creates a query vector and document vector separately, then compares them with a cheap score. A cross-encoder processes the query and document together, allowing their tokens to interact throughout the model. [1] [2]

**Why precomputation changes scale.** A million document vectors can be computed once and stored in an ANN index. Each new query needs one query encoding plus a vector search. A cross-encoder cannot generally precompute its final document score independently of the query, so scoring every one of a million query–document pairs would be expensive.

**Why joint processing can be more precise.** For “a red bag that is not leather,” token-level interaction can help check the negation and material constraint against each candidate. One compressed document vector must support many possible future queries, which can lose fine distinctions. This is a representational tradeoff, not proof that every cross-encoder is better than every bi-encoder.

A practical cascade retrieves a broad shortlist with the bi-encoder, then uses the cross-encoder to rerank perhaps tens or hundreds of candidates. The exact cutoff depends on latency and relevance requirements. The reranker cannot recover a relevant document excluded by the first stage.

Measure first-stage recall and reranked quality separately, along with end-to-end latency. Include query/document length effects and batching. A perfect reranker on an artificially excellent shortlist can exaggerate production quality if the real retriever misses difficult candidates.
@@ 728
**ColBERT preserves several vectors per document instead of compressing it into one.** Query and document token representations are encoded separately, so document computation can still happen offline. At scoring time, each query token looks for its best-matching document token, and those best matches are combined. This is called late interaction. [1]

**Read MaxSim with a small example.** A query contains “red” and “bicycle.” In a document, the best similarity for “red” might be 0.9 and for “bicycle” 0.8, producing a summed score of 1.7 under the simple formulation. Another document may match “red” well but have no good bicycle match, lowering its total. The score retains more token-level evidence than one document vector.

**Why this is cheaper than full cross-encoding.** Document token vectors are reusable across queries, and the interaction is a structured similarity calculation rather than rerunning all Transformer layers jointly for every pair. However, storing many token vectors costs more memory than one-vector-per-document retrieval, and efficient candidate generation/compression is important.

MaxSim also has limitations: individual query-token matches can be found in different places, and the scorer does not perform unrestricted joint reasoning over every query–document token interaction. Negation, compositional constraints, and document length can still be challenging.

“Near cross-encoder accuracy” is an empirical result under particular datasets and models, not a theorem. Compare quality, index size, build/update cost, and query latency in the intended setting. Late interaction occupies a useful middle ground, but its operational costs differ materially from a conventional bi-encoder index.
@@ 729
**Matryoshka training makes the beginning of an embedding useful on its own.** Ordinary embedding coordinates are learned jointly; simply cutting off the last three quarters can destroy important information. MRL trains several nested prefixes of the representation to solve the task, so shorter prefixes become meaningful lower-cost embeddings. [1]

**A nested-doll example.** A model outputs 768 dimensions, but training also evaluates prefixes such as 64, 128, and 256. A memory-constrained index may store 128 dimensions; a later reranker may use a longer representation. The smaller vector is trained to retain broadly useful information, while additional coordinates can refine discrimination.

This is not PCA applied after training, nor a guarantee that every arbitrary prefix length has identical quality. The selected training dimensions, objective weights, normalization, and data determine the quality–size curve. Follow the model's recommended normalization after truncation where applicable.

**Why it helps a retrieval system.** Short vectors reduce storage, transfer, and distance-computation cost. A first pass can search a compact representation and refine a shortlist using more dimensions. But the initial pass can still miss candidates, so measure its recall before celebrating reranking quality.

Compare actual bytes including index overhead, not just vector dimension. For graph indexes, edges can remain a substantial memory term even after vectors shrink. Test multiple dimensions on representative queries and report the tradeoff explicitly; MRL makes truncation intentional and trained, not free of information loss.
@@ 730
**A RAG answer can fail at three different relationships.** Context relevance asks whether retrieved material helps answer the question. Groundedness asks whether the response's factual claims are supported by that material. Answer relevance asks whether the response actually addresses what the user requested. [1]

**Use one question to separate them.** The user asks, “Why did revenue fall in Q3?” Retrieving a paragraph about Q1 hiring is poor context relevance. Retrieving the correct Q3 report but inventing a price increase is poor groundedness. Accurately summarizing the report's office locations is grounded but not answer-relevant.

This separation makes debugging actionable. Improve retrieval when evidence is missing; improve evidence use or abstention when claims are unsupported; improve task interpretation when the response answers a different question. One aggregate score cannot tell you which repair is needed.

**Avoid rewarding empty caution.** An answer saying “I cannot determine anything” may contain no unsupported claims and thus look perfectly faithful, while failing the task. Pair groundedness with answer completeness and correctness. Likewise, keyword overlap does not establish that a passage contains the needed evidence.

Use labeled examples with answer-bearing passages, claim-level support checks, and human calibration for automated judges. Evaluate citations separately: a true statement with an unrelated citation is still a citation failure. Track slices such as multi-document synthesis, tables, dates, and access-filtered requests, because aggregate averages can hide the hardest evidence paths.
@@ 731
**An LLM judge is another learned model with preferences and blind spots.** It can make evaluation cheaper, but its score is not an objective ground truth merely because it is written confidently. Position, style, length, familiarity, and prompt wording can influence its decisions. [1]

**Three concrete biases.** Position bias favors the first or second answer in a pair. Verbosity bias rewards a longer answer even when extra text adds no value. Self-preference can favor outputs resembling the judge's own model/style. A fluent wrong explanation may also beat a terse correct one unless the rubric demands verifiable correctness.

**Calibrate rather than assume.** Assemble representative examples labeled by qualified humans, including clear wins, close comparisons, factual traps, and different writing styles. Compare judge decisions with those labels and inspect disagreement categories. Randomize or swap answer order, hide model identities, and use explicit task-specific criteria.

Agreement is not sufficient if both humans and the judge share a blind spot, so include executable checks or authoritative references where possible. Multiple judges can reduce some idiosyncrasies but may have correlated errors. Report uncertainty instead of forcing every close case into a confident winner.

Keep the evaluation set separate from repeated optimization. If a model is tuned specifically to maximize one judge, it may learn the judge's tastes or loopholes rather than improve user outcomes. Use periodic human audits and new held-out cases to detect this evaluator overfitting.
@@ 732
**Diffusion learns to reverse a controlled corruption process.** During training, start with a real image and add Gaussian noise at a selected level. The model sees the noisy image and a timestep indicating how much corruption was applied. It learns information needed to move back toward the data distribution. [1]

**Why several noise levels matter.** At light noise, visible edges and details remain; at heavy noise, little recognizable structure survives. The model must learn both fine cleanup and broader distributional structure. A known forward formula lets training create a noisy sample at any timestep directly, without simulating every earlier corruption step.

**Predicting noise is a convenient parameterization.** Because training sampled the noise, it knows an exact target epsilon. Given the noisy image, noise level, and a noise estimate, one can derive an estimate of the clean image or the reverse-step mean. Predicting noise, clean data, or another related parameterization can be mathematically connected but produce different loss weighting and optimization behavior.

It is therefore too strong to say models must predict noise because predicting the clean image is impossible. Different diffusion systems use different targets. The chosen parameterization and schedule determine which errors receive emphasis.

At generation time, begin from noise and apply learned reverse steps to produce a sample. The network is not recovering one hidden original photograph from that random noise; it is generating a new image according to the learned distribution and any conditioning. Evaluate both quality and diversity, not just training noise-prediction error.
@@ 733
**The training noise levels do not force one unique sampling trajectory.** DDIM constructs a different reverse process compatible with the diffusion training setup, allowing larger timestep jumps and, under a particular setting, deterministic updates. This can reduce the number of network evaluations needed to generate an image. [1]

**What deterministic means here.** Fix the initial noise, model, conditioning, and deterministic sampler settings; the subsequent path is determined rather than adding fresh random noise at every reverse step. Different initial noise still produces different samples. Deterministic sampling is not the same as producing one universal image for every prompt.

A DDIM step uses the current noisy state and the model's estimate to infer a clean-image component and construct the state at an earlier selected noise level. Skipping many intermediate levels saves computation, but larger jumps can make approximation error more visible.

**Why “1000 to 20 steps” is not a universal guarantee.** Sample quality depends on the model, training schedule, selected timesteps, guidance, and task. A model trained with 1000 noise levels may support good sampling with far fewer evaluations, but the acceptable speed–quality tradeoff must be measured.

Compare wall-clock time, number of function evaluations, perceptual quality, prompt adherence, and diversity under the same conditions. Modern solvers and distillation methods offer additional routes to fast sampling; DDIM's central lesson is that a learned denoising model can be paired with different inference dynamics rather than blindly replaying every training timestep.
@@ 734
**Pixel-space diffusion spends effort on a very large array.** A high-resolution RGB image contains millions of scalar values. A learned autoencoder can compress it into a smaller latent representation that retains useful visual structure. Latent diffusion runs its iterative generation process there, then decodes the final latent back to pixels. [1]

**Count the reduction carefully.** If height and width each shrink by a factor of eight, spatial positions shrink by 64. Latent channel count may differ from three, so total scalar reduction is not automatically exactly 64. The denoising network's architecture also affects compute savings.

**Two models have different jobs.** The autoencoder learns how to compress/reconstruct images. The diffusion model learns a distribution over those compressed representations. At generation, sample a latent through denoising, then decode once. Text or other conditions guide the latent generation process.

**Compression is a quality bottleneck too.** If the autoencoder loses small text or fine texture, a perfect latent generative model cannot trivially recover every detail that representation discarded. Reconstruction quality should therefore be evaluated separately from diffusion sampling quality. Lower compute comes with a representational tradeoff, not a lossless guarantee.

Latent spaces are learned numerical representations, not necessarily human-interpretable factors such as “one coordinate for color.” Evaluate generated images on the actual use case, including fine detail, consistency, and artifacts. The method's efficiency comes from moving repeated work into a smaller space while retaining a decoder capable of mapping back to a useful image.
@@ 735
**Classifier-free guidance compares “with the prompt” to “without the prompt.”** A model is trained to make both conditional and unconditional predictions, commonly by dropping conditioning on some training examples. At sampling time, the difference between those predictions indicates how the condition changes the denoising direction. [1]

**Read the scale.** In $\hat\epsilon=\epsilon_{uncond}+w(\epsilon_{cond}-\epsilon_{uncond})$, w=0 uses the unconditional prediction and w=1 uses the ordinary conditional prediction. Values above one extrapolate beyond it, pushing more strongly in the condition-dependent direction. This is not merely averaging two images.

**A conceptual example.** For “a red bicycle,” unconditional generation may consider many colors and objects. Guidance emphasizes the denoising changes associated with that phrase. Increasing guidance can strengthen recognizable prompt attributes, but excessive strength can produce oversaturation, artifacts, reduced variety, or unnatural combinations.

The exact tradeoff depends on the model and sampler; a larger number is not always better adherence in a meaningful sense. Strongly rendering one noun while breaking spatial relationships is not a successful interpretation of the whole prompt.

Compute cost can also increase because conditional and unconditional predictions both need evaluation, though batching or specialized methods can change overhead. Compare prompt adherence, diversity, visual quality, and latency across scales. Guidance modifies the sampling distribution; it is an intentional control, not an exact no-cost acceleration of the original conditional model.
@@ 736
**Backbone architecture and generative objective are separate choices.** A U-Net or Transformer specifies how the neural network processes spatial/noisy representations. Diffusion, rectified flow, and flow matching specify related but distinct ways to learn and integrate a generative transformation. A flow objective does not itself require a Transformer, and a Transformer does not automatically imply flow matching. [1] [2]

**What DiT changes.** Diffusion Transformers tokenize image or latent patches and use Transformer blocks to predict the quantity required by the diffusion model, with timestep and conditioning information supplied appropriately. This replaces a traditional convolution-heavy U-Net backbone while retaining the surrounding generative framework.

**A flow intuition.** Imagine particles moving from a simple noise distribution toward the data distribution. A velocity field tells each particle which direction and speed to move at each time. Flow-matching methods train a network to approximate a suitable field; sampling integrates that field from noise to a generated result. The chosen path and numerical solver matter.

Rectified-flow-style formulations aim for useful transport paths, often involving straight interpolation in training constructions, but learned trajectories and finite-step numerical integration still have approximation error. Do not infer that “flow” means one exact jump from noise to a perfect image.

Compare architecture scaling, training objective, conditioning, and solver separately. Report quality against actual network evaluations and wall-clock time. A change attributed to “DiT versus diffusion” may really combine several changes at once, making it impossible to identify which one produced the improvement.
''')
