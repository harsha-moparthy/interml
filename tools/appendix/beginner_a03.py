from expand import write
write(r'''
@@ 685
**The problem is not the size of one coordinate, but the sum of many coordinates.** Attention compares a query and key by adding coordinate-wise products. Even if every coordinate has a sensible scale, a wider head sums more random contributions and therefore produces a wider range of scores. Softmax interprets those larger differences as stronger confidence. [1]

**Derive the variance carefully.** Write $s=\sum_{r=1}^{d_k}q_rk_r$. Under the stated independence and standard-normal assumptions, each product has mean zero. Its second moment is $\mathbb E[q_r^2]\mathbb E[k_r^2]=1$, so its variance is one. Different coordinate products are uncorrelated, and variances add: $\operatorname{Var}(s)=d_k$. Standard deviation is therefore $\sqrt{d_k}$, not $d_k$.

Dividing the score by $\sqrt{d_k}$ divides its variance by $d_k$, returning variance one. For head width 64, the unscaled score has standard deviation eight. The scaling is a normalization of typical score variation; it is not averaging the coordinates.

**See the effect on softmax.** Two logits $(0,1)$ give probabilities approximately $(0.269,0.731)$. Logits $(0,8)$ give approximately $(0.000335,0.999665)$. In the second case, changing the large score a little barely changes the already-saturated probability. The softmax Jacobian contains factors such as $p(1-p)$, which become small near zero or one.

This does not prove every attention gradient vanishes, nor does it imply the combined cross-entropy gradient is small for a confidently wrong classification. It explains why unscaled random attention scores can begin unnecessarily saturated. Trained features need not remain independent standard normals, but the scaling remains a useful architectural convention.
@@ 686
**A residual connection gives information a direct route through a block.** For a sublayer F, the expression $x+F(x)$ lets the block learn a correction instead of reconstructing the entire representation. Backward also receives a direct contribution through x. Normalization placement determines whether that direct route itself passes through a normalization operation. [1]

**Compare the two computation orders.** Pre-norm first normalizes the input to the sublayer, then adds the sublayer result to the original x. Post-norm adds the sublayer result first, then normalizes the entire sum. Both can produce the same-shaped tensor, but their Jacobians differ.

For pre-norm, differentiation produces an explicit identity term plus the sublayer path. For post-norm, the normalization Jacobian multiplies both the identity and sublayer paths. Across many blocks, these differences affect gradient scales at initialization and how sensitive training is to warmup and residual scaling.

**An intuition with a nearly-zero sublayer.** If F initially contributes almost nothing, a pre-norm block behaves approximately like the identity map. A post-norm block still normalizes x, so it is not approximately identity merely because F is small. This helps explain the simpler direct gradient route in pre-norm.

**Do not turn the intuition into an absolute guarantee.** Products of many nontrivial Jacobians can still amplify or suppress gradients. Pre-norm and RMSNorm are separate decisions: one specifies placement, the other specifies the normalization formula. Also, named model families may use additional norms or modified residual arrangements. Check the actual architecture instead of assuming the simplest pre-norm diagram describes every modern LLM.
@@ 687
**LayerNorm does two jobs; RMSNorm keeps one of them.** LayerNorm subtracts a token's feature mean and controls its centered spread. RMSNorm controls overall magnitude without subtracting the mean. The motivation is that magnitude control can be valuable even when recentering is unnecessary for a particular architecture. [1]

**Compare the same vector.** For $(3,4)$, RMS is about 3.536, so RMS normalization gives approximately $(0.849,1.131)$ before learned scales. LayerNorm centers first to $(-0.5,0.5)$ and then rescales. The former retains the original direction; the latter removes the shared offset.

If every feature receives an added constant, LayerNorm cancels it through centering, while RMSNorm generally changes. Thus they are not mathematically equivalent, and “works just as well” must refer to measured task quality under a training recipe, not identical outputs.

**Where speed can come from.** Dropping mean subtraction reduces arithmetic and reductions, and a suitable fused kernel can reduce memory traffic. But end-to-end speed depends on how much runtime normalization occupied in the first place. If normalization is only a small fraction of a model's time, even a large improvement to that operation produces a modest overall gain.

The 10–15% figure in the question is therefore not a universal constant. Measure representative tensor shapes, precision, hardware, and compiled kernels, then compare validation quality. Include epsilon and accumulation dtype in correctness checks: the cheap operation is only useful if its numerical behavior matches the intended model.
@@ 688
**Imagine rotating two arrows together.** Their absolute orientation changes, but their angle difference does not. RoPE uses this property to make the positional contribution to attention depend on the separation between tokens rather than on two unrelated absolute-position vectors. [1]

Split a query/key head into coordinate pairs. For one pair with frequency theta, rotate the query at position m by $m\theta$ and the key at position n by $n\theta$. Different pairs use different frequencies, providing several positional scales.

**The algebra mirrors the picture.** A rotation matrix is orthogonal, so its transpose reverses the angle. In the dot product, the query rotation becomes a transpose and combines with the key rotation: $R(m\theta)^TR(n\theta)=R((n-m)\theta)$. The shared absolute rotation cancels, leaving the relative angle. Depending on which vector is transposed and the rotation convention, the offset may be written with the opposite sign; consistency matters more than that naming choice.

**A quarter-turn example.** Original vectors $(1,0)$ at positions zero and one become perpendicular for theta equal to pi/2, giving dot product zero. Moving both positions forward by one keeps them perpendicular. Moving only one changes their relative angle and therefore their score.

**What “depends only on relative distance” does not mean.** The complete dot product still depends on the content vectors. Two token pairs at the same distance can have very different scores. Nor does the rotation identity guarantee unlimited-context reasoning: unfamiliar phase combinations and learned content dynamics can still fail outside training lengths. Separate the exact positional algebra from empirical extrapolation claims.
@@ 689
**A longer context asks the model to interpret positions it did not practice.** If training used 8k tokens, directly feeding 128k positions changes the range of positional patterns, even when memory is sufficient. Context extension methods modify those patterns so the new range is more compatible with learned behavior. [1]

**Position interpolation is like compressing a ruler.** To fit 128k positions into the positional range previously used for 8k, divide position indices by 16. The last new token then uses an angle comparable to the old last token. But adjacent positions also become sixteen times closer in that encoding, potentially making fine local distinctions harder.

**Why frequency-aware adjustments help.** RoPE contains several frequency bands. Fast-changing bands and slow-changing bands serve different positional scales, so uniformly shrinking every angle is not always the best compromise. NTK-aware scaling changes the frequency base, while YaRN blends different treatments across bands and includes attention-scaling adjustments. The detailed recipe matters; these names are not interchangeable switches.

Some continued training on longer examples can teach the model to use the new positional regime. “No full retraining” usually means avoiding training the entire foundation model again from scratch, not proving that no adaptation or quality loss is possible.

**Evaluate three separate outcomes.** Does the sequence fit in memory? Can the model retrieve facts from different positions? Can it combine those facts to answer a real task? A successful needle lookup answers only part of that evaluation. Also check short-context regression, prefill/decode latency, and the cost of the larger KV cache; positional changes do not make stored context free.
@@ 690
**All four methods change how attention information is represented or shared.** MHA gives every query head its own keys and values. MQA shares one KV head across queries. GQA shares KV heads within groups. MLA uses a compressed latent representation and compatible projections rather than merely reducing the count of otherwise identical KV heads. [1] [2]

**Why this matters during generation.** At every new token, a layer must access its past attention information. Long contexts and many simultaneous requests make that cache consume substantial memory and bandwidth. Reducing stored KV information can allow more requests or longer contexts and can reduce the bytes read per decode step.

**A head-count example.** If 32 query heads use 32 KV heads in MHA, switching to eight KV heads in GQA reduces the basic KV payload by four at fixed head width and precision. MQA with one KV head reduces it by 32. The query heads still compute distinct attention distributions; sharing keys does not mean every head produces the same answer.

**MLA requires a different accounting exercise.** Count the cached latent and any separate positional components, then inspect how computation reconstructs or absorbs projections. Calling it “another one-KV-head method” misses the factorization that determines its memory and compute costs.

Quality can change because sharing or compression restricts the representation. These methods primarily target inference memory capacity/bandwidth, not the serial dependence between generated tokens. Compare actual cache bytes, achievable throughput, latency, and task quality rather than assuming a parameter-count change predicts the serving improvement.
@@ 691
**A bias adds a learned offset after a linear transformation.** In $y=Wx+b$, W determines how input features combine and b shifts the output independently of the current input. Removing b simplifies the layer, but does not make it mathematically equivalent to every biased layer.

**Count the actual saving.** A square width-4096 projection has over 16 million weights and only 4096 bias values. Removing the bias saves a small fraction of parameters. The more relevant implementation benefits may be fewer operations, simpler fusion, or a consistent architecture, rather than a dramatic reduction in total model size. [1]

**Normalization changes the role of some offsets.** LayerNorm removes a shared mean component, so certain shifts can be redundant in particular arrangements. RMSNorm does not subtract the mean and does not generally cancel arbitrary biases. A bias vector with different values in different coordinates is also not the same as one common scalar offset.

Successful bias-free architectures show that these offsets are not always necessary to obtain strong performance under their full training recipes. That is an empirical design result, not a proof that biases are useless. Other architectures retain biases or introduce learned offsets for specific purposes.

When comparing models, inspect linear biases, normalization shifts, residual placement, and output-head choices separately. For implementation, removing a checkpoint's bias without retraining can change outputs even if another model family was trained successfully without biases from the beginning. Architectural simplification and post-hoc parameter deletion are different experiments.
@@ 692
**SwiGLU lets one learned feature stream modulate another.** A standard FFN expands a token, applies a nonlinearity, then contracts it. SwiGLU creates two expansions: a value stream and a gate stream passed through SiLU. Their elementwise product returns through an output projection. This introduces input-dependent feature interactions. [1]

**Follow the three matrices.** The value matrix maps model width d to hidden width h. The gate matrix independently maps d to h. The down-projection maps h back to d. That gives three weight matrices with approximately dh parameters each, or $3dh$ total, ignoring biases.

A conventional FFN with width 4d uses two matrices and therefore $2d(4d)=8d^2$ weights. To match that budget, solve $3dh=8d^2$, giving $h=8d/3$. The “two-thirds” in the question refers to two-thirds of the conventional hidden width 4d, not two-thirds of model width d.

**Why call it a gate?** If a value feature is large but its activated gate is near zero, the product suppresses it. If the gate is large, it can pass or amplify the feature. SiLU is not bounded to [0,1], so this is more general than a probabilistic on/off gate.

Actual models round h for hardware alignment and may use different parameter budgets. Gating can improve expressivity, but an extra branch is not a mathematical guarantee of higher accuracy. Compare under matched training compute and data, and check matrix ordering carefully when implementing a fused gate/value projection.
@@ 693
**Soft-capping limits extreme logits smoothly.** A logit is a score before softmax. If a few scores become enormous, attention can concentrate almost entirely on one position, and numerical/optimization behavior may become difficult. Soft-capping compresses scores toward a finite range without a hard discontinuous cutoff. [1]

A common form is $\tilde z=c\tanh(z/c)$. Near zero, tanh behaves approximately like its input, so ordinary scores are nearly unchanged. For very large positive or negative inputs, the result approaches +c or -c. With c=10, a raw score of 100 no longer enters softmax as 100; it is close to 10.

**Why this differs from clipping.** Hard clipping abruptly makes every value above the threshold identical and has a kink at the boundary. Tanh changes smoothly, though its derivative also becomes small in the saturated region. The cap controls the range of differences softmax sees, not just the largest positive value.

**Do not overstate the protection.** Bounding logits can reduce extreme score behavior, but it does not prove the entire model is numerically stable. Upstream activations, optimizer updates, precision, and other layers can still fail. The cap also changes the model's representational behavior, so it is not a free numerical rewrite like subtracting a common softmax maximum.

Check whether a model applies soft-capping to attention logits, output vocabulary logits, or both, and preserve the specified cap values when loading checkpoints. Measure quality and saturation statistics alongside training stability rather than assuming one cap transfers unchanged to every architecture.
@@ 694
**Full attention stores access to individual past tokens; a recurrent state compresses the past.** In full attention, the current query can compare with each earlier key. A state-space model instead updates a fixed-size state as tokens arrive. Linear-attention constructions can also accumulate suitable summaries so a new query uses stored statistics rather than revisiting a full token-by-token score matrix. [1] [2]

**A running-summary analogy.** To compute a running average, you need only a count and sum, not every previous number. A recurrent model learns a much richer summary, but the basic tradeoff remains: compact state saves memory while potentially discarding details needed by future questions.

For fixed state width, each new token updates a bounded amount of state, making recurrent-state memory independent of sequence length. Training can exploit scans or structured/chunked matrix operations, giving linear scaling in length under the architecture's fixed-size assumptions. Constants and parallel efficiency still matter.

**Why exact retrieval can be hard.** Suppose a long context contains thousands of arbitrary key–value pairs, and only at the end do you learn which key will be queried. A compact state must preserve enough information about many possible answers in advance. Full attention retains explicit token-level access, giving it a different capability–memory tradeoff.

“O(1) memory per step” refers to recurrent sequence state, not all model weights, training activations, or arbitrary output storage. Hybrid models add attention layers to recover some explicit retrieval ability. Evaluate the actual tasks and lengths; asymptotic efficiency alone does not determine which architecture gives the best quality at a fixed serving budget.
@@ 695
**Sparse experts create a scheduling problem as well as a modeling opportunity.** A router sends each token to a few experts. If all experts were equally popular, their workloads would balance. In practice, a router can prefer a small subset, creating overloaded experts and idle ones. [1]

**Capacity is a queue limit.** Suppose an expert is allocated room for eight token assignments in a step but receives twelve. A capacity-limited implementation must drop, reroute, or defer four assignments according to its policy. “Dropped” often means skipping that expert computation while preserving another path such as the residual; the exact behavior depends on the implementation. Dropless designs make a different memory/communication tradeoff.

**Why collapse can reinforce itself.** Experts receiving more tokens get more training opportunities. If early routing makes one expert useful, the router may send it even more traffic while underused experts improve slowly. A plain task-loss-trained softmax router does not automatically value balanced utilization.

Balancing losses, capacity policies, routing noise, or explicit bias adjustments can encourage broader expert use. But forcing perfect uniformity may also interfere with useful specialization, so balance and task quality must be evaluated together.

**Measure the real consequences.** Track assignments, dropped tokens, expert output contribution, device load, and task quality by slice. A globally balanced count can still hide one overloaded device if several popular experts share it. Do not equate “more total parameters” with “more computation per token”: sparse routing is specifically intended to separate those two quantities.
@@ 696
**Softmax must place its probability mass somewhere.** Some attention heads learn to allocate substantial weight to early tokens even when those tokens carry little obvious semantic information for the current query. These positions can act as attention sinks: stable places for excess or background attention mass rather than informative content retrieval. [1]

**Why a naive rolling cache can break behavior.** If training and early inference repeatedly included those initial positions, later attention patterns may rely on their presence. Evicting the oldest tokens removes the sinks along with old content, changing normalization and behavior even when the remaining recent context seems sufficient.

StreamingLLM retains a small set of initial sink positions together with a moving window of recent tokens. The cache stays bounded while preserving those special positions. This can stabilize streaming generation far beyond the original cache length without retaining every old token.

**Understand what is not preserved.** The model does not gain lossless memory of the entire conversation. Facts in evicted middle tokens are no longer directly available through that cache. “Infinite-length generation” means continuing the stream with bounded state, not accurately answering arbitrary questions about every event from an unlimited history.

Evaluate generation stability separately from long-term factual recall. The sink phenomenon also does not mean initial tokens are always meaningless: a system instruction at the beginning may have genuine semantic importance. Position-based sink behavior and content-based instruction following are distinct effects that can coexist.
@@ 697
**The attention mask determines what a token can see.** Encoder-only models typically let tokens attend bidirectionally across the input, making them effective at building a representation of a complete text. Decoder-only models use causal attention, predicting the next token from the preceding context. Encoder–decoder models encode an input bidirectionally and generate an output through a causal decoder that can also attend to the encoder. [1] [2]

**Match architecture to task.** For classifying a finished review or embedding a document, having access to words on both sides can help. For open-ended generation, next-token prediction naturally supports producing one token after another. For translation or summarization, an encoder–decoder explicitly separates source understanding from target generation.

**Why decoder-only became prominent.** One next-token objective works across text, code, instructions, and many serialized tasks, and large-scale training/serving ecosystems developed around it. This is a practical convergence of data, objectives, scaling, and infrastructure, not proof that bidirectional encoding is obsolete.

For retrieval, a compact encoder can precompute item/document vectors and create a query vector in one forward pass. Generation-oriented models can also be adapted for embeddings, but their existence does not remove the latency and training advantages of a purpose-built embedding model.

Compare quality, parameter count, input/output lengths, and latency for the actual workload. A family name does not uniquely determine behavior: pooling, training objective, masking, and adaptation can matter as much as whether the backbone was originally called an encoder or decoder.
@@ 698
**A ViT turns an image into a sequence of patches.** Divide an image into fixed-size nonoverlapping regions, flatten each patch's pixels, and project that vector into an embedding. Add positional information so the model can distinguish where patches came from, then process the sequence with Transformer blocks. [1]

**Count an example.** A 224-by-224 image with 16-by-16 patches produces $14\times14=196$ patch tokens. With RGB input, each flattened patch initially has $16\times16\times3=768$ pixel values. A learned projection maps those values to the chosen model width; some designs add a separate class token for classification.

**Compare the built-in assumptions.** A CNN applies local filters with shared weights, naturally emphasizing locality and a form of translation equivariance. A basic ViT can mix distant patches through attention and has weaker built-in spatial assumptions, so data, augmentation, pretraining, and positional design play a large role. This is a difference in inductive bias, not a claim that CNNs cannot model global information.

Smaller patches preserve finer detail but increase token count and full-attention cost. Larger patches reduce sequence length but can obscure small objects or text. Resolution changes also change the patch grid and may require positional-embedding handling.

Evaluate the relevant visual task rather than comparing architecture labels alone. A classifier can tolerate some lost fine detail that OCR or small-defect inspection cannot. The patch embedding is learned; it is not merely an average color assigned to each square.
@@ 699
**The vision encoder and language model do not automatically speak the same vector language.** A vision encoder produces features useful for images, while an LLM expects embeddings in the representation space it was trained to process. A connector translates or exposes visual information in a form the language model can use. [1] [2] [3]

**Three common bridges.** A linear or small MLP projection maps visual patch features to the LLM's input width, after which they can be inserted as visual tokens. A Q-Former uses learned queries to extract a smaller set of useful features from image representations. Cross-attention lets language states query visual features through dedicated attention modules rather than only concatenating projected tokens.

**A concrete design choice.** A high-resolution image might produce hundreds or thousands of patch features. Passing all of them preserves detail but increases context and compute. Compressing them through learned queries reduces cost but may discard small text or spatial detail. The right choice depends on whether the task is broad captioning, OCR, counting, or detailed grounding.

Training often first aligns the connector while freezing pretrained components, then adapts selected parts on multimodal instructions. This is a practical recipe, not a rule that every system must freeze the same modules.

Test whether the answer actually depends on the image: swap images while holding the question fixed, inspect grounding, and evaluate visually ambiguous cases. Fluent answers based on language priors can look successful even when the connector transmits little useful visual evidence.
@@ 700
**Count cached scalars, not model parameters.** The 70B parameter label does not directly determine this cache size. KV storage depends on layers, active sequences, sequence length, KV-head count, head width, and bytes per stored element. Every layer stores both keys and values for every cached token. [1]

**Build the count one factor at a time.** One token in one layer stores $2\times8\times128=2048$ scalars. FP16 uses two bytes per scalar, so that is 4096 bytes. Across 80 layers, one token occupies 327,680 bytes. Across 16,384 tokens, one sequence occupies 5,368,709,120 bytes, exactly 5 GiB.

For 32 such sequences, multiply by 32: 171,798,691,840 bytes. That is 160 GiB using powers of 1024, or approximately 171.8 decimal GB. State the unit explicitly; saying “160 GB” without qualification mixes two conventions.

**Why this surprises people.** A model's weights are shared among requests on a replica, but each request generally needs its own changing KV state, apart from deliberately shared prefixes. Increasing batch size therefore increases cache memory directly. A long-context serving workload can be cache-limited even when the weights fit comfortably.

This is the exact tensor payload under the stated assumptions, not the exact total GPU allocation. Add weights, temporary workspaces, allocator/block overhead, unused reserved capacity, and any extra architecture-specific state. GQA has already reduced KV heads to eight here; substituting query-head count would overestimate this model's basic cache.
@@ 701
**Prefill reuses weights across many tokens at once.** Processing a long prompt creates large matrix operations: many token vectors multiply the same weight matrices. That reuse raises arithmetic intensity, allowing the GPU to spend more time doing useful math per byte fetched. Decode often processes only one new token per active request, with less reuse and repeated reads of weights and growing KV state. [1]

**Use a factory analogy.** Prefill runs a large batch through the same machinery before changing setup. Single-request decode repeatedly brings the machinery's information into use for one new item, then waits for that result before producing the next. The serial dependency is intrinsic to autoregressive generation, while batching several users can partly restore reuse.

The roofline compares compute demand with data movement. A workload with many operations per byte can approach the compute ceiling; one with few operations per byte hits bandwidth first. FlashAttention, quantization, GQA, and batching change different parts of that balance.

**The statement in the question is a common regime, not a law.** Very short prefills may be launch- or memory-limited. Large decode batches can become more compute-intensive. Very long contexts can make attention-cache reads dominate, and CPU scheduling or networking can become the bottleneck before the GPU.

Measure representative batch sizes and lengths with traces and achieved bandwidth/FLOPs. Distinguish time to first token from time per output token. An optimization that improves large-prompt prefill may leave single-user decode unchanged, because it solved a bottleneck that was not limiting decode.
@@ 702
**The expensive intermediate is often the attention matrix, not just the arithmetic.** Standard attention can write an entire query-by-key score matrix to high-bandwidth memory, read it for softmax, and write/read probabilities again before multiplying by values. For long sequences, those quadratic intermediates create substantial memory traffic. [1]

**FlashAttention changes the execution plan.** Load blocks of queries, keys, and values into fast on-chip memory. Compute local scores and update a running softmax normalization and weighted-value accumulator. Earlier contributions are rescaled when the running maximum changes, so the final output matches dense attention without storing the whole probability matrix.

**Why backward can recompute.** Saving an intermediate trades memory for avoiding future arithmetic. FlashAttention keeps compact needed statistics and outputs, then regenerates local scores/probabilities during backward. Additional computation can be cheaper than repeated transfers to slow memory. “Exact attention” describes the same mathematical operation, with ordinary floating-point ordering differences—not a low-rank approximation.

FlashAttention-2 improves parallel work partitioning and reduces non-matmul overhead in its target settings. FlashAttention-3 adds hardware-specific scheduling/overlap and precision-oriented techniques for supported architectures. The version names refer to algorithms and kernels with particular hardware assumptions, not universally increasing speed on every device. [2] [3]

Benchmark the complete layer at actual shapes and masks. A kernel unavailable on the target GPU, unsupported dtype, or different attention pattern may force another path. Avoid claiming that removing quadratic storage also removes quadratic full-attention arithmetic; those are separate complexity statements.
@@ 703
**A contiguous cache wastes space when request lengths are unpredictable.** Reserving a large maximum-length buffer for every sequence leaves unused capacity inside short requests. Repeated allocation and freeing of differently sized buffers can also leave gaps that are hard to reuse for a large contiguous request.

**Paging separates logical order from physical placement.** Divide KV storage into fixed-size token blocks. A request's block table records which physical blocks hold its first, second, and later logical segments. The blocks need not be adjacent in memory; the attention kernel follows the mapping. [1]

**A small example.** With four-token blocks, a nine-token sequence needs three blocks. Its first eight tokens fill two blocks, while the last block holds one token and leaves three slots unused. This bounds tail waste at block granularity rather than reserving the sequence's entire possible future length. It also shows why “eliminates all internal fragmentation” is too strong: the final partial block can still waste space.

Reference counting and copy-on-write can let compatible requests share prefix blocks until their continuations differ. Correct identity must include the relevant model/context conditions; sharing memory for merely similar text would be wrong.

Paging improves allocation flexibility and serving capacity but introduces metadata and kernel complexity. It does not reduce the number of KV values logically needed for a given unshared history. Evaluate memory utilization, throughput, cache sharing, and correctness under request cancellation and eviction, not just whether individual blocks allocate successfully.
@@ 704
**Requests do not finish together.** In a static batch, several prompts begin generation as one group. One request may need five output tokens and another 500. If the batch cannot admit replacements until the longest request finishes, completed slots remain idle or padded while useful work waits in the queue.

**Continuous batching reschedules at token iterations.** After an iteration, remove completed or canceled requests and admit new work into available capacity. The batch membership changes while surviving requests retain their own state. This uses the GPU more continuously under variable-length traffic. [1]

**A concrete timeline.** Four requests start together. After ten steps, two finish. Static batching may continue with only two productive requests until both finish; continuous batching can fill the freed capacity with new requests at the next suitable scheduling boundary. The benefit comes from avoiding unused capacity, not from making any one token mathematically independent of its prefix.

Prefill work for new requests can be much heavier than one decode step for existing requests. A good scheduler must balance throughput with time-to-first-token and inter-token latency, sometimes chunking prefills or reserving decode capacity. Unlimited admission can exhaust KV memory or increase tail latency.

Measure under realistic arrival rates, prompt lengths, and output distributions. A perfectly uniform synthetic batch can hide the advantage, while an overloaded system can still have bad latency despite high GPU utilization. Fairness, cancellation, priorities, and backpressure belong in the design along with batching efficiency.
@@ 705
**The draft model is a proposal distribution, not a replacement target.** A smaller model generates candidate tokens cheaply; the large model verifies them. To preserve target sampling, acceptance must compensate for tokens the draft proposes too often and resample mass it proposes too rarely. [1]

**One calculation proves the mechanism.** A draft proposes token x with probability q(x). Accept it with probability $\min(1,p(x)/q(x))$. The accepted mass at x is therefore $\min(p(x),q(x))$. Rejections occur with the remaining total probability; their replacement distribution is proportional to $\max(p(x)-q(x),0)$.

Adding accepted and replacement mass gives p(x) for every token. For example, if $p=(0.8,0.2)$ and $q=(0.5,0.5)$, accepted mass is $(0.5,0.2)$ and the rejected 0.3 is assigned to the first token, recovering $(0.8,0.2)$.

**Why a block still needs sequential logic.** The target can evaluate several draft positions in parallel, but each probability is conditional on a particular draft prefix. At the first rejection, later rows correspond to a prefix that will not be emitted. Stop verifying that suffix and restart from the corrected prefix. If all proposals pass, a bonus target draw uses the final accepted prefix.

Tokenizer/vocabulary alignment and consistent target transformations are essential. If temperature or top-p changes the intended target distribution, verification must use that distribution. The exact sampling theorem does not guarantee speed: draft overhead, acceptance rate, memory traffic, and production batch size determine whether the method is worthwhile.
@@ 706
**A separate draft model is not the only way to propose future tokens.** Medusa adds prediction heads that propose multiple future-token candidates from a shared backbone. EAGLE-style methods use learned feature-level prediction to construct proposals efficiently. Both try to reduce the serial cost of ordinary target-only decoding, but their architecture and verification choices differ. [1] [2]

**Think of shared preparation versus a second worker.** A standalone draft model maintains its own weights and computation path. Extra heads can reuse information already computed by the target backbone, potentially reducing duplicate work. However, they add training, memory, candidate-tree construction, and verification complexity.

A proposal tree can contain several plausible next-token paths. A specially masked target pass evaluates these candidate paths without allowing one branch to see another branch's tokens. Accepted tokens then extend the real sequence; rejected branches are discarded.

**Keep distributional claims tied to the algorithm.** Producing several candidates at once does not automatically preserve the original target distribution. Exact speculative sampling requires the appropriate proposal probabilities and rejection/residual procedure. Greedy or heuristic acceptance can instead define a different decoding behavior. State which setting is being discussed before promising identical outputs or quality.

Compare acceptance depth, draft/head overhead, target verification cost, and end-to-end latency at realistic batch sizes. Multi-token proposals are most useful when they save more expensive serial work than they introduce. Report performance on the tasks and model versions actually tested, because proposal quality can change substantially under domain shift or model updates.
''')
