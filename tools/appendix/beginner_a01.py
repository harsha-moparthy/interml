from expand import write
write(r'''
@@ 601
**What attention is trying to do.** Read “The animal did not cross the road because it was tired.” To interpret “it,” a model needs information from earlier words, but not equally from every word. Attention computes a different weighted average of the available token representations for each query token. The weights are learned from content rather than fixed by distance.

A query is the current token's description of what it needs; a key is another token's description of what it offers; a value is the information that token contributes if selected. These are vectors produced by learned linear projections, not literal English descriptions. Dotting a query with each key produces compatibility scores. Softmax converts the scores into nonnegative weights summing to one, and the weighted sum of values becomes the output. [3]

**Work one row by hand.** Suppose a query has two visible keys with scaled scores $(0,\log3)$. Exponentiating gives $(1,3)$, so attention weights are $(1/4,3/4)$. If the values are $(2,0)$ and $(0,4)$, the output is $(0.5,3)$. The operation mixes information; it does not usually choose a single token.

**Build it in this order.** Compute the query–key score matrix; divide by the square root of key width; apply visibility masks; run softmax over the key axis; multiply by the values. A padding mask hides artificial filler tokens. A causal mask hides future tokens so training cannot cheat by reading the answer. Mask before softmax: zeroing weights afterward without renormalizing changes their sum and therefore changes the operation.

**The easy-to-miss boundary case.** If a query has no visible keys, ordinary softmax receives only negative infinities and can produce NaNs. This implementation explicitly returns zero for that row. Test it separately from normal masking. Also verify that adding an extra padded token does not change an existing query's output: this directly checks whether padding truly remains invisible.
@@ 602
**Why several attention heads?** One weighted average may be useful for locating a pronoun's referent, another for nearby syntax, and another for a distant instruction. Multi-head attention gives the model several learned spaces in which to compare tokens. Heads are not assigned those human-readable jobs in advance, and a particular head need not have one clean interpretation. [3]

**Track a concrete shape.** Take a batch of two sequences, each five tokens long, with model width 12 and three heads. The input shape is $(2,5,12)$. A fused projection returns $(2,5,36)$ because each token needs 12 query, 12 key, and 12 value features. Split those three groups, then reshape each to $(2,3,5,4)$: two examples, three heads, five positions, and four features per head.

Each head produces a $5\times5$ attention matrix and a five-by-four output. Put the three outputs back together to obtain $(2,5,12)$, then use the output projection to mix information across heads. The key axis is the last axis of each attention score matrix; softmax over the head axis would make heads compete instead of letting each query distribute attention over positions.

**What “fused QKV” actually means.** Three separate linear projections can be placed side by side in one larger weight matrix. One larger matrix multiplication produces exactly their concatenated outputs, provided the weight slices and bias ordering match. Fusion improves execution scheduling; it does not mean queries, keys, and values share identical weights.

**A useful correctness test.** Initialize a three-projection version and a fused version with matching slices, disable dropout, and compare their outputs and gradients. This detects permutation mistakes that a shape-only test misses. A tensor can have the correct dimensions while pairing the wrong head or token with its keys.
@@ 603
**The memory problem behind GQA.** During generation, every layer remembers keys and values for previous tokens. Standard multi-head attention stores a separate pair for every attention head. As the conversation grows, this cache becomes large and expensive to read for every new token. GQA reduces the number of distinct stored key/value heads while retaining multiple query heads. [3]

**Use eight questioners and two reference books.** Think of eight query heads asking different questions, with each group of four consulting the same key/value “book.” The queries still differ, so their attention weights can differ even though they consult shared keys and values. With eight query heads and two KV heads, cache storage for K/V is one quarter of the eight-KV-head version, assuming equal head width and precision.

MQA is the extreme case: all query heads share one KV head. MHA is the other extreme: every query head has its own KV head. GQA lies between them and offers a quality–memory tradeoff rather than a free mathematical equivalence to MHA.

**Implement the grouping explicitly.** Reshape query heads into a KV-head axis and a group axis. For $H_q=8,H_{kv}=2$, each key/value head broadcasts across four query heads. The number of query heads must be divisible by the KV-head count under this simple grouping convention. Keep the head ordering consistent through concatenation and output projection.

**A common misconception in the prompt.** `repeat_interleave` repeats tensor values and ordinarily allocates expanded storage; it is not itself a memory-free broadcast. Broadcasting can avoid storing repeated K/V, but the attention score tensor may still be large. Check the actual kernel and allocated memory. Cache savings, intermediate-score savings, and arithmetic savings are three different measurements.
@@ 604
**Why cache at all?** Suppose the prompt is “The capital of France is” and the model generates “Paris.” On the following step, the representations of earlier tokens in an ordinary causal decoder do not change merely because one more token was appended. Recomputing their keys and values would repeat work already done. KV caching retains those projections for reuse. [3]

**Walk through two steps.** After processing a prompt of length five, each layer has five keys and five values. The next token produces one new query, key, and value. Append its key/value to obtain six stored positions; use its query to attend to the six allowed positions. The following token repeats the operation with seven positions. Previous queries are unnecessary because their outputs have already been computed.

**Separate prefill and decode.** Prefill processes the prompt, often with large parallel matrix operations. Decode usually adds one token per request at a time, repeatedly reading the growing cache. This explains why a system can have fast prompt processing yet slow token generation: their shapes and bottlenecks differ.

**Positions and masks are part of the cache.** A newly appended token must use its absolute position, not restart positional encoding at zero. If you append several tokens together, the first new token must not see later tokens in that same block. Its causal-mask row therefore starts at the existing cache length, rather than behaving like position zero in a new sequence.

**Test the invariant.** In evaluation mode, processing a sequence all at once with causal attention should agree with processing its prefix and then appending the remainder through the cache, within numerical tolerance. Test one-token and multi-token appends. Keep caches per request: accidentally sharing mutable state across users is both a correctness and isolation failure.
@@ 605
**Why position needs to enter attention.** A dot product between content vectors alone does not explicitly distinguish “dog bites person” from the same tokens in a different order. RoPE attaches position to queries and keys by rotating pairs of their coordinates. Rotation changes orientation while preserving length, letting a dot product respond to relative displacement. [3]

**Picture a clock hand.** A two-dimensional vector is an arrow. At position zero, leave it unchanged; at position one, rotate it by one step; at position two, rotate it by two steps. Both the query and key follow the same rule. If both positions move forward by five steps, both arrows rotate together and their angle difference stays unchanged. That is the intuition behind the relative-position identity.

**Calculate a tiny example.** Let both original vectors be $(1,0)$ and use a rotation step of $\pi/2$. At positions 0 and 1 they become $(1,0)$ and $(0,1)$, whose dot product is zero. At positions 1 and 2 they become $(0,1)$ and $(-1,0)$, again giving zero. The absolute positions changed, but their offset remained one. Real models use many smaller frequencies across coordinate pairs, not one quarter-turn frequency for everything.

**Translate the picture into code.** Split an even-width head into coordinate pairs, construct sine/cosine values for each position and frequency, and apply the two-dimensional rotation formula. Broadcast positions over batch and head dimensions. Rotate queries and keys; values are not rotated in the standard construction.

**Know the limit of the claim.** Relative-angle structure does not guarantee good extrapolation to unseen lengths. Learned content vectors and attention behavior still depend on context, and unfamiliar phase patterns can confuse the model. Test the algebraic rotation identity separately from downstream long-context accuracy: one is a mathematical property, the other an empirical capability.
@@ 606
**Give every position a recognizable pattern.** A learned token embedding represents which token appeared. Positional encoding represents where it appeared. The original Transformer adds a vector of sine/cosine values to each token embedding so that equal tokens at different locations do not enter every layer identically. [3]

Imagine several clocks moving at different speeds. A fast clock distinguishes adjacent positions; a slow clock changes gradually over a longer span. Reading all clocks together gives a much richer position pattern than reading only one. Sinusoidal encoding uses this idea with mathematically chosen frequencies.

**A four-feature example.** For width four, position $p$ receives $[\sin p,\cos p,\sin(p/100),\cos(p/100)]$ under the original base-10000 formula. At position zero this is $(0,1,0,1)$. At position one the first pair changes substantially, while the second changes only slightly. Different scales provide both local variation and slower positional structure.

**Vectorize the construction.** Make a column vector of positions and a row vector of inverse frequencies. Their outer product contains every position–frequency angle. Apply sine and cosine, then interleave their columns. No loop over tokens is needed. If model width is odd, the last sine coordinate has no partner cosine column; handle the slice lengths explicitly.

**Why relative shifts can be useful.** The sine and cosine addition rules express the encoding at $p+k$ as a fixed linear transformation of the encoding at $p$ for a fixed offset $k$. This gives attention layers a structure they can learn to exploit. It does not mean the network automatically learns every relative relation.

Unlike RoPE, this method adds position to embeddings rather than rotating query/key pairs. Preserve the absolute position offset when encoding a continuation; resetting to zero would make a late token look like the first token again.
@@ 607
**ALiBi expresses a preference for nearby history.** A causal model can attend to all previous tokens, but nearby tokens are often useful. ALiBi adds a head-specific distance penalty to attention scores. Content still determines relevance; distance acts as an additional preference. [1]

**Read a numerical example.** Suppose three visible keys are at distances 0, 1, and 2 from the query, and their content scores are all zero. With slope 0.5, adjusted scores become $(0,-0.5,-1)$. Softmax gives approximately $(0.506,0.307,0.186)$. Nearby content receives more weight, but distant content remains possible. A sufficiently strong content score can overcome its distance penalty.

Different heads use different slopes. A steep slope strongly favors local context; a shallow slope permits broader context. This is similar to giving several readers different preferences for how far back to look, while allowing every reader to respond to content.

**Do not confuse a preference with a prohibition.** A finite distance penalty does not prevent attention to future tokens. Causality still requires a mask assigning future positions no probability. Apply the bias to allowed query–key score pairs before softmax, and use absolute positions when queries are appended to a cache.

**Why it can help longer inputs.** The distance rule can be evaluated beyond the training positions without extending a learned positional lookup table. However, computability at a new length is not proof of good performance there. Large distances, training distribution, and head slopes still affect how much information remains usable. Compare retrieval and reasoning accuracy at target lengths, not merely whether the forward pass runs.
@@ 608
**Fine-tuning does not always need a full weight update.** A large pretrained matrix already contains useful structure. LoRA freezes that matrix and learns an additional update constrained to a low-rank factorization. Instead of learning every entry of a large change independently, it learns how to combine a small number of update directions. [3]

**Count the savings.** A $4096\times4096$ weight contains 16,777,216 parameters. A rank-eight adapter has an $8\times4096$ matrix and a $4096\times8$ matrix: 65,536 parameters, about 0.39% of the original count. This reduces trainable parameters and optimizer state. It does not remove the frozen base weights or all activation memory.

**Follow the forward path.** For input $x$, calculate the original output $Wx$. In parallel, project through the small matrix A, then back through B, and scale the result by $\alpha/r$. Add this adapter output to the original output. The adapter can change behavior while the original matrix remains fixed.

**Why start B at zero?** A zero adapter output makes the initial adapted model exactly match the pretrained model. A is random so B can receive a meaningful gradient on the first update. At that instant A's gradient through B is zero, but after B changes, A can learn as well. Initializing both A and B to zero would stall this simple factorized update.

**Merge only with a clear convention.** Once training is finished, compute $W'=W+(\alpha/r)BA$ and use it as a normal linear layer. This removes the separate adapter computation at inference, subject to precision and quantization choices. Avoid adding the same update twice, and compare merged versus unmerged outputs before deploying. Low rank is an inductive constraint, not a guarantee that every task can be learned with an arbitrarily tiny rank.
@@ 609
**RMSNorm controls magnitude without recentering.** During a deep network's computation, feature vectors can grow or shrink. RMSNorm rescales each token's feature vector according to its root-mean-square magnitude, then applies learned coordinate scales. It does not mix different tokens or estimate statistics from a minibatch. [3]

RMS means “square every coordinate, average the squares, then take a square root.” Squaring prevents positive and negative features from canceling. For $x=(3,4)$, the RMS is $\sqrt{(9+16)/2}=\sqrt{12.5}\approx3.536$. Ignoring epsilon and using learned scales of one, the output is approximately $(0.849,1.131)$. Its RMS is one, while its coordinate ratio remains 3:4.

**Compare with LayerNorm using the same vector.** LayerNorm first subtracts the mean 3.5, producing $(-0.5,0.5)$, then normalizes the centered values. It therefore changes direction as well as scale. RMSNorm preserves direction before learned scaling. Neither method is simply “the other with a faster implementation”; they compute different functions.

**Why epsilon and precision matter.** For an all-zero input, division by its RMS would divide by zero. Epsilon provides a positive floor inside the square root. In low precision, accumulating many squared values can lose accuracy or overflow earlier than expected, so the reference accumulates the second moment in float32 before converting the result as appropriate.

**Implementation checks.** Normalize the final feature axis, not the batch or sequence axis. Test zero vectors, constant nonzero vectors, and large/small magnitudes. Check the checkpoint's scale convention: some implementations store gamma directly, while others store a parameter whose effective scale is one plus that parameter. Loading one convention into the other silently changes the model.
@@ 610
**A gated feed-forward layer learns what to pass through.** A conventional FFN expands a token into more features, applies a nonlinearity, then contracts it. A GLU-style FFN creates two expanded branches. One carries candidate features; the other controls how strongly each candidate feature contributes. Their elementwise product is then projected back to model width. [3]

**Use a two-coordinate illustration.** Suppose the value branch returns $(2,-3)$ and the activated gate returns $(0.1,0.9)$. Their product is $(0.2,-2.7)$. The first feature is strongly reduced and the second mostly retained. These numbers are illustrative: SiLU and GELU are not restricted to the interval [0,1], so this “gate” can also amplify or change signs. It is not necessarily a probability.

SwiGLU uses SiLU, $z\sigma(z)$, on the gate branch. GeGLU uses GELU. In both cases, the gate depends on the same input token through a separate learned projection, allowing interactions between learned features that a single fixed activation does not express in the same way.

**Follow the dimensions.** For input width D and expanded width h, gate and value projections each map D to h. The output projection maps h back to D. This uses three matrices, compared with two in a conventional FFN. Choosing h near $8D/3$ roughly matches the weight count of a conventional width-$4D$ FFN, ignoring biases and rounding.

**Code and test carefully.** A fused input projection can produce both branches together, but split them in the same order used when storing weights. A swapped gate/value branch usually still has valid shapes yet computes a different function. Compare against separate explicit projections and check gradients through both branches, including negative gate inputs.
@@ 611
**An MoE router chooses which specialists work on each token.** A dense FFN applies one network to every token. A mixture-of-experts layer owns several FFNs, but a router selects only a few for each token. This increases available parameter capacity without executing every expert on every input. The analogy to specialists is useful, but experts learn their roles rather than being given fixed human subject labels. [3]

**A tiny routing example.** Suppose four experts receive probabilities $(0.6,0.2,0.15,0.05)$ and top-k uses k=2. Choose the first two experts, then normalize their selected weights to $(0.75,0.25)$ if that is the implementation's convention. Their weighted outputs are combined into one token representation. The router's full probability vector and the selected mixture weights serve different purposes.

**Why a balancing loss exists.** If almost every token goes to one expert, that expert becomes overloaded while others receive little training. Parallel hardware also sits idle waiting for the busiest device. A balancing term encourages a distribution of assignments across experts rather than rewarding collapse.

The supplied loss combines how many assignments each expert actually received with its average routing probability. The hard assignment count is nondifferentiable, but the probability part still gives the router a gradient. For top-k larger than one, specify whether counts are divided by token count or total token–expert assignments; otherwise a change in k can accidentally rescale the objective.

**Routing correctness is not the whole system.** After choosing experts, tokens must be dispatched, processed, and gathered in their original order. Capacity limits may drop or reroute excess tokens. Test duplicate indices, empty experts, normalized selected weights, and gradient flow. A mathematically balanced router can still be slow if communication or expert workloads differ, so measure actual device load as well as counts.
@@ 612
**Sampling turns model probabilities into a choice.** A language model gives a score to every vocabulary token. Softmax converts those scores to probabilities, but always picking the highest-probability token makes generation deterministic and can reduce diversity. Sampling draws a token according to a chosen probability distribution. Temperature, top-k, and top-p define how that distribution changes. [3]

**Temperature changes concentration.** Divide logits by a positive temperature T before softmax. A smaller T makes score differences larger relative to the scale and concentrates probability on high-scoring tokens. A larger T spreads probability more evenly. T=0 is not a valid division; an API may deliberately define it as the separate operation of choosing argmax.

**Compare top-k and top-p on one distribution.** Suppose sorted probabilities are $(0.50,0.30,0.15,0.05)$. Top-k with k=2 keeps the first two and renormalizes them to $(0.625,0.375)$. Top-p with p=0.9 must keep the first three: the first two total only 0.8, while three reach 0.95. The token that crosses the threshold belongs in the retained set.

Top-k fixes the number of options even when confidence varies. Top-p adapts the number: a highly confident model may need very few tokens to reach the threshold, while a diffuse model may keep many. Neither setting automatically improves factuality; they change the diversity and risk of sampled alternatives.

**Implementation pitfalls.** Apply operations in a documented order, preserve at least one token, renormalize after filtering, and use a controlled random generator for reproducible tests. A seed makes a particular implementation repeatable under its conditions; it does not guarantee identical samples across different kernels or platforms. Test retained support and frequencies over many draws rather than expecting a single random output to prove correctness.
@@ 613
**Beam search is limited look-ahead over possible continuations.** Greedy decoding keeps only the best next token. That choice can be locally attractive but lead to a poor sequence later. Beam search keeps several partial sequences at each step, giving some alternatives a chance to recover. It remains approximate because alternatives outside the beam are discarded permanently. [3]

**See why greedy can lose.** At the first step, A has probability 0.6 and B has 0.4. Suppose the best continuation after A has probability 0.2, while the best continuation after B has probability 0.9. Their two-step sequence probabilities are 0.12 and 0.36. Greedy commits to A too early; a beam wide enough to retain B can discover the better continuation.

**What each beam stores.** Keep its token sequence, cumulative log probability, and whether it has ended. Log probabilities add across steps, avoiding numerical underflow from multiplying many small probabilities. Expand only unfinished beams, score candidates, and retain the best limited set. Completed beams must remain candidates without generating extra tokens after EOS.

**Why length normalization appears.** Multiplying by another probability usually decreases sequence probability, so raw log-probability comparison often favors short outputs. Dividing by a specified length penalty changes this preference. That is a deliberate change to the search score; the normalized score is no longer simply the original sequence log probability. Include or exclude prompt/EOS lengths consistently.

**How to validate it.** For a tiny vocabulary and short maximum length, enumerate all possible sequences and compare the beam result as beam width increases. Test early EOS, all beams completed, and length limits. Stopping at the first completed candidate is generally unsafe; an unfinished beam may later achieve a better normalized score. Better search of the model distribution also does not guarantee a better answer if the model's probabilities are themselves wrong.
@@ 614
**The draft model proposes; the target model remains authoritative.** A small model can generate several candidate tokens cheaply. A larger target model evaluates that block in parallel, accepting suitable proposals and correcting others. The aim is to save expensive serial target steps while preserving the target model's output distribution under the exact sampling algorithm. [3]

**Why “accept if it looks likely” is insufficient.** Suppose a two-token vocabulary has target probabilities $p=(0.8,0.2)$ and draft probabilities $q=(0.5,0.5)$. Accepting every draft token would produce 50/50 outputs rather than the required 80/20. The acceptance rule must compensate for how often the draft proposes each option.

For token A, acceptance is $\min(1,0.8/0.5)=1$. For token B, it is $\min(1,0.2/0.5)=0.4$. Accepted A mass is $0.5\times1=0.5$; accepted B mass is $0.5\times0.4=0.2$. The remaining rejection probability is 0.3. All positive residual mass $\max(p-q,0)$ lies on A, so every rejection resamples A. Final probabilities are therefore $(0.8,0.2)$ exactly.

**Extend this one-token argument carefully.** Each probability in a draft block is conditional on the prefix accepted so far. Once a proposal is rejected and replaced, subsequent target rows computed for the old draft prefix are no longer valid. Stop that verification block and restart from the corrected prefix. If every proposal is accepted, the target's extra final row can supply a bonus token.

**What determines speed.** Acceptance rate, draft cost, verification overhead, and serving batch size all matter. A draft that rarely agrees can waste work, and high-throughput batching can reduce the benefit. Test output frequencies on a tiny known distribution and prefix consistency before reporting speedups. Exactness refers to the specified target sampling distribution; vocabulary mismatches or inconsistent temperature/filtering rules break that claim.
@@ 615
**BPE builds reusable pieces rather than memorizing every word.** A word-level vocabulary struggles with rare words and grows huge; a character-only vocabulary produces long sequences. Byte-pair encoding learns frequently occurring adjacent pieces so common strings can be represented compactly while less common words can be assembled from smaller units. [3]

**Follow a small training example.** Suppose words initially split into characters and an end marker. If “l o w” appears often, the pair “l o” may have the highest count and be merged into “lo.” Later “lo w” may merge into “low.” In words containing “lower,” the learned “low” piece can be reused without requiring a separate vocabulary entry for every related spelling.

The exact next merge depends on pair counts across the training corpus, including word frequencies. After each merge, affected adjacency changes, so pair counts must be updated or recomputed. Overlapping occurrences require a consistent replacement rule: merging “aa” in “aaa” cannot independently consume both overlapping pairs at once.

**Training and tokenization are different phases.** Training learns an ordered list of merges. Tokenizing new text applies those fixed rules in their learned rank order; it does not decide fresh merges by counting pairs in the new sentence. Otherwise the same string could receive incompatible token IDs from one request to another.

**Define the surrounding conventions.** Case normalization, whitespace, Unicode/bytes, word boundaries, and unknown symbols affect the result. The educational implementation uses characters and an end marker rather than claiming to reproduce a particular production tokenizer. Test encode/decode round trips under its supported alphabet and deterministic ties. A shorter token sequence is not always linguistically cleaner: BPE optimizes frequency-driven compression, not semantic word boundaries.
@@ 616
**Preference data says which of two completions is better.** For one prompt, suppose a reviewer prefers a helpful answer over an evasive answer. DPO trains a language model to increase its relative preference for the chosen completion, measured against a fixed reference model. The reference anchors the change so the objective is about improvement from a baseline rather than only raw likelihood. [3]

**Read the margin before the loss.** Compute the policy's chosen-minus-rejected log probability, then subtract the same difference under the reference. If the policy favors the chosen answer more strongly than the reference does, the margin is positive. If both models have identical relative preferences, the margin is zero.

The loss is $-\log\sigma(\beta\Delta)$. At zero margin it equals $\log2\approx0.693$. With $\beta=1$ and margin 2 it is about 0.127; with margin -2 it is about 2.127. Thus the objective rewards moving the chosen answer ahead in relative log-probability. Beta changes how the preference margin enters the objective; it is not simply the learning rate.

**Why sequence masking matters.** A completion's log probability is the sum of its token log probabilities under the prompt and previous completion tokens. Padding should contribute nothing, and prompt tokens should not be counted as the completion. Replacing sums by average token log probabilities changes the length behavior and therefore changes the objective.

**Implementation checks.** Reference values should not receive gradients. Swapping chosen and rejected answers should negate the margin, though not the loss. Identical policy/reference ratios should give log two. Preference labels can be noisy or biased, so lower DPO loss does not guarantee factuality, broad capability, or agreement with every user's values. Evaluate the behaviors actually desired, including regression on ordinary tasks.
@@ 617
**A sliding window limits how far back each query can look.** Full causal attention considers every earlier token. Windowed attention considers only a recent region, reducing the number of permitted comparisons. This resembles reading with a fixed-size recent context instead of reopening the entire document for every new word. [3]

**Make the boundary concrete.** With a window size of three that includes the current token, query position five can see keys at positions three, four, and five. It cannot see positions zero through two, and causality still forbids positions six onward. The predicate is $0\le i-j<3$. If the window is one, each query sees only itself.

The same logic must hold during cached decoding. If five tokens are already cached, the first new query has absolute position five, not zero. Constructing its mask with a reset position would expose the wrong keys or hide every key. A test with nonzero query offset is therefore essential.

**A mask alone is not a speedup.** If the code first creates the full $S\times S$ score matrix and then sets most entries to negative infinity, it has already paid much of the dense memory/computation cost. A sparse or local kernel must avoid generating the forbidden entries to obtain the intended scaling benefit.

**Understand the modeling tradeoff.** Local attention can work well for nearby dependencies, and multiple layers can carry information farther indirectly. But it is not equivalent to direct access to arbitrary distant tokens. A detail far outside the window may be lost or require a separate global/memory mechanism. Evaluate long-distance tasks explicitly instead of assuming reduced memory leaves model quality unchanged.
@@ 618
**The obstacle is softmax's denominator.** Attention needs normalized weights, and every weight depends on the sum of exponentials over all keys. A naive implementation stores all scores before normalizing. Online softmax instead maintains compact running statistics while processing blocks, allowing attention's weighted average to be accumulated without retaining every score. [3]

**Why keep a running maximum?** Exponentiating a large raw score can overflow. Subtracting the maximum score makes all exponent arguments nonpositive, while leaving normalized probabilities unchanged. When a later block contains a larger maximum, earlier contributions must be rescaled to the new reference point.

**Work through scores 1 and 3.** After reading score 1 alone, the running maximum is 1 and the normalized-scale sum is $e^{1-1}=1$. On reading score 3, the maximum changes to 3. Rescale the previous sum to $e^{1-3}=e^{-2}$ and add the new contribution $e^{3-3}=1$. The final sum is $e^{-2}+1$, exactly what a stable softmax over both scores would use.

For values $v_1,v_2$, keep a running numerator too. It becomes $e^{-2}v_1+v_2$ under the new scale. Divide by $e^{-2}+1$ to obtain the same weighted output as ordinary attention. Apply this logic to entire blocks using their local maxima and sums.

**Be precise about what can stream.** Producing only the final weighted average needs running maximum, denominator, and value numerator. Producing every individual normalized probability also requires retaining or revisiting earlier scores, because the final denominator was not known when those scores arrived. This distinction explains how FlashAttention can avoid storing the attention matrix while still computing the exact attention output, up to floating-point ordering differences.
''')
