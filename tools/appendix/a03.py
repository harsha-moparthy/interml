from author import write
write(r'''
@@ 685|attention flash prenorm dl d2l
Let $s=\sum_{r=1}^{d_k}q_rk_r$. Independence and zero means give $E[q_rk_r]=0$ and $E[q_r^2k_r^2]=1$. Cross-covariances vanish, so $\operatorname{Var}(s)=d_k$; dividing by $\sqrt{d_k}$ restores unit variance. For $d_k=64$, the unscaled standard deviation is 8, versus 1 after scaling. These assumptions describe initialization, not a guarantee about trained features. [1]

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad \frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j).$$

Large score differences make softmax nearly one-hot and shrink its Jacobian. Scaling reduces that tendency while retaining meaningful relative scores. Distinguish the attention-softmax Jacobian from the combined cross-entropy gradient $p-y$, which need not vanish for a confidently wrong prediction.
@@ 686|prenorm norm attention gemma dl
Pre-norm computes $x_{l+1}=x_l+F_l(N(x_l))$. The residual branch therefore retains a direct identity path through each block. Post-norm computes $x_{l+1}=N(x_l+F_l(x_l))$; even the skip path passes through a normalization Jacobian. This changes initialization-scale gradients and helps explain why some post-norm configurations need careful warmup. [1]

$$J_{\rm pre}=I+J_FJ_N,\qquad J_{\rm post}=J_N(I+J_F).$$

Neither formula promises perfect gradient preservation: products of many Jacobians can still amplify or shrink directions. RMSNorm versus LayerNorm is a separate choice from placement. Also, the question's model list is not uniform: Gemma 2 uses additional normalization around sublayers rather than only the simplest pre-norm layout. Check the actual architecture, depth, residual scaling, and training recipe. [2] [4]
@@ 687|norm prenorm attention gemma torch
LayerNorm subtracts the feature mean and divides by the centered standard deviation; RMSNorm divides by root mean square without centering. Both usually learn a featurewise scale. RMSNorm removes a reduction and subtraction while controlling overall feature magnitude. It sacrifices invariance to adding a constant offset. [1]

$$\operatorname{RMSNorm}(x)_i=\gamma_i\frac{x_i}{\sqrt{d^{-1}\sum_jx_j^2+\epsilon}}.$$

For $x=(3,4)$ the RMS is $\sqrt{12.5}$, so the normalized vector keeps its direction. LayerNorm instead centers it to $(-0.5,0.5)$. Similar downstream accuracy is empirical, not mathematical equivalence. The claimed 10–15% saving is not universal: fused kernels, tensor sizes, precision, and the share of runtime spent in normalization determine end-to-end speed. Benchmark both latency and validation quality on the intended workload.
@@ 688|rope yarn attention gqa d2l
Partition each query and key into coordinate pairs. For pair $j$, rotate the query at position $m$ by $m\theta_j$ and the key at position $n$ by $n\theta_j$. Rotations are orthogonal and compose by adding angles. [1]

$$R(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix},\quad (R(m\theta)q)^TR(n\theta)k=q^TR((n-m)\theta)k.$$

The explicit positional component depends only on the offset, with sign determined by the rotation convention. The score still depends on token content through $q,k$; it is not a function of distance alone. Rotation preserves each pair's norm. Different frequencies represent offsets at different scales. Long-distance extrapolation can nevertheless fail because the model has not learned to interpret those phase combinations.

![RoPE rotates query and key coordinate pairs; their dot product depends on the angle difference.](/answer-diagrams/rope.svg)
@@ 689|yarn rope attention gemma v3
Position interpolation maps a new position $m$ to $m/s$, where $s=L_{new}/L_{train}$; extending 8k to 128k gives $s=16$. This keeps angles within the old range but compresses distinctions between nearby positions. Frequency-aware methods adjust different RoPE frequencies differently, rather than uniformly shrinking every angle. [1] [2]

$$\phi_j(m)=m\theta_j\quad\longrightarrow\quad \phi'_j(m)=m\theta'_j.$$

NTK-aware scaling changes the frequency base; YaRN blends interpolation and extrapolation across frequency bands and adjusts attention scaling. These modify positional encoding, not the information capacity of a fixed-size KV cache. Short continued training is often used; “without full retraining” does not mean no adaptation or no quality loss. Evaluate short-context regression, long-document tasks, retrieval across positions, and latency—not just whether 128k tokens fit.
@@ 690|gqa v3 attention paged scaling
MHA has separate keys and values for every query head. MQA shares one KV head across all query heads; GQA shares each KV head within a group. Thus MQA/GQA reduce cached state and decode-time KV reads, at a possible quality cost. Query-head count and attention score work can remain unchanged. [1]

$$M_{KV}=2BLSH_{KV}d_hb.$$

Here $b$ is bytes per scalar; the factor 2 counts keys and values. MLA instead caches a compressed latent representation plus positional information and uses compatible projections to recover attention behavior. Its saving is not simply “one KV head”: account for latent and positional dimensions and implementation details. [2] These methods primarily relieve memory capacity/bandwidth pressure during decoding; they do not remove autoregressive dependence or make all attention computation constant-time.
@@ 691|attention norm gemma v3 dl
Removing biases slightly reduces parameters, memory traffic, and operations, and simplifies some fused kernels. Large matrix weights dominate parameter count, so the savings are usually modest. A surrounding normalization may also make certain offsets redundant, but this depends on placement: RMSNorm does not subtract the mean and does not generally cancel an arbitrary bias. [2]

$$y=Wx+b,\qquad \Delta P=d_{out}\ \text{when the bias is removed}.$$

For a square $d\times d$ projection the removed fraction is about $1/d$, not a dramatic reduction. Bias-free training is an architectural choice validated by experiments, not a theorem that biases are useless. Some successful architectures retain biases or use special learned offsets. Inspect the model definition rather than generalizing “almost all layers” to every LLM. [3] [4]
@@ 692|glu attention gemma v3 dl
A conventional feed-forward block has an expansion matrix and a contraction matrix. SwiGLU adds a second expansion: one branch supplies values, while the other supplies a SiLU gate. Ignoring biases, the matrices have shapes $d\times h,d\times h,h\times d$. [1]

$$\operatorname{SwiGLU}(x)=(\operatorname{SiLU}(xW_g)\odot xW_u)W_d,\qquad \operatorname{SiLU}(z)=z\sigma(z).$$

The standard $h=4d$ MLP uses $2d(4d)=8d^2$ weights. SwiGLU uses $3dh$, so matching counts gives $h=8d/3=(2/3)4d$. Actual models round this for hardware alignment. The gate makes the transformation input-dependent and preserves smooth gradients, but it does not guarantee better quality for every dataset or optimizer.
@@ 693|gemma attention norm prenorm torch
Soft-capping maps a raw logit $z$ into a bounded value before softmax. Large magnitudes approach a finite ceiling instead of increasing without limit. Gemma 2 uses this technique in specified attention and output-logit paths. [1]

$$\tilde z=c\tanh(z/c),\qquad |\tilde z|<c,\qquad \frac{d\tilde z}{dz}=1-\tanh^2(z/c).$$

For small $|z|/c$, the mapping is approximately the identity. It limits extreme score differences and can improve stability, but extreme raw logits also have small gradients through the cap. It does not repair NaNs already produced upstream. Apply causal/padding exclusion as a true mask; transforming negative infinity through tanh would turn an excluded position into a finite score. The cap value and its location are architectural hyperparameters.
@@ 694|mamba ssd attention flash scaling
A state-space layer updates a fixed-size state, while kernelized linear attention accumulates sufficient statistics. For fixed feature/state dimension, both avoid an explicit sequence-by-sequence score matrix. [1] [2]

$$h_t=A_th_{t-1}+B_tx_t,\qquad S_t=S_{t-1}+\phi(k_t)v_t^T,\quad z_t=z_{t-1}+\phi(k_t),\quad y_t=\frac{\phi(q_t)^TS_t}{\phi(q_t)^Tz_t}.$$

The linear-attention state costs $O(rd_v+r)$, independent of sequence length but not of model width. Associative scans or chunked algorithms permit parallel training; autoregressive inference updates the state once per token. Compression can discard distinctions needed for exact retrieval of arbitrary old items. Full attention retains token-level keys/values and offers direct content-addressable access. “Linear” assumes fixed state dimensions and does not make every practical implementation faster at short lengths.
@@ 695|v3 attention mamba scaling dl
With $T$ tokens, $E$ experts, and top-$k$ routing, average assignments per expert are $kT/E$. A capacity-limited implementation may reserve approximately $\lceil c kT/E\rceil$ slots per expert. If more tokens select an expert, excess assignments must be dropped, rerouted, or processed with variable-size work. Dropping is an implementation choice, not inherent to MoE. [1]

A router can amplify its own early advantage: an expert receives more tokens, learns faster, and becomes even more attractive. This produces unbalanced load and poorly trained alternatives. Balancing penalties, router noise, capacity choices, and load-dependent selection biases address different parts of the problem. DeepSeek-V3 adjusts expert-selection biases for balancing while retaining affinity-based combination weights. Track load histograms, overflow, throughput, and quality, not only the mean routing probability.
@@ 696|sink rope paged attention scaling
Softmax must distribute total probability one, even when none of the visible tokens is especially useful. During training, initial tokens can become stable destinations for otherwise uninformative attention mass. Removing them at a sliding-window boundary changes the distribution the model expects. [1]

StreamingLLM retains a small initial-token cache plus a recent-token window. Cache memory is then $O((s+w)LH_{KV}d_h)$ rather than growing with the entire stream. The initial cache stabilizes attention while recent tokens carry local content.

This enables processing arbitrarily long streams with bounded cache, not remembering arbitrary facts from all past tokens. A fact evicted from the middle is unavailable unless encoded elsewhere. Test long-range recall separately from fluent continuation. Positional handling and the chosen sink/window sizes matter; the mechanism is not equivalent to simply increasing the trained context length.
@@ 697|bert t5 attention sbert rag
Encoder-only models see both left and right context and are well suited to classification, tagging, and pooled retrieval representations. Decoder-only models use a causal mask and learn next-token prediction. Encoder–decoder models encode an input bidirectionally and generate an output with causal decoding plus cross-attention. [1] [2]

$$p(y\mid x)=\prod_t p(y_t\mid x,y_{<t}).$$

Decoder-only models offer a simple common training/generation objective and flexible prompting at scale; that practical success does not establish universal architectural superiority. A bi-encoder can precompute document vectors and score queries with dot products, avoiding token-by-token generation. [4] Encoder–decoders remain useful when input conditioning and output generation have distinct roles. Compare quality at matched data/compute, latency, context size, and output requirements rather than assuming one architecture wins every task.
@@ 698|vit attention clip d2l dl
Split an image of height $H$, width $W$, and $C$ channels into nonoverlapping $P\times P$ patches. Flatten each patch to $P^2C$ values, project it to width $d$, and add positional information. A class token or pooling produces an image-level representation. [1]

$$N=HW/P^2,\qquad X_{tokens}\in\mathbb R^{N\times d}.$$

For a 224×224 RGB image and 16×16 patches, there are 196 tokens of 768 raw values each. CNNs bake in local connectivity and translation-equivariant weight sharing; ViT attention can mix distant patches immediately but must learn more spatial structure from data. Absolute positional encodings and patch boundaries weaken translation equivariance. Smaller patches improve spatial resolution while increasing quadratic full-attention work; hybrid convolutional stems and window attention offer intermediate tradeoffs.
@@ 699|llava blip flamingo clip vit
A vision encoder emits feature vectors that generally do not match an LLM's input distribution or width. A learned linear/MLP projector maps those vectors into language-model embedding space, often treating them as visual prefix tokens. This is the simplest bridge and retains many visual tokens. [1]

BLIP-2's Q-Former uses a small set of learned queries to extract a compact representation from frozen visual features. Flamingo uses a visual resampler and gated cross-attention inserted into the language model. [2] [3]

$$Z_{visual}\in\mathbb R^{N\times d_v}\ \longrightarrow\ Z_{bridge}\in\mathbb R^{M\times d_{LM}}.$$

Smaller $M$ reduces language-model attention cost but can discard fine print and spatial detail. OCR, high-resolution tiling, video sampling, and multimodal instruction data determine capabilities beyond the bridge architecture itself.
@@ 700|gqa paged scaling v3 attention
Count both K and V for every token, KV head, layer, and batch element. FP16 occupies 2 bytes per scalar. The number of model parameters does not enter this cache calculation directly. [1] [2]

$$M=2\times32\times80\times16384\times8\times128\times2=171{,}798{,}691{,}840\ \text{bytes}.$$

That is **171.79869184 GB** in decimal units, or **160 GiB** in binary units. This is logical unsharded KV storage only; weights, activations, allocator overhead, and temporary buffers are additional. Uniformly sharding these KV heads across eight compatible devices would give 20 GiB per device before overhead; other parallel layouts may replicate state. Reducing sequence length or batch size halves memory proportionally, whereas using fewer query heads alone does not change this formula.
@@ 701|scaling flash paged gqa cuda
Prefill processes many known prompt tokens at once, so weight matrices are reused across a large matrix–matrix multiplication. Decode commonly processes one new token per sequence, making weight and KV reads large relative to arithmetic, particularly at small batch sizes. [1] [2]

$$T\gtrsim\max\left(\frac{F}{P_{peak}},\frac{M_{bytes}}{BW}\right),\qquad I=F/M_{bytes}.$$

The roofline comparison is between arithmetic intensity $I$ and the device's compute/bandwidth ratio. Larger decode batches improve weight reuse and may become compute-bound. Long-context prefill can have substantial attention and memory costs. Therefore “prefill compute-bound, decode bandwidth-bound” is a useful common regime, not a universal law. Measure kernels, batch sizes, context lengths, and communication to identify the actual bottleneck.

![Roofline analysis compares compute time and memory-transfer time; the larger bounds execution.](/answer-diagrams/roofline.svg)
@@ 702|flash flash2 flash3 attention cuda
FlashAttention computes the same dense attention, up to floating-point rounding, without materializing the full $N\times N$ score/probability matrix in HBM. It loads query/key/value tiles into fast memory and maintains a running row maximum, normalization sum, and weighted value sum. [1]

$$m'=\max(m,m_b),\quad \ell'=e^{m-m'}\ell+\sum_j e^{s_j-m'},\quad o'=e^{m-m'}o+\sum_j e^{s_j-m'}v_j.$$

Backward reloads tiles and reconstructs probabilities from saved row normalization statistics and inputs. This spends arithmetic to save memory traffic. FlashAttention-2 improves parallel work partitioning and reduces non-matmul operations. FlashAttention-3 exploits Hopper-specific asynchronous execution and low-precision techniques. [2] [3] Dense attention FLOPs remain quadratic in sequence length; lower IO does not make the algorithm sparse or approximate.
@@ 703|paged gqa scaling flash v3
PagedAttention divides each sequence's logical KV cache into fixed-size token blocks. A block table maps logical positions to physical GPU blocks, allowing sequences to grow without reserving one large contiguous allocation. Freed blocks can be reused by other requests; compatible prefixes can share blocks with reference counts and copy-on-write. [1]

If a block holds $b$ tokens, the last block wastes at most $b-1$ token slots per sequence. Thus paging reduces fragmentation; it does not literally eliminate all internal waste. Metadata, partially used tail blocks, and non-cache allocations remain.

This increases useful cache occupancy and supports dynamic scheduling, but does not reduce the mathematical KV values needed for an uncompressed sequence. Admission control must still reserve room for growth, and a shared cache must respect tenant, model, and adapter identity.
@@ 704|paged scaling gqa flash sre
Static batching waits for a group of requests and runs them together. When one response ends early, its slot can remain idle while longer responses continue. Continuous batching makes a new scheduling decision each decoding iteration: remove finished requests and admit waiting work. [1]

For lengths 10 and 100, a static two-slot batch has 90 unused sequence-steps after the short response finishes. Reusing that slot improves occupancy. Chunked prefill can interleave long prompt work with decode iterations to reduce stalls.

Throughput alone is insufficient: optimize time-to-first-token, inter-token latency, queue delay, and tail response time. Large batches consume more KV memory and can delay interactive requests. Fairness, maximum waiting time, token budgets, and cache-aware admission keep long or high-volume tenants from monopolizing the server. [2]
@@ 705|spec medusa eagle attention scaling
Let the draft distribution be $q$ and target be $p$ at the same prefix. Sample $x\sim q$ and accept with $a(x)=\min(1,p(x)/q(x))$. If rejected, sample from the normalized positive residual $r(x)\propto[p(x)-q(x)]_+$. [1]

$$P(output=x)=\min(p(x),q(x))+[p(x)-q(x)]_+=p(x).$$

The residual is weighted by total rejection probability, equal to its normalizing mass. For a proposed block, verify conditioned prefixes in parallel, stop at the first rejection, and discard later proposals because their prefixes are now wrong. If all proposals pass, draw the extra target token. Exactness requires consistent temperature/filter transformations and correct residual sampling; approximate acceptance rules can change the distribution. Speed depends on draft cost, acceptance, and verification hardware utilization.
@@ 706|medusa eagle spec paged scaling
A standalone draft model has its own parameters and autoregressive state. Medusa attaches multiple prediction heads to target-model hidden states, creating candidate continuation trees that can be verified with tree-structured attention. EAGLE drafts using target features and token information, rather than independently modeling only token history. [1] [2]

These approaches reduce drafting overhead and share information with the target, but require extra training and architecture support. Their speedup depends on how many proposed tokens survive verification relative to proposal and verification cost.

Do not infer distributional exactness solely from “multi-head decoding.” Exact speculative sampling needs the appropriate proposal probabilities and rejection/residual rule; some practical acceptance modes trade exactness for speed. Compare acceptance rate, quality, latency, and memory under the actual decoding configuration. [3]
@@ 707|gptq awq smooth qlora torch
GPTQ uses an approximate second-order reconstruction objective to quantize weights sequentially and compensate remaining weights for quantization error. AWQ identifies activation-sensitive channels and rescales them to preserve important weights. SmoothQuant applies an equivalent channel scaling that shifts difficult activation magnitudes into weights, enabling lower-precision activation computation. [1] [2] [3]

$$XW=(XD^{-1})(DW).$$

Activation outliers can vary across inputs and channels, so one shared scale wastes most quantization bins on extremes. Weights are fixed and easier to inspect/calibrate once. Per-channel or block scales help but add metadata and kernel complexity. Evaluate representative calibration data, out-of-domain inputs, long sequences, and end-to-end kernel support: a numerically valid compressed model is not automatically faster on a given GPU.
@@ 708|gqa paged qlora smooth scaling
KV quantization reduces bytes per cached scalar, allowing more concurrent or longer sequences. Quantization error perturbs keys before softmax and values after weighting; errors can compound across layers and affect rare retrieval events even when average perplexity changes little. No fixed perplexity penalty follows solely from “FP8” or “INT4.” [1] [4]

$$\hat x=s(q-z),\qquad \Delta s_{ij}=q_i^T(\hat k_j-k_j)/\sqrt{d_h}.$$

Key and value tensors may need different granularity because their outlier structure differs. Include scales, zero points, and an optional recent-token high-precision buffer in memory accounting. Measure perplexity versus context length, retrieval at different positions, and downstream quality. A cache that is nominally four times smaller can deliver less than four times practical capacity once metadata and other model allocations are counted.
@@ 709|paged scaling rag sre gqa
Prefix caching reuses KV states for an identical token prefix instead of recomputing its prefill. Cache identity must include token IDs, model/version, relevant positional configuration, adapter state, and isolation policy. Similar semantic meaning is not enough: token sequences and computation must match. [1]

$$T_{prefill,new}\approx T_{lookup}+T_{uncached\ suffix}.$$

Put stable shared instructions before dynamic material to maximize reusable prefixes. Multi-turn systems can append to a retained conversation state, but inserting or editing earlier tokens invalidates downstream causal states. Bound caches with eviction and per-tenant quotas; invalidated permissions must not expose cached content. Cached prefixes save prefill work, not the output decoding itself. Measure token-weighted hit rates and total latency separately, since many short hits may save little compute.
@@ 710|zero qlora scaling torch checkpoint
Under the commonly specified mixed-precision AdamW layout: FP16/BF16 model weights use 2 bytes/parameter, low-precision gradients 2, FP32 master weights 4, and FP32 first/second moments 4 each. Total persistent parameter-related storage is 16 bytes/parameter. [1]

$$7\times10^9\times(2+2+4+4+4)=112\times10^9\ \text{bytes}\approx104.31\ \text{GiB}.$$

This excludes activations, attention/KV-like training intermediates, temporary optimizer buffers, communication buffers, and allocator overhead. If gradients are FP32, the corresponding layout uses 18 bytes/parameter. Some implementations keep FP32 parameters directly and use autocast without a separate master copy. State the actual optimizer and dtype layout before quoting a number. Sharding and quantized optimizer states change per-device memory without changing the parameter count.
@@ 711|zero megatron scaling checkpoint torch
DDP replicates parameters and optimizer state, divides examples across workers, and synchronizes gradients. ZeRO-1 shards optimizer states; ZeRO-2 additionally shards gradients; ZeRO-3/FSDP additionally shards parameters, gathering needed parameters around computation. [1]

Tensor parallelism splits individual matrix operations, requiring collectives within layers. Pipeline parallelism assigns different layers to stages and sends activations/gradients between them; microbatches reduce bubbles. Sequence parallelism distributes selected activation operations across tokens, whereas context parallelism distributes long sequences and exchanges information needed for attention. Terminology varies by implementation. [2] [3]

Choose according to the constrained resource: model-state capacity, activation memory, per-layer compute, or global throughput. Combinations are common. A theoretical memory saving may be erased by slow collectives, temporary all-gathers, poor overlap, or small microbatches; profile the whole step.
@@ 712|megatron scaling zero cuda torch
Tensor parallel layers exchange partial results frequently, often several times per transformer block. Their collectives are on the critical path and can involve relatively small messages. Fast low-latency links within a node therefore help. Pipeline stages communicate at boundaries, while data-parallel gradient communication can often overlap backward computation. [1] [2]

$$T_{comm}\approx\alpha\cdot n_{messages}+\frac{bytes}{BW}.$$

The latency term matters for repeated small collectives; bandwidth matters for large transfers. “TP only within a node” is a useful default, not a correctness rule. Fast inter-node fabrics, very large models, and sequence/batch shapes can justify cross-node TP. Conversely, poor topology can make even intra-node TP inefficient. Map logical ranks to physical links and compare scaling efficiency rather than relying on node boundaries alone.
@@ 713|checkpoint zero scaling torch flash
Autograd normally saves intermediate activations needed by backward. Checkpointing saves only selected boundaries and reruns the intervening forward computations during backward. Parameters and optimizer state are unchanged; the saving is chiefly activation memory. [1]

If an ordinary step costs one forward $F$ plus backward approximately $2F$, recomputing one forward changes $3F$ to $4F$, about 33% extra arithmetic. This is a rough model, not a guaranteed runtime penalty. Selective checkpointing, fused attention, communication overlap, and kernel costs alter it.

For a chain of $L$ similarly sized layers, checkpointed segments can reduce stored boundary/interior activations toward $O(\sqrt L)$ under a suitable schedule. Preserve RNG behavior for dropout and handle mutable state carefully; rerunning a stateful forward can otherwise change the function being differentiated. [4]
@@ 714|lora dora qlora prefix torch
Write the adapted weight as $W'=W_0+sBA$, with $A\in\mathbb R^{r\times d_{in}}$ and $B\in\mathbb R^{d_{out}\times r}$. Random $A$ and zero $B$ make the initial update exactly zero, preserving the base model's outputs while allowing $B$ to learn immediately. [1]

$$\nabla_B\mathcal L=sGA^T,\qquad \nabla_A\mathcal L=sB^TG,\qquad G=\nabla_{W'}\mathcal L.$$

If both matrices start at zero, both gradients are zero and the adapter is stuck under ordinary gradient descent. If both are random, learning can proceed, but the initial function is perturbed and may need careful scale control. The zero-initialized factor determines which side receives the first nonzero update; the asymmetry is intentional.
@@ 715|lora qlora dora prefix scaling
There is no universal best LoRA target set. Updating only query/value projections is cheap and can work well when attention adaptation dominates the task. Updating all attention projections and MLP projections gives more capacity to change representations and often helps harder domain adaptation, at higher optimizer/activation cost. [1] [2]

$$P_{adapter}=r(d_{in}+d_{out})\quad\text{per adapted linear layer}.$$

Choose targets, rank, scaling, and training data jointly. Compare on a held-out task set, including retention of general capabilities, rather than training loss alone. A rank budget spread across many layers differs from the same number of parameters concentrated in a few. Mergeability, quantized-base precision, adapter switching, and inference kernel support are additional practical constraints. Report the exact target module names to make results reproducible.
@@ 716|qlora lora gptq awq zero
QLoRA stores a frozen base model in 4-bit quantized form while training small higher-precision LoRA updates. During computation, needed base blocks are dequantized into a compute dtype; gradients update adapters, not the discrete base codes. [1]

NF4 uses nonuniform levels designed around a normal weight distribution. Double quantization compresses the block-scale metadata as well as weights. Paged optimizers use managed memory to accommodate optimizer-memory spikes, with potential transfer overhead.

$$W_{effective}=\operatorname{dequant}(W_{4bit})+sBA.$$

Four-bit storage does not mean every multiplication or activation is four-bit. Include quantization metadata, adapters, gradients, optimizer states, activations, and temporary dequantized tiles in capacity estimates. Calibration/quantization choices and adapter capacity still affect quality; QLoRA is parameter-efficient fine-tuning, not unrestricted full-parameter training at four bits.
@@ 717|lora dora prefix qlora dl
LoRA learns an additive low-rank matrix update; it can usually be merged into a compatible dense weight for inference. DoRA separates weight magnitude from direction, applying low-rank adaptation to the direction with a separately learned magnitude. Prefix tuning learns continuous prefix states that attention can consult. Bottleneck adapters insert trainable down-projection, nonlinearity, and up-projection modules between frozen layers. [1] [2] [3]

$$\Delta W_{LoRA}=BA,\qquad W_{DoRA}=m\frac{W_0+BA}{\|W_0+BA\|}\ \text{along the chosen weight axis}.$$

Prefix tuning increases attention/cache work; unmerged adapters add computation; merged LoRA avoids its extra matrix operations but requires storing or materializing a merged model. Compare trainable parameter count, activation memory, task quality, switching latency, and multi-adapter serving—not just checkpoint size.
@@ 718|rlhf ppo dpo zero scaling
The policy generates responses and receives gradients. A frozen reference policy defines a behavioral baseline, often the SFT model. A reward model scores responses according to learned preferences. A value model estimates expected returns so advantages can reduce policy-gradient variance. Classic implementations may hold four large networks, but sharing, offloading, or sequential execution can change memory requirements. [1] [2]

$$r_t^{shaped}=r_t^{task}-\beta\log\frac{\pi_\theta(a_t\mid s_t)}{\pi_{ref}(a_t\mid s_t)},\qquad A_t\approx R_t-V_\phi(s_t).$$

PPO reuses rollouts with clipped policy ratios and trains the critic on returns. Account separately for policy/critic optimizer state, frozen-model weights, rollout KV caches, and activations. Reward-model quality is a bottleneck: optimizing an imperfect scorer harder can degrade true human-valued behavior.
@@ 719|rlhf dpo ppo prm r1
A reward model is a proxy, so unconstrained optimization can exploit its mistakes. A KL penalty charges the policy for moving too far from a trusted reference and balances score improvement against distributional drift. [1] [2]

$$\max_\pi E_{y\sim\pi}[r(x,y)]-\beta D_{KL}(\pi(\cdot\mid x)\|\pi_{ref}(\cdot\mid x)).$$

Large $\beta$ constrains adaptation; small $\beta$ permits larger changes. Reward hacking occurs when behavior improves the proxy while worsening the intended objective—for example, verbose answers exploiting a length-biased scorer. KL does not guarantee safety or truth: the reference and scorer can both be wrong. Use independent evaluations, adversarial cases, reward-distribution monitoring, and human review. Also specify whether KL is estimated tokenwise, sequencewise, or through an approximation.
@@ 720|dpo rlhf ppo ipo simpo
For a fixed prompt, maximizing expected reward minus forward KL gives $\pi^*(y)\propto\pi_{ref}(y)e^{r(y)/\beta}$. Rearranging expresses reward through a policy/reference likelihood ratio plus a prompt-specific constant. In a Bradley–Terry preference model, that constant cancels between chosen and rejected responses. [1]

$$\mathcal L_{DPO}=-\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y^+\mid x)}{\pi_{ref}(y^+\mid x)}-\log\frac{\pi_\theta(y^-\mid x)}{\pi_{ref}(y^-\mid x)}\right]\right).$$

This trains directly on preference pairs without explicitly fitting a separate reward model or sampling PPO rollouts. It does not remove the need for reliable preference data, an appropriate reference, or held-out evaluation. The objective increases a relative margin; it does not require the absolute probability of every preferred answer to increase.
@@ 721|dpo ipo kto simpo rlhf
DPO optimizes a relative chosen/rejected log-ratio, so both absolute probabilities can decrease while the preferred response decreases less. Noisy or separable preference data can also encourage excessive margins; length and reference-policy effects complicate evaluation. [1]

IPO uses a squared target for a preference-related log-ratio rather than continually increasing a logistic margin. KTO learns from individual desirable/undesirable labels using a prospect-theoretic utility formulation. SimPO uses length-normalized sequence log probability and a target margin without a separate reference model. [2] [3] [4]

These are different objectives, not universal fixes for every DPO failure. Compare absolute likelihoods, length distributions, win rates under controlled judging, and out-of-distribution tasks. Preserve the distinction between an observed training dynamic and proof that a new objective improves the underlying human preference.
@@ 722|r1 ppo dpo prm rlhf
GRPO samples a group of responses to the same prompt, scores them, and uses within-group rewards as a baseline instead of fitting a separate value network. A common normalized advantage is [1]

$$A_i=\frac{r_i-\bar r}{s_r+\epsilon},\qquad \bar r=G^{-1}\sum_{j=1}^{G}r_j.$$

A clipped policy-ratio objective then updates the policy, often with a reference KL term. Eliminating the critic saves weights, optimizer state, and value-learning work; generating multiple responses and scoring them still costs compute. If all rewards are identical, the group provides no ranking signal and its normalized advantages are zero. Reward verifiability matters: passing a unit test can provide stronger feedback than a fallible stylistic judge. Group normalization also changes weighting across prompts with different reward variability.
@@ 723|prm r1 rlhf ppo dpo
An outcome reward model scores the final answer or overall completion. A process reward model scores intermediate steps, making it possible to identify where a derivation first becomes invalid and to guide search before the final answer. [1]

A correct final number can follow faulty reasoning, and a sound partial derivation can end with an arithmetic slip. Outcome-only labels do not distinguish these cases. Process supervision can provide denser feedback but requires reliable step boundaries and annotations; alternative valid derivations should not be penalized merely for differing from a reference.

During search, aggregate step scores carefully: a product of probabilities penalizes long solutions, while a minimum focuses on the weakest step. Validate process judges against human or executable checks and keep final-answer verification. Neither scorer's confidence is a proof of correctness.
@@ 724|rag lora t5 ragas sre
Use RAG when answers depend on changing, attributable, access-controlled documents: retrieve permitted evidence at request time and cite it. Use SFT/LoRA to alter behavior, format, terminology, or task execution through examples; it is a poor replacement for an auditable fact store. Use long-context prompting when the relevant corpus is small enough to include and its latency/cost is acceptable. [1] [2]

Estimate total cost as ingestion/index maintenance plus retrieval/reranking plus input/output token serving. RAG can miss evidence; long context can distract or dilute attention; fine-tuning can memorize stale facts. Evaluate on answerable, unanswerable, conflicting, and permission-restricted cases.

These choices combine: a fine-tuned model can answer from retrieved passages in a long prompt. Select by freshness, attribution, domain accuracy, privacy, volume, and measured latency rather than treating the methods as mutually exclusive.
@@ 725|faiss hnsw mrl sbert irl
Flat search compares a query with all stored vectors: it provides exact nearest neighbors under the chosen metric but costs $O(Nd)$ per query. IVF partitions the space, probing selected clusters; fewer probes reduce work and may miss neighbors. PQ compresses vectors into subvector codebooks, reducing storage while introducing distance error. HNSW navigates a multilayer neighbor graph, trading graph memory and build cost for high-recall search. [1] [2]

$$M_{flat}\approx Ndb,\qquad M_{PQ,codes}\approx Nm\log_2(K)/8.$$

PQ's formula excludes codebooks and index metadata. IVF and PQ can be combined; HNSW may also index compressed representations. Benchmark recall against exact search, p95 latency, update/delete behavior, and RAM on representative queries. Recall is approximate-search recall, not proof that the embedding represents user relevance well.
@@ 726|splade irl sbert colbert faiss
Dense representations smooth semantic similarity, which helps paraphrases but can blur rare strings or exact identifiers such as a one-character-different SKU. Lexical retrieval preserves token evidence. BM25 scores term matches with inverse-document-frequency and length normalization; SPLADE learns sparse vocabulary weights and expansion terms. [1] [2]

A robust pipeline retrieves lexical and dense candidates independently, unions authorized results, and fuses rankings before reranking. Reciprocal rank fusion avoids directly adding incomparable raw scores:

$$RRF(d)=\sum_m\frac{1}{c+rank_m(d)}.$$

Tokenization and normalization still matter: preserve meaningful hyphens, identifiers, and field boundaries. Evaluate exact-match slices separately from semantic queries. Hybrid retrieval adds index-consistency and latency costs; it should earn that complexity through measured recall gains.
@@ 727|sbert colbert rag faiss irl
A bi-encoder maps the query and document independently to vectors, allowing documents to be embedded offline and indexed. A cross-encoder processes the query/document pair jointly, so token interactions depend on that specific query but cannot be fully precomputed. [1]

$$s_{bi}(q,d)=f(q)^Tg(d),\qquad s_{cross}(q,d)=h([q;d]).$$

Use a bi-encoder or hybrid retriever over the corpus, then cross-encode a bounded candidate set. If first-stage recall misses the relevant document, the reranker cannot recover it. Measure recall@k upstream and ranking metrics downstream, and include truncation of long documents in evaluation. Joint scoring often improves relevance but is not inherently calibrated across queries, and its cost scales with the number and length of candidate pairs.
@@ 728|colbert sbert faiss irl mrl
ColBERT retains contextualized token vectors rather than compressing each document to one vector. For every query token, find the strongest matching document token, then sum those scores. Document token representations are precomputed. [1]

$$s(q,d)=\sum_{i=1}^{|q|}\max_{j\le |d|}q_i^Td_j.$$

This preserves fine-grained matches while avoiding a full joint transformer pass for every query/document pair. For example, different document tokens can separately satisfy “red,” “waterproof,” and “jacket.” It needs substantially more index storage than one vector per document, plus efficient token-level candidate generation and compression. MaxSim also discards some relational structure, so “near cross-encoder accuracy” is benchmark-dependent rather than guaranteed. Compare storage, latency, and relevance on the intended corpus.
@@ 729|mrl sbert faiss colbert clip
Matryoshka training applies objectives to nested prefixes of one representation, so early coordinates are explicitly useful rather than arbitrary slices of a full embedding. [1]

$$\mathcal L=\sum_{d\in\mathcal D}w_d\,\mathcal L_d(z_{1:d}).$$

At retrieval time a short prefix can support cheap coarse search; longer vectors refine shortlisted candidates. For 1 million float32 vectors, reducing 768 to 128 dimensions reduces raw vector storage from 3.072 GB to 0.512 GB, before index overhead. Normalize consistently after truncation when using cosine similarity. This behavior requires compatible training: truncating an ordinary embedding is not equivalent. Measure recall at each dimension and account for whether the encoder still computes the full vector even when the index stores only a prefix.
@@ 730|ragas rag judge irl sbert
Context relevance asks whether retrieved passages help answer the query. Groundedness asks whether claims in the answer are supported by those passages. Answer relevance asks whether the response actually addresses the question. A response can pass one and fail another: faithfully repeating an irrelevant passage is grounded but unhelpful. [1]

Build examples with known relevant evidence, independently check claims against cited spans, and score answer completeness and correctness. Include abstention when evidence is missing. Use retrieval metrics such as recall@k only when relevance labels exist; reference-free LLM scores are noisy proxies, not ground truth.

Separate retrieval failures from generation failures by rerunning generation with gold context. Calibrate automated graders on human-reviewed slices, track judge agreement, and version prompts, evidence, and models so regression comparisons are meaningful. [3]
@@ 731|judge ragas prm rlhf irl
A judge can favor the first answer, longer prose, familiar model style, or its own outputs. These effects can produce an apparent win without better correctness. [1]

Blind model identity, randomize answer order, repeat with swapped order, and use a rubric separating correctness, completeness, citation support, and style. Compare against independently annotated human examples, including disagreements and difficult edge cases. Report agreement and confidence intervals instead of a single unqualified score.

For pairwise tests, cluster uncertainty by prompt because repeated judgments of one prompt are dependent. Fix judge version and decoding settings for regression tests, but periodically recheck calibration. Executable tests or source verification are stronger for claims they can directly assess. A fluent judge rationale is not evidence that its verdict is correct.
@@ 732|ddpm ddim ldm cfg flow
The forward process adds Gaussian noise on a fixed schedule. Combining transitions yields a direct noisy sample at any training timestep. [1]

$$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),\quad x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.$$

Train a network to predict the sampled noise with a weighted or simplified squared-error objective. The reverse sampler uses that prediction to parameterize denoising transitions. Predicting noise is convenient because its target scale is standardized, but predicting $x_0$ or velocity is also valid with appropriate parameterization and weighting.

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.$$

At very noisy timesteps this conversion can magnify errors. The training objective, schedule, and sampling update must be consistent; noise prediction is not uniquely required by diffusion theory.
@@ 733|ddim ddpm flow cfg ldm
DDIM constructs sampling trajectories compatible with the same training marginals but changes the reverse process. With stochasticity parameter $\eta=0$, the update is deterministic given the starting noise. It can visit a subset of timesteps rather than every original diffusion step. [1]

$$x_{t'}=\sqrt{\bar\alpha_{t'}}\hat x_0+\sqrt{1-\bar\alpha_{t'}}\epsilon_\theta(x_t,t),\qquad t'<t.$$

This is the deterministic form, using a consistent predicted clean sample. Different initial noise still gives different outputs, so deterministic does not mean every generated image is identical. Skipping many steps reduces network evaluations but introduces approximation error because the denoiser is imperfect. “1000 to 20” is an example budget, not a quality guarantee. Compare sample quality, diversity, guidance strength, and latency under a fixed model and evaluation protocol.
@@ 734|ldm ddpm ddim dit cfg
Latent diffusion first trains an autoencoder that maps images into a lower-resolution latent space. A diffusion model then learns to generate those latents; the decoder converts the final latent back to an image. [1]

$$z=E(x),\qquad z_T\to z_0,\qquad \hat x=D(z_0).$$

Spatial compression greatly reduces denoiser work and memory, making high-resolution generation more practical. The tradeoff is an autoencoder bottleneck: fine text, small objects, and high-frequency details can be lost before diffusion begins. Scaling and normalization of latents must match training. Conditioning can enter through cross-attention, concatenation, or modulation. Judge both reconstruction fidelity and generation quality; a strong denoiser cannot recover information the chosen representation cannot express.

![Latent diffusion compresses an image, denoises in latent space, then decodes back to pixels.](/answer-diagrams/diffusion.svg)
@@ 735|cfg ddpm ddim ldm dit
Classifier-free guidance trains one conditional denoiser while occasionally dropping the condition, allowing both conditional and unconditional predictions. Sampling extrapolates in the direction of their difference. [1]

$$\hat\epsilon=\epsilon_u+w(\epsilon_c-\epsilon_u).$$

Here $w=0$ is unconditional and $w=1$ uses the ordinary conditional prediction. Larger values generally increase prompt adherence but can reduce diversity or produce oversaturation and artifacts; the exact effect depends on model and scheduler. No separate classifier is required, but two branches commonly increase sampling compute. Batched evaluation can improve utilization without eliminating that work. The conditional and unconditional predictions must correspond to the same noisy sample and timestep. Guidance scales are not directly comparable across all parameterizations or pipelines.
@@ 736|dit flow ddpm ldm cfg
DiT changes the denoiser backbone: patchify latent images, process tokens with transformer blocks, and condition through mechanisms such as adaptive normalization. Diffusion or flow objectives can both use a transformer; the architecture and generative training path are distinct decisions. [1] [2]

For a simple flow-matching path between noise $x_0$ and data $x_1$,

$$x_t=(1-t)x_0+tx_1,\quad u_t=x_1-x_0,\quad \mathcal L=E\|v_\theta(x_t,t)-u_t\|^2.$$

Sampling integrates $dx/dt=v_\theta(x,t)$ from noise to data. Rectified-flow methods seek easier, straighter transport paths; discretization and model error still matter. A transformer does not itself guarantee fewer sampling steps, and flow matching does not inherently require a transformer. Compare objective, backbone, solver, and guidance independently.
''')
