from expand import write
write(r'''
@@ 863
**A fast calculator still waits if numbers arrive too slowly.** A GPU has a compute ceiling, measured in FLOPs per second, and a memory-delivery ceiling, measured in bytes per second. Arithmetic intensity I says how much calculation you perform for each byte fetched. The roofline bound is $performance\leq\min(C,BI)$, where C is peak compute and B is bandwidth. [1]

The two roofs meet when $C=BI$, so $I_{ridge}=C/B$. Using the question's nominal specifications gives 989/3.35=295.2 FLOPs per byte, or 989/4.8=206.0. Below this point, memory can limit performance before the arithmetic units are fully occupied. These are theoretical bounds using the stated numbers, not measurements of every H100/H200 configuration.

**Why one-token decoding is different from a large matrix multiply.** In a weight-dominated BF16 matrix-vector product, each weight costs about two bytes to load and contributes a multiply plus an add: about two FLOPs. That is roughly one FLOP per byte. With 3.35 TB/s, the corresponding bandwidth roof is only 3.35 TFLOPs/s, about 0.34% of 989 TFLOPs/s.

Batching lets one fetched weight serve multiple input vectors, increasing reuse and arithmetic intensity. Quantization reduces bytes per weight. Neither guarantees peak compute because activations, caches, kernel shape, and other operations matter. A low percentage of peak FLOPs is not by itself evidence of a bad kernel: it can be exactly what the memory budget predicts.
@@ 864
**HBM is the large warehouse; on-chip storage is the small workbench.** A tensor can fit in device memory while being far too large for shared memory or registers. Efficient kernels move a tile onto the chip, reuse it, and write only the needed result back. Shared memory is explicitly coordinated by the program; caches manage reuse differently. [1]

Consider three elementwise transformations of a million-element float32 tensor. If each runs separately and writes its result to device memory, each stage moves approximately four megabytes in and four megabytes out: about 24 MB across three stages, ignoring caching. An ideal fused kernel reads the initial four megabytes, computes all three operations locally, and writes the final four megabytes: about 8 MB.

The arithmetic did not disappear. The savings came from avoiding intermediate trips to the warehouse. This is especially valuable when those trips cost more time than the operations themselves.

**The workbench can overflow.** A fused kernel that keeps too many temporary values may use excessive registers or shared memory, reducing how many work groups run concurrently. Spilling registers can bring memory traffic back. Hence “fuse everything” is not a sound rule. Use a profiler to compare traffic, occupancy, and elapsed time. Capacity and aggregate bandwidth numbers depend on the device and measurement convention; compare actual bottlenecks rather than treating all on-chip storage as one uniform memory.
@@ 865
**The eager expression normally has three operations: sine, cosine, and addition.** The question's two-kernel description omits the addition. Conceptually, eager execution computes temporary arrays a=sin(x) and b=cos(x), then y=a+b. Each operation may launch its own GPU kernel. Compiler fusion can calculate all three values for each element before storing y. [1]

**Count the transfers.** The sine kernel reads x and writes a. The cosine kernel reads x and writes b. The addition kernel reads a and b and writes y. That is seven element-sized transfers in the simple no-cache model. A fused loop reads x once and writes y once: two transfers. For a million float32 elements, the simplified traffic falls from 28 MB to 8 MB.

This is a traffic estimate, not a guaranteed 3.5-times speedup. Sine and cosine themselves can be expensive, caches can serve some reads, and launch overhead matters differently for small and large arrays. Compilation can also change numerical details within the selected dtype and optimization settings.

Measure steady-state execution after compilation and warmup, using proper GPU synchronization or events. Inspect graph breaks if expected fusion does not occur. A Python branch, unsupported operation, or unsuitable shape can separate the computation into multiple regions. The useful question is whether generated code reduces the measured bottleneck while preserving required numerical accuracy.
@@ 866
**Small GPU tasks can spend more time waiting for dispatch than doing arithmetic.** Suppose a workload has 100 tiny operations, each with an illustrative 3 microseconds of launch overhead and 2 microseconds of useful work. Repeated CPU dispatch can become a substantial part of the timeline. CUDA Graph replay submits a previously captured operation sequence with much less repeated setup. The numbers here illustrate the mechanism; they are not a hardware benchmark. [1]

Think of recording a production schedule and replaying it with new materials in the same input buffers. The operations and their dependencies are already established. This does not make a large matrix multiply mathematically cheaper; it removes orchestration work around the GPU operations.

**Replay needs stable execution assumptions.** The captured path uses fixed buffer addresses and a known operation structure. Applications often use persistent buffers and shape buckets, copying new input into those buffers before replay. Different shapes or branches may require another captured variant or an ordinary execution path; supported graph updates are implementation-dependent.

Include capture and warmup cost when judging short-lived jobs. For serving, also account for memory reserved by graph variants and padding wasted by buckets. Inspect the CPU/GPU timeline first: replay is compelling when dispatch gaps matter, but it may barely change a workload already dominated by large compute-bound kernels.
@@ 867
**An AllReduce makes every worker receive the sum of everybody's gradients.** A ring accomplishes this without requiring one central worker to receive and redistribute the whole tensor. Split each N-byte tensor into P equal chunks, then pass chunks between neighbors. [1]

During reduce-scatter, each received chunk is combined with local contributions as it travels. After P-1 rounds, each rank owns one fully reduced chunk. During all-gather, those completed chunks circulate for another P-1 rounds until every rank has the full reduced tensor.

**Count a four-worker example.** If N=400 MB, each chunk is 100 MB. Each worker sends three chunks during reduce-scatter and three during all-gather: 600 MB total. The formula gives $2(4-1)400/4=600$ MB. Each worker also receives 600 MB. Whether a bandwidth accounting convention reports sent bytes or sent-plus-received bytes must be stated.

As P grows, sent volume approaches 2N, which explains the attractive bandwidth scaling. It does not mean runtime is independent of worker count. There are 2(P-1) rounds, each with latency, and physical links may be shared or uneven. Small messages can be dominated by latency rather than bytes. Collective algorithms should therefore be selected for tensor size and network topology, not solely the asymptotic volume formula.
@@ 868
**Communication can begin before all gradients exist.** Backpropagation usually computes the output-layer gradients first, then works toward earlier layers. If distributed training waited for the entire backward pass before synchronizing anything, compute and communication would occupy separate blocks of time. Bucketing permits a ready group of gradients to be reduced while other gradients are still being computed. [1]

Imagine backward takes 20 ms and synchronization takes 8 ms. Fully serial execution needs about 28 ms. If most of the communication runs during those 20 ms, only its uncovered tail adds to the step. This is an illustrative schedule; real overlap depends on readiness and contention.

**Bucket size changes when communication can start.** A huge bucket waits for its last gradient, delaying the reduction. Extremely small buckets become ready early but launch many latency-heavy collectives. The best compromise depends on layer sizes, compute order, and link speed. Reverse layer order is a useful intuition because it follows typical backward readiness, although actual framework bucket construction can adapt or differ.

All workers must participate consistently, and the optimizer must wait for required reductions before using gradients. Unused parameters or divergent branches can disrupt readiness. Read a timeline to identify the exposed tail and tune based on measured overlap; asynchronous launch alone does not prove that communication is hidden.
@@ 869
**A collective is a contract about what each worker starts with and ends with.** AllReduce combines corresponding entries and gives every worker the complete result. ReduceScatter gives each worker only its assigned part of the combined result. AllGather distributes existing parts to everyone without adding them. AllToAll routes different pieces to different destinations. [1]

Take two workers with vectors (1,2,3,4) and (10,20,30,40). A sum AllReduce leaves both with (11,22,33,44). A ReduceScatter into two contiguous chunks leaves worker zero with (11,22) and worker one with (33,44). AllGather of those chunks reconstructs (11,22,33,44) on both workers.

That example proves the composition's meaning: first compute and partition the sum, then distribute its partitions. In general, let $s_j=\sum_r x_j^{(r)}$ denote reduced chunk j. ReduceScatter places $s_j$ on worker j; AllGather returns the concatenation of all $s_j$ everywhere.

**AllToAll solves a different problem.** In expert routing, worker zero may send some tokens to expert owner one and other tokens to expert owner two. Those tokens are not added elementwise. Matching the collective to data ownership avoids unnecessary replication. The semantic equality between AllReduce and the two-stage composition does not promise identical floating-point summation order or the fastest implementation on every topology.
@@ 870
**Sharding saves memory by making each worker own only part of the model state.** In ZeRO-3 or fully sharded training, a worker temporarily gathers the parameters needed to execute a module, uses them, and may release the full copy afterward. This creates communication that a permanently replicated parameter copy would avoid. [1]

A simplified accounting uses one unit for a full parameter-sized gather or a gradient-sized reduce-scatter. If parameters must be gathered separately for forward and backward, the step has two gathers plus one gradient reduction: three units. A comparison schedule with one corresponding gather/synchronization plus one reduction has two units, giving 3/2=1.5. This explanation requires matching tensor sizes and communication conventions.

**Why the ratio changes.** Keeping full parameters after forward avoids a second gather but consumes memory. Different dtypes change bytes per parameter or gradient. Module wrapping and sharding policies alter both peak memory and the communication schedule. The quoted ratio is a model under stated assumptions, not a constant of FSDP.

Backward prefetch requests the next module's weights while the current module computes. Prefetch too late and the GPU waits; prefetch too aggressively and multiple full modules coexist, possibly causing an out-of-memory error. Tune module granularity and prefetch depth jointly using a timeline and peak-memory trace. The goal is a step that both fits and overlaps useful work.
@@ 871
**A pipeline takes time to fill and drain.** If four devices each hold one consecutive group of layers, the last device cannot begin the first microbatch until earlier devices process it. Near the end, some devices finish while others still have work. Those idle periods are the pipeline bubble. [1]

For p balanced stages and m microbatches, a simplified schedule has m useful slots and p-1 overhead slots. With p=4 and m=12, overhead relative to useful work is 3/12=25%, while idle fraction of total slots is 3/15=20%. The question's formula uses the first denominator; calling both quantities the same fraction creates confusion.

**Why 1F1B helps memory.** A forward pass saves activations required by backward. GPipe's forward wave can accumulate activations for many microbatches before their backward passes run. After warmup, 1F1B alternates forward and backward work so an older microbatch's saved activations can be released earlier. It changes how long activations remain live, even when idealized bubble overhead is similar.

Increasing microbatch count can reduce relative bubbles, but smaller microbatches may make matrix operations less efficient. Uneven stages also create idle time that the balanced formula misses. Choose partitions and microbatch sizes from both timing and memory measurements, including communication between stages. A beautiful scheduling formula does not compensate for one stage taking twice as long as the others.
@@ 872
**Experts live on different devices, so tokens must travel to the experts chosen for them.** Each token initially belongs to a worker processing part of the batch. Its router may select experts elsewhere. Dispatch sends token activations to those owners; combine returns expert results to the original tokens for weighted aggregation. This is a personalized exchange, naturally implemented with AllToAll-like communication. [1]

Imagine eight workers and an expert that receives 40% of tokens. Even if total traffic is moderate, that expert's worker becomes a hotspot while other workers wait. With top-k routing, a token may need several expert computations, increasing dispatch volume. Routing imbalance can therefore waste compute and stress particular network paths simultaneously.

AllReduce often moves regular chunks through a well-structured schedule. Expert routing can produce uneven, many-to-many transfers whose demands change by batch. Cross-node links and the network's bisection bandwidth may become bottlenecks. However, “AllToAll is always harder” is too broad: topology, message size, expert placement, and load distribution determine the actual cost.

Measure expert load, communication time, dropped or overflow tokens, and achieved tokens per second. Load balancing, capacity policies, topology-aware placement, and overlap address different symptoms. More experts increase available parameter capacity; they do not automatically increase throughput or remove the need to provision the network for routing.
@@ 873
**Divide the sequence, but still let each query see every permitted key.** Each worker keeps a block of query positions. Key/value blocks circulate among workers. When a block arrives, the worker computes that block's contribution to its local query outputs and combines it with contributions already processed. [1] [2]

The subtlety is softmax normalization. Softmax over each block separately and then averaging the outputs would be wrong because blocks do not generally have equal total probability mass. Instead maintain a running maximum, a rescaled exponential sum, and a rescaled value-weighted numerator.

**A tiny example.** If one query has scores 0 in the first block and log(3) in the second, the global probabilities are 1/4 and 3/4. Treating each single-element block as a complete softmax would give probability one within both blocks, losing their relative weights. Online softmax retains that cross-block normalization and returns the same dense attention result up to numerical rounding.

Sharding reduces per-device activation storage and avoids materializing the entire score matrix on one GPU. It does not remove dense attention's quadratic number of query-key interactions. Global position-aware masking, backward communication, and enough computation to overlap transfers are essential. A million-token sequence fitting across devices is a memory achievement; whether training it is affordable remains a separate compute and network question.
@@ 874
**A timeout tells you workers stopped making collective progress; it may not identify the original failure.** If rank 37 crashes before entering an AllReduce, other ranks can wait until their watchdogs report timeouts. The most informative log may therefore precede the timeout on a different machine. [1]

Start with step identifiers and timestamps across ranks. Find the earliest exception, missing collective, or unusually long operation. Compare tensor shapes and collective order. A rank with one fewer data batch may leave the loop while everyone else enters another synchronization; increasing the timeout cannot fix that mismatch.

**Separate software mismatch from a slow or broken component.** A long dataloader gap suggests input starvation. Reduced clocks or high temperatures suggest throttling. Communication diagnostics and network counters can reveal a faulty path. Preserve evidence before restarting, then run isolated tests to narrow the cause instead of changing several settings at once.

At large scale, one failed worker can stall the whole synchronous job. Stop or recover workers coherently and resume from a checkpoint containing the state required for consistent progress, including optimizer and data position where applicable. Checkpoint frequency trades recovery work against checkpoint overhead. A larger timeout is appropriate only when a legitimate slow operation is understood; otherwise it can simply turn a visible failure into a longer wait.
@@ 875
**Two different optimizations are being combined.** DataLoader workers prepare batches concurrently on CPUs. Pinned memory provides host buffers suitable for efficient direct transfer to the GPU. Increasing workers addresses preparation time; pinning addresses the transfer path. Neither replaces the other. [1] [2]

Think of three stages: decode/transform a batch, transfer it, then compute on it. If each waits for the previous stage to finish, their times add. A pipeline can prepare batch three while transferring batch two and computing batch one, provided buffers, streams, and dependencies allow it.

Pinned memory cannot be paged out by the operating system. Nonblocking copies can help overlap transfers with other work, but the consumer must still wait before using the transferred data. Calling a synchronization immediately after every copy eliminates much of the intended overlap.

**Tune the actual bottleneck.** If storage is saturated, adding workers may increase contention. If CPU transforms dominate, more workers may help until cores or memory bandwidth are exhausted. Pinning excessive amounts of memory can pressure the host. Profile time waiting for the next batch, host-to-device transfer, and GPU computation separately. Also verify worker sampling: duplicated records or incorrect sharding can make throughput look better while changing the training data.
@@ 876
**False sharing is a hardware collision between logically independent writes.** Two workers may update different counters that happen to occupy the same CPU cache line. Cache coherence transfers ownership of the whole line, so the workers repeatedly invalidate each other's copies even though they never intentionally share a variable. [1]

For example, worker A increments one progress counter while worker B increments its neighboring counter millions of times. Padding the counters onto separate lines or aggregating counts locally can reduce coherence traffic. Padding every structure indiscriminately wastes memory; first establish that frequent writes to shared lines are a meaningful cost.

**NUMA is about where memory physically lives.** On a multi-socket machine, memory attached to the local CPU socket is generally cheaper to access than memory reached through another socket's interconnect. A preprocessing worker on one socket feeding a GPU attached to another may create avoidable remote traffic. Allocation placement matters as well as CPU affinity.

Inspect CPU/GPU topology and measure locality, bandwidth, and worker stalls. Bind workers and allocate buffers near their consuming GPU when the evidence supports it. False sharing and NUMA misalignment are distinct: one is cache-line ownership contention, the other is remote-memory access. Both can leave GPUs hungry even when aggregate CPU utilization looks high, but neither explains every slow input pipeline.
@@ 877
**Four-bit weights primarily reduce the amount of weight data that must move.** Compared with 16-bit weights, ideal four-bit storage uses one quarter as many payload bytes, before scales and metadata. At small batch sizes, where weights are read for little arithmetic reuse, that reduction can be very valuable. W4A16 typically retains higher-precision activations and must interpret or dequantize packed weights for computation. [1] [2]

For a simplified matrix multiply, each weight contributes about two FLOPs per input vector. With batch size b and BF16 weights, weight-only arithmetic intensity is approximately b FLOPs per byte. As b increases, the same weights support more arithmetic, so the bottleneck can move from reading weights toward executing the matrix multiply.

**That changes which quantization helps.** A W4A16 implementation may save memory traffic but add unpacking or dequantization work. If arithmetic dominates, those costs can cancel the benefit. W8A8 or FP8 may also accelerate supported arithmetic, subject to hardware and kernel details.

There is no defensible universal batch-size crossover. Shapes, padding, quantization groups, kernel fusion, device generation, and prefill/decode mix all matter. Benchmark the intended workload and quality constraints. Lower memory use can also enable more concurrent requests, so evaluate total serving throughput and latency rather than only one isolated matrix multiplication.
@@ 878
**Quantization needs a ruler that maps real values to a small integer range.** The scale is the ruler's spacing; the zero point locates real zero when an asymmetric mapping is used. Static activation quantization chooses that ruler from calibration examples in advance. Dynamic activation quantization measures the current activation range and chooses it during execution. [1]

Imagine calibrating a symmetric quantizer for values mostly between -1 and 1, then encountering a deployment activation of 8. A fixed narrow range clips that value. Expanding the range to include 8 avoids clipping but gives coarser steps for the many small values. This is the range-versus-resolution tradeoff.

Dynamic scaling adapts to the current batch or token, but an isolated outlier can still make a shared scale coarse. More granular scales, such as per-channel or per-token, change both accuracy and implementation cost. Dynamic min/max reductions also add runtime work; they are not free adaptation.

Use calibration examples matching deployment preprocessing, sequence lengths, and important rare cases. Measure clipping and task quality on separate validation data. “Calibration” here means estimating quantizer ranges, not making predicted probabilities statistically reliable. Verify the deployed kernel's scale conventions and supported granularities, because a mathematically attractive quantizer may be slow or unsupported on the target hardware.
@@ 879
**QAT lets the model practice under the numerical restrictions it will face later.** During the forward pass, fake quantization rounds and clips values as the deployment quantizer would. The model still keeps trainable high-precision parameters so small updates can accumulate. This differs from training entirely in full precision and quantizing only afterward. [1]

Suppose the quantization step is 0.1. Values 0.21 and 0.24 both round to 0.2. Small changes within that interval do not change the quantized output, so the true derivative of rounding is zero almost everywhere and undefined at transitions. Using that derivative literally would prevent useful gradient learning through the operation.

**The straight-through estimator deliberately uses a surrogate.** It often treats the derivative as one inside the supported range, passing the upstream gradient to the underlying value. Some versions suppress gradients outside clipping limits. This is not the exact derivative of rounding; it is a biased optimization device that allows parameters to adapt to quantization effects.

Check that training simulates deployment's rounding, clipping, scale granularity, and accumulator behavior closely enough. Test the exported low-precision model, not only the fake-quantized training graph. A good training metric can hide differences in the eventual inference kernel. QAT can recover accuracy, but it cannot guarantee arbitrary bit-width reductions without a task-quality cost.
@@ 880
**The factor six comes from counting three matrix operations, each with multiply-and-add work.** For a linear layer $Y=XW$, forward computes XW. Backward computes an input gradient using W and a weight gradient using X. Under the usual dense approximation, each of these operations costs roughly two FLOPs per parameter per token: one multiply plus one addition. [1]

Thus forward contributes about 2ND and backward about 4ND, giving 6ND for N relevant parameters and D tokens. This is an approximation to useful model computation, not a complete record of every device instruction.

**A worked utilization example.** Suppose N is one billion and the training system processes 10,000 tokens per second. Estimated useful work is $6\times10^9\times10^4=6\times10^{13}$ FLOPs/s, or 60 TFLOPs/s. If the aggregate compatible peak is 200 TFLOPs/s, estimated MFU is 60/200=30%. The peak must use the same precision and dense/sparse convention as the calculation.

Checkpoint recomputation can keep hardware busy without increasing useful model-token throughput. Padding and optimizer work also complicate comparisons. Long-sequence attention and sparse expert activation can make a simple total-parameter approximation inaccurate. Report the FLOP model, token-count convention, timing window, and hardware denominator alongside the percentage. MFU is useful for diagnosing efficiency only when those definitions are consistent.
''')
