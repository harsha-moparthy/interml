# First-version lesson selection

Date: 2026-10-24.
Select two questions from each of the 36 workbook sections. A section is a chapter in this plan.
The total is 72 deep lessons. All 600 questions remain in the index.
These choices are editorial judgments about importance, conceptual difficulty, and teaching value.
They are not claims about measured interview frequency or source difficulty rankings.
Read each full question contract before authoring its lesson. Preserve its stated scope.

Select prerequisite articles by need. Prerequisites do not count against the two-question allocation.
For example, question 030 needs broadcasting, shape alignment, and singleton-axis articles.
Question 328 needs dot products, softmax, masking, and multi-head attention.
Question 391 needs causal attention, position offsets, and next-token prediction.
Cross-chapter links can refer to an article before its chapter is complete.
Publish the required article before marking its dependent lesson complete.

| Batch | Chapter | Selected question IDs and titles | Reason and teaching approach |
| --- | --- | --- | --- |
| C01 | Pandas: tabular data manipulation | 016: Add a Per-Group Average Column with GroupBy Transform<br>017: Merge DataFrames | Explain group alignment and join cardinality. Trace row identities through grouped transforms and duplicate-key joins. |
| C02 | NumPy: arrays, broadcasting, and vectorization | 027: Diagnose NumPy array mutation behavior<br>030: Pairwise Differences | Explain memory sharing and broadcasting. Use a 3D broadcasting explorer and linked views of array storage. |
| C03 | PyTorch: modules, autograd APIs, and data loading | 054: Implement a Custom `autograd.Function`<br>057: Trade Compute for Memory with Gradient Checkpointing | Explain custom derivatives and saved activations. Trace forward and backward values; compare saved and recomputed nodes. |
| C04 | Linear algebra and numerical computation | 066: Implement the Conjugate Gradient Method for Solving Linear Systems<br>077: Truncated SVD Rank-r Approximation of Weight Updates | Explain iterative solving and low-rank approximation. Show residual directions and rank-dependent reconstruction error. |
| C05 | Probability, statistics, and information theory | 092: Bayesian Inference for Beta-Binomial Model<br>100: Calculate KL Divergence Between Two Multivariate Gaussian Distributions | Explain Bayesian updates and distribution distance. Update a posterior with evidence; compare Gaussian contours and KL terms. |
| C06 | Calculus, manual backpropagation, and an autograd engine | 108: Reduce Gradient Across Broadcasted Dimensions<br>113: Run Reverse-Mode Autodiff | Explain broadcast gradients and reverse differentiation. Trace a computation graph and sum gradients along reused dimensions. |
| C07 | Feature engineering, splitting, and data pipelines | 122: Target Encoding<br>132: Pipeline Composition (Chain Transformers and Estimator) | Explain leakage and fit/transform boundaries. Track training folds and validation rows through a pipeline. |
| C08 | Supervised ML: regression, classification, and kernels | 137: Implement Lasso Regression using ISTA<br>152: SVM Dual Formulation via Coordinate Descent | Explain sparse optimization and constrained dual updates. Show thresholding and coordinate updates beside objective values. |
| C09 | Decision trees and ensemble learning | 162: CART Recursive Tree Builder<br>168: Implement Gradient Boosting Regressor Step | Explain recursive partitioning and residual fitting. Grow tree splits and fit successive residuals on a small dataset. |
| C10 | Unsupervised learning and anomaly detection | 181: Gaussian Mixture Model with EM Algorithm<br>182: Principal Component Analysis (PCA) Implementation | Explain latent assignments and principal axes. Step through EM responsibilities and rotate a covariance ellipse. |
| C11 | ML evaluation, calibration, and production monitoring | 207: Implement Precision-Recall Curve<br>209: Isotonic Regression Calibration | Explain threshold effects and calibration. Move a threshold through ranked scores; merge violating calibration blocks. |
| C12 | Neural networks: activations, losses, normalization, and recurrence | 233: Implement Layer Normalization for Sequence Data<br>241: Implement a Simple RNN with Backpropagation Through Time (BPTT) | Explain normalization axes and recurrent gradients. Highlight reduced axes and trace gradients through time. |
| C13 | Optimization and training stability | 248: Implement Adam Optimization Algorithm<br>249: Implement AdamW Optimizer Step | Explain adaptive moments and decoupled weight decay. Compare update terms on the same tiny parameter trajectory. |
| C14 | Computer vision: images, convolutions, detection, and ViTs | 267: 2D Transposed Convolution<br>287: Implement a Vision Transformer (ViT) Image Encoder | Explain spatial expansion and image-token computation. Trace overlap accumulation and patches through a small ViT. |
| C15 | NLP: tokenization, embeddings, and language-model data | 294: Byte Pair Encoding (BPE) Tokenizer<br>308: Cross-Document Attention Mask Construction | Explain learned token merges and document boundaries. Animate pair counts and construct a block attention mask. |
| C16 | Attention mechanisms: causal, cross, multi-head, and efficient attention | 316: Implement Multi-Head Attention<br>328: Implement cross-attention with different query and key/value sequences | Explain separate heads and separate query/source sequences. Trace projections, scores, weighted values, concatenation, and output projection. |
| C17 | Transformers and LLM architecture | 331: Rotary Positional Embeddings (RoPE)<br>339: Build a Tiny GPT | Explain rotary geometry and full decoder composition. Link rotation equations to vector pairs and trace one tiny GPT step. |
| C18 | LLM fine-tuning, preference optimization, and alignment | 354: LoRA: Low-Rank Adaptation Forward Pass<br>361: Direct Preference Optimization (DPO) Loss | Explain low-rank updates and preference objectives. Show matrix dimensions and the terms of a preference loss. |
| C19 | NLP and LLM evaluation: BLEU, ROUGE, perplexity, and judges | 374: Calculate Perplexity for Language Models<br>379: Pass@k and Majority Voting Evaluation Metrics | Explain token likelihood and sample-based success estimates. Accumulate log probabilities and enumerate small pass-at-k cases. |
| C20 | LLM decoding, KV caches, and serving | 391: KV Cache for Efficient Autoregressive Attention<br>397: Implement Speculative Decoding Verification | Explain cache reuse and speculative acceptance. Generate a short text step by step; compare cached results and acceptance paths. |
| C21 | Quantization and model compression | 409: Asymmetric Uniform Quantization with Zero-Point<br>413: Simplified GPTQ: Sequential Quantization with Error Compensation | Explain zero-points and error compensation. Move values onto quantization bins and propagate residual error. |
| C22 | Distributed ML and performance modeling | 420: Simulate DDP All-Reduce Gradient Averaging<br>429: Classify LLM Prefill vs Decode as Compute-Bound or Memory-Bound | Explain distributed averaging and inference bottlenecks. Track rank-local gradients and compare bytes with operation counts. |
| C23 | CUDA and Triton coding patterns | 433: Tiled Matrix Multiply CUDA Kernel<br>439: Tiled Flash Attention | Explain tiling and online attention accumulation. Trace tile loads, reuse, synchronization, and online softmax statistics. |
| C24 | Reinforcement learning and search | 461: Compute GAE Advantages with Episode Boundaries<br>462: PPO Clipped Surrogate Loss with Clip Diagnostics | Explain temporal credit and clipped policy objectives. Trace episode boundaries and plot the clipped objective by advantage sign. |
| C25 | Generative models and multimodal learning | 469: Implement Variational Autoencoder (VAE) Loss (ELBO)<br>476: Implement DDPM Reverse Sampling Step | Explain latent objectives and reverse diffusion. Separate ELBO terms and trace one numerical denoising step. |
| C26 | Retrieval, recommenders, graphs, and time series | 485: BM25 Ranking<br>499: Purged Walk-Forward Cross-Validation | Explain relevance scoring and temporal leakage. Trace term contributions and move train/test windows with purge gaps. |
| C27 | DINO and self-supervised representation learning | 503: Implement cross-view DINO loss<br>509: Implement the original-DINO training step | Explain cross-view targets and teacher/student updates. Trace crop pairs, stopped gradients, center updates, and teacher EMA. |
| C28 | LoRA-DINOv3 and image restoration | 515: Insert Q/V-only LoRA into a fused QKV layer<br>519: Build a shifted-window attention mask | Explain selective adaptation and shifted windows. Trace Q/V adapter paths and mark valid shifted-window token pairs. |
| C29 | Sparse and late-interaction image retrieval | 525: Implement ColBERT MaxSim with masks<br>528: Implement SPLADE-style pooling and sparsity penalties | Explain late interaction and sparse representations. Highlight token-wise maxima and show pooling with sparsity penalties. |
| C30 | Neural two-stage ranking and learning to rank | 542: Implement LambdaRank pseudo-gradients<br>545: Build and evaluate a two-stage ranking pipeline | Explain ranking gradients and retrieval limits. Swap ranked pairs and trace candidate recall into reranker results. |
| C31 | Amazon demand gaps, forecasting, and large data | 549: Perform an as-of feature join<br>555: Build a small leakage-resistant gap forecast benchmark | Explain time-valid joins and forecast evaluation. Move the observation cutoff and expose future-data leakage. |
| C32 | Modular multi-task LLMs and training correctness | 557: Merge weighted LoRA updates correctly<br>562: Accumulate unequal micro-batches by valid tokens | Explain adapter composition and token-weighted gradients. Compare matrix products and weight unequal micro-batches by valid tokens. |
| C33 | Differential privacy and subset selection | 567: Implement noisy clipped-gradient aggregation<br>573: Run a selected-subset DP-SGD step | Explain clipping, noise, and the scope of a private training step. Show per-example norms and noise; distinguish a step from a privacy guarantee. |
| C34 | BERT bottlenecks and liver segmentation | 577: Distinguish additive noise from a variational bottleneck<br>581: Implement a small complete U-Net | Explain latent bottlenecks and segmentation architecture. Compare latent distributions and trace U-Net shapes and skip connections. |
| C35 | Program repair and durable agent execution | 588: Implement a checkpointed workflow state machine<br>592: Test recovery at every effect/checkpoint boundary | Explain durable state and recovery boundaries. Inject a failure between effects and checkpoints; replay the state trace. |
| C36 | Streaming and serving systems from the resume | 593: Implement a sequential Count-Min Sketch<br>598: Route requests and fail over streamed responses | Explain sketch collisions and streamed failover. Trace hash counters and show the response boundary after the first streamed byte. |

## Teaching method for each selection

Follow [the teaching guide](TEACHING_GUIDE.md).
Use the table above as a starting point. Choose interactions that expose each question's mechanism.
Guide one small example before allowing free exploration.
Connect selected objects across numbers, geometry, equations, and code where applicable.
Use construction, comparison, failure repair, or component removal when it clarifies the concept.
Provide a checked prediction and a new example for independent practice.
Provide authored step help and complete prerequisite articles.
Keep the existing two-question allocation for each chapter.

## Completion check for each batch

Read both original prompts and their PDF solution sections.
Inspect accessible source links and relevant primary references.
Write prerequisite articles and a clear explanation of each task.
Connect the mathematics, worked example, visual state, and solution lines.
Check the worked examples with independent calculations or reference operations.
Provide a prediction exercise and a new case. Check and explain both answers.
Include a useful failure case or controlled comparison.
Verify shared state, linked selections, reverse steps, reset, and input edits.
Verify authored step help and prerequisite links.
Check keyboard controls, reduced motion, and a static alternative.
Record the exact checks and any remaining limitations in `STATUS.MD`.
