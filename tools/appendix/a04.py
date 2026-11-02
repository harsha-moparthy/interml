from author import write
write(r'''
@@ 737|optim dl resnet reg d2l
Backpropagation multiplies weight matrices and activation derivatives across layers. Sigmoid derivatives are at most $1/4$, so repeated multiplication can rapidly shrink gradients; tanh also saturates. ReLU has derivative one on positive inputs, but zero on negative inputs. [1]

$$\frac{\partial\mathcal L}{\partial x_0}=J_1^TJ_2^T\cdots J_L^T\frac{\partial\mathcal L}{\partial x_L}.$$

A unit pushed negative for every training example receives no ordinary ReLU gradient and can remain dead. Large learning rates and unfavorable bias/initialization contribute. ReLU does not control weight-matrix singular values, so exploding/vanishing gradients can still occur. Use variance-aware initialization, residual paths, normalization where appropriate, and activation/gradient monitoring; leaky ReLU keeps a nonzero negative-side slope.
@@ 738|glu optim dl d2l reg
ReLU is $\max(0,x)$; leaky ReLU replaces the negative half with $ax$; ELU uses $\alpha(e^x-1)$ below zero. GELU is $x\Phi(x)$ and SiLU is $x\sigma(x)$, providing smooth gates rather than a hard threshold. [1]

Smooth gates retain small negative outputs and can work well in transformer feed-forward blocks, especially gated variants such as SwiGLU. This is empirical, not a general proof that smoothness implies better generalization. ELU can saturate negatively; ReLU is cheap but may die; GELU/SiLU involve more complex evaluation that kernels may fuse. Compare training stability, model quality, and actual inference cost at matched parameter/FLOP budgets rather than choosing solely from activation plots.
@@ 739|resnet prenorm optim dl d2l
A residual block learns a correction to its input. When the correction is unnecessary, setting it near zero approximates identity without forcing many nonlinear layers to reproduce identity individually. The backward signal contains a direct path. [1]

$$y=x+F(x),\qquad \nabla_x\mathcal L=\nabla_y\mathcal L+J_F(x)^T\nabla_y\mathcal L.$$

This makes optimization easier but does not prove an unimpeded gradient through arbitrary depth: the two terms can cancel, and successive blocks can amplify directions. Dimension-changing skips need projections, whose derivative is not identity. Normalization placement and residual scaling also matter. Distinguish optimization degradation—deeper models fit training data worse—from ordinary overfitting, where training improves but held-out performance worsens.
@@ 740|bn norm prenorm torch d2l
For NCHW images, BatchNorm computes statistics per channel across batch and spatial axes; InstanceNorm uses spatial axes per sample/channel; GroupNorm uses spatial axes and a channel group per sample. LayerNorm normalizes a chosen feature shape per example or token. [1] [4]

$$\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon},\qquad y=\gamma\hat x+\beta.$$

Small batches make BN estimates noisy, but batch size one is not automatically invalid if there are many spatial observations. A channel with only one training value cannot supply meaningful variance. Autoregressive models need consistent per-token behavior and must avoid leaking future positions through statistics. LN/RMSNorm avoid dependence on other examples; GN offers a batch-independent alternative for vision. Check train/eval running-stat behavior explicitly.
@@ 741|bn torch resnet d2l dl
BatchNorm with affine parameters learns one scale and one shift for each of $C$ channels, hence **$2C$ trainable parameters**. Running mean and variance are buffers, not gradient-trained parameters. Some implementations also store a batch counter. [1] [2]

For a convolutional bias $b_c$ constant over the normalized elements, centering cancels it:

$$\frac{(z+b_c)-(\mu_z+b_c)}{\sqrt{\sigma_z^2+\epsilon}}=\frac{z-\mu_z}{\sqrt{\sigma_z^2+\epsilon}}.$$

BN's learned shift supplies the channel offset, so an immediately preceding convolution typically uses no bias. This reasoning assumes compatible normalization and a correctly maintained train/eval state; it does not apply blindly across an intervening nonlinearity or to RMSNorm, which does not center.
@@ 742|optim adamw torch d2l reg
SGD moves against the current gradient; momentum averages a velocity to reduce oscillation and accelerate consistent directions. Nesterov momentum evaluates or approximates the gradient at a look-ahead point. RMSProp divides by a running root mean square of gradients. Adam combines first and second moments with bias correction. AdamW decouples parameter decay from those moments. [1] [2]

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad \theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

Adaptive methods handle coordinate-scale differences but require extra state. SGD's lower state cost and different implicit bias can be useful. Compare tuned schedules, regularization, compute, and validation performance; optimizer names alone do not determine quality.
@@ 743|adamw optim reg torch d2l
Adding $\lambda\|\theta\|^2/2$ to the objective changes the gradient to $g+\lambda\theta$. Adam then feeds that combination into both moment estimators, entangling shrinkage with gradient history and coordinatewise preconditioning. Decoupled decay applies shrinkage separately. [1]

$$\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

If the adaptive denominator is large, a coupled penalty's immediate contribution is attenuated; the full dynamics are more complex because the penalty also changes the moments. Ordinary SGD makes L2 and weight decay equivalent under matched conventions, but adaptive optimization generally does not. Biases and normalization scales are often excluded by deliberate parameter groups; this is a tunable practice rather than an algebraic necessity.
@@ 744|adamw optim reg attention resnet
SGD with momentum sometimes performs well on vision benchmarks, while AdamW is widely effective for transformers. Neither observation makes AdamW mathematically mandatory or SGD universally better at generalization. Architecture, data, tuning budget, schedules, and regularization confound simple comparisons. [1] [2]

“Flat minima generalize better” is a useful hypothesis with caveats: parameter rescaling can change apparent sharpness without changing the represented function. Adaptive preconditioning and stochastic-gradient noise influence which solutions training reaches, but do not establish a universal ranking.

A fair experiment tunes learning rate and decay independently for each optimizer, matches training compute and augmentation, and measures held-out quality across seeds. Include optimizer-state memory and convergence speed when selecting for a production training budget.
@@ 745|optim adamw scaling torch d2l
For SGD, $k$ small steps with similar gradients total roughly $-k\eta g$. Replacing them with one update from a batch $k$ times larger motivates learning rate $k\eta$. The approximation fails when parameters and gradients change substantially across those skipped steps. [1]

Square-root scaling for adaptive optimizers arises under particular stochastic-dynamics assumptions; it is not a universal Adam rule. Warmup limits unstable early updates while moment estimates and network statistics settle. Beyond a critical batch size, larger batches yield diminishing statistical efficiency and fewer updates per epoch.

Track global batch, accumulation, update count, schedule units, and gradient-noise behavior. Retune when scaling hardware; keeping the same number of epochs does not preserve the same optimization trajectory or training budget.
@@ 746|amp torch cuda scaling optim
FP32 has 8 exponent and 23 stored fraction bits; FP16 has 5 and 10; BF16 has 8 and 7. BF16 therefore retains FP32-like exponent range but lower precision. TF32 is a tensor-core computation mode with reduced mantissa precision, commonly using FP32 storage. FP8 formats trade still fewer exponent/fraction bits, with format-specific rules. [1] [3]

FP16's narrow range can underflow small gradients or overflow activations. Loss scaling multiplies the loss before backward, then unscales gradients before clipping and updating. BF16 usually avoids that underflow problem because of its exponent range, but still rounds aggressively. Neither dtype guarantees stability. Keep sensitive reductions or optimizer states in adequate precision and check the actual hardware/operator implementation.
@@ 747|double optim dl prml islp
Classical intuition predicts a U-shaped test-error curve as capacity rises. Around interpolation, a model may fit training noise using an unstable solution, creating a peak. Far beyond that threshold, optimization and implicit/explicit regularization can select smoother or lower-norm interpolating solutions, so test error decreases again. [1] [2]

This is double descent, observed across model size, data size, or training time in different settings. Zero training error is not enough to predict generalization: the chosen solution and data distribution matter. The effect is not guaranteed, and the second descent can disappear under different noise, regularization, or optimization. Plot validation performance across a sufficiently broad capacity range and retain an untouched test set rather than interpreting one overfit model as evidence of the full phenomenon.
@@ 748|lottery optim dl d2l resnet
The lottery ticket hypothesis proposes that a dense randomly initialized network can contain a sparse subnetwork which, when trained from a suitable original initialization, reaches competitive accuracy in similar training time. Finding it commonly involves train–prune–reset cycles. [1]

The claim concerns both connectivity and initialization, not simply that pruning after training leaves a small model. Rewinding to an early checkpoint may work better than resetting exactly to initialization in harder settings. Discovering the subnetwork can itself be expensive, so the existence of a winning ticket does not imply a cheap training recipe.

Evaluate sparsity, accuracy, search/training cost, and actual sparse-kernel speed. Unstructured zeros may reduce stored nonzeros without accelerating dense hardware; structured pruning changes dimensions and is easier to exploit.
@@ 749|distill dl reg torch d2l
A teacher's soft probability vector contains relationships absent from a one-hot label: an image may resemble two dog breeds much more than a truck. Raising temperature softens the distribution and exposes these relative preferences. [1]

$$p_i^{(T)}=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}},\quad \mathcal L=\alpha\mathcal L_{hard}+(1-\alpha)T^2D_{KL}(p_T^{(T)}\|p_S^{(T)}).$$

The $T^2$ factor compensates the usual temperature-induced gradient scaling. Use $T=1$ at ordinary inference unless calibration dictates otherwise. Students can inherit teacher errors and biases; soft targets are useful only insofar as they encode relevant information. Compare against training the student directly under equal data and compute, and assess real deployment latency rather than parameter count alone.
@@ 750|torch amp optim rules mltest
First reproduce on one fixed batch in FP32 and locate the **first** nonfinite tensor, rather than inspecting only the final loss. Check input/target ranges and label validity; then intermediate activations, the custom loss, gradients, and optimizer state. [1] [2]

Use stable primitives such as log-softmax/BCE-with-logits; avoid evaluating invalid operations before masking them. Verify learning rate, initialization, reduction denominators, normalization, and gradient clipping after unscaling. If FP32 works but FP16 fails, inspect overflow and scaler behavior. If both fail at the same operation, precision is probably not the root cause.

After the first bad operation is understood, fix it and rerun the minimal batch before returning to distributed training. Save the triggering data and state as a regression case; blindly reducing learning rate can hide a genuine formula or data error.
@@ 751|calibration metrics distill optim scikit
Accuracy considers only which class has the largest probability; cross-entropy also measures confidence. A few increasingly confident mistakes can dominate average negative log likelihood even as more borderline examples become correctly classified. [1] [2]

$$\mathcal L_i=-\log p(y_i\mid x_i).$$

Changing a wrong example's true-class probability from 0.1 to 0.001 increases its loss from about 2.30 to 6.91 without changing its correctness. Inspect confidence distributions, loss by slice, reliability diagrams, and mislabeled/outlier examples. Select checkpoints according to the actual deployment objective; probability-sensitive decisions may prefer lower log loss. Temperature scaling fitted on separate validation data can improve calibration, but cannot fix incorrect rankings or data leakage.
@@ 752|torch optim rules mltest d2l
Freeze a tiny deterministic batch, disable augmentation/dropout, and attempt to memorize it. Confirm features align with labels, the loss expects logits or probabilities as supplied, and gradients reach trainable parameters. [1] [2]

Inspect gradient norms and actual parameter differences before/after `step()`. Check that parameters belong to the optimizer, `requires_grad` is enabled, no accidental detach/no-grad context cuts the graph, the learning rate is sensible, and zeroing happens before the backward pass rather than between backward and step. Verify loss reductions and target dtype/shape.

If a tiny batch cannot be fit, extra data and a larger model usually obscure the bug. Once it can, restore regularization, data loading, and distributed features one at a time to identify the component that breaks learning.
@@ 753|reg bn metrics torch optim
Four common explanations are: training uses dropout or harder augmentation; training logs include a regularization penalty excluded from validation; training loss is averaged while parameters improve throughout the epoch but validation uses the final parameters; or the validation distribution is easier. [1] [2]

A fifth possibility is leakage, which is a defect rather than reassuring evidence. Also verify matching reduction conventions and `train()`/`eval()` behavior, especially BatchNorm running statistics.

To compare fairly, evaluate both datasets with the same frozen checkpoint, no training-only augmentation, the same data loss, and evaluation mode. If the gap remains, examine class balance, example difficulty, duplicates, temporal splits, and confidence intervals. Lower validation loss alone neither proves excellent generalization nor proves a bug.
@@ 754|torch scaling bn optim cv
With per-device batch unchanged, eight-way DDP increases the global batch eightfold and reduces optimizer updates per epoch by roughly eightfold. The same epoch count and learning-rate schedule therefore describe a different optimization process. [1] [2]

Check global batch including accumulation, learning-rate/warmup tuning, scheduler units, gradient averaging versus summation, and whether the distributed sampler changes shuffle each epoch. Ensure evaluation aggregates all examples without accidental duplicates or dropped tails. BatchNorm uses local statistics unless synchronized; its effective normalization batch may differ from the global optimization batch.

Compare first with a configuration preserving the original global batch and update count. Then change batch size deliberately and retune. There is no single setting guaranteed to recover exactly the missing three percentage points.
@@ 755|torch d2l dl resnet vit
For each spatial axis, dilation makes the effective kernel $K_{eff}=D(K-1)+1$. With symmetric padding and floor-based convolution output sizing, [1]

$$H_{out}=\left\lfloor\frac{H+2P-D(K-1)-1}{S}\right\rfloor+1,\quad W_{out}=\left\lfloor\frac{W+2P-D(K-1)-1}{S}\right\rfloor+1.$$

For groups $G$, trainable weights are $C_{out}(C_{in}/G)K^2$, plus $C_{out}$ if bias is enabled. With 3 input channels, 16 output channels, 3×3 kernels, stride 2, padding 1, dilation 1, and 32×32 input, output is 16×16×16 and parameters are $16\cdot3\cdot9+16=448$. Asymmetric padding and framework-specific “same” rules need separate accounting.
@@ 756|resnet d2l torch dl vit
A 1×1 convolution mixes channels independently at each spatial location. It can reduce width before an expensive spatial convolution, expand width afterward, project residual dimensions, or create new channel features with a following nonlinearity. [1]

$$y_{n,o,h,w}=\sum_cW_{o,c}x_{n,c,h,w}+b_o.$$

It does not directly combine neighboring pixels when stride is one. In a bottleneck, reducing $C$ channels to $r$ makes the central 3×3 convolution cost $9r^2$ rather than $9C^2$, with projection costs $Cr+rC$. If $r=C/4$, the saving can be large. Too narrow a bottleneck loses information, and stride greater than one also subsamples spatial positions. Treat it as a shared per-pixel linear map, not a spatial averaging operation.
@@ 757|torch d2l dl resnet scaling
A standard convolution uses $K^2C_{in}C_{out}$ weights. Depthwise convolution filters each input channel separately with $K^2C_{in}$ weights, followed by a 1×1 pointwise mix with $C_{in}C_{out}$ weights. Ignoring biases and equal output spatial size, [1]

$$\frac{cost_{sep}}{cost_{standard}}=\frac{1}{C_{out}}+\frac{1}{K^2}.$$

For $K=3,C_{out}=256$, the ratio is about 0.115, or 8.7× fewer multiply-accumulates. This factor does not automatically translate into latency: depthwise kernels can have low arithmetic intensity, and extra launches/reads matter. The factorization restricts representational structure; intermediate nonlinearities and additional layers affect both quality and the fair comparison.
@@ 758|d2l torch resnet dl vit
For stride-one, dilation-one convolutions, each 3×3 layer increases the theoretical receptive-field width by two. Three layers therefore give width seven. More generally, track receptive field $r$ and input-space jump $j$: [1]

$$r_l=r_{l-1}+(K_l-1)D_lj_{l-1},\quad j_l=j_{l-1}S_l.$$

With constant width $C$ and no biases, three 3×3 layers use $27C^2$ weights versus $49C^2$ for one 7×7 layer, while inserting more nonlinear transformations. This is not equality of the represented function. Padding, strides, and varying widths change the calculation. The effective receptive field—the input region with substantial influence after training—can be much smaller than the theoretical support.
@@ 759|detr focal vit torch d2l
Two-stage detectors first propose regions and then classify/refine them. Single-stage detectors predict classes and boxes densely in one pass; focal loss addresses the many easy background candidates. DETR uses a fixed set of learned object queries and matches predicted objects to ground truth during training. [1] [2]

$$\hat\sigma=\arg\min_\sigma\sum_i\mathcal C(y_i,\hat y_{\sigma(i)}).$$

The matching cost combines class and box terms. Each target matches one prediction; unmatched predictions learn “no object,” discouraging duplicates without conventional NMS. This avoids hand-designed anchor assignment in original DETR, but does not imply every later transformer detector uses identical mechanisms. Compare small-object performance, convergence, latency, and recall; architecture family alone does not determine those outcomes.
@@ 760|unet detr focal vit d2l
Semantic segmentation assigns a class to each pixel, so two people may share one class region. Instance segmentation distinguishes individual objects and supplies a mask for each. Panoptic segmentation combines uniquely identified “thing” instances with semantic “stuff” regions such as road or sky. [1]

Encoder downsampling captures context but loses fine spatial detail. U-Net skips pass high-resolution encoder features to the decoder, helping recover boundaries and localization. They do not automatically guarantee precise masks: alignment, sampling, loss, and annotation quality matter.

Use IoU/Dice for semantic masks, detection-style mask AP for instances, and panoptic quality for panoptic output. Evaluate thin structures and small objects separately; a good aggregate score can hide systematic boundary errors.
@@ 761|dl d2l optim resnet torch
An RNN repeatedly applies a recurrent weight matrix and activation. Backpropagation through time multiplies their Jacobians, so directions can shrink or grow exponentially with sequence length. LSTM introduces an additive cell-state path controlled by gates. [1] [2]

$$c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad \frac{\partial c_t}{\partial c_{t-1}}\bigg|_{gates}=f_t.$$

When forget gates remain near one, information and direct gradients can persist. The qualification “holding gates fixed” matters: the full derivative also includes gate dependencies. LSTM does not eliminate all vanishing/exploding gradients, and saturated gates can block learning. Initialization, clipping, sequence truncation, and task length remain important. The additive path makes retention learnable instead of forcing it through a new nonlinear transformation at every step.
@@ 762|ctc rnnt whisper attention d2l
CTC predicts frame-level label/blank distributions and sums over alignments that collapse to the same transcript. Its conditional-independence assumption permits parallel frame scoring; decoding with a language model or beam can still be sequential. RNN-T adds a label-history prediction network and marginalizes time/label alignments, making it suitable for streaming. Whisper uses an autoregressive attention-based decoder over audio features. [1] [2] [3]

$$p_{CTC}(y\mid x)=\sum_{\pi:\mathcal B(\pi)=y}\prod_t p(\pi_t\mid x).$$

On silence or weak audio, an autoregressive language prior can dominate and generate plausible unsupported text. Voice activity detection, no-speech thresholds, repetition checks, and appropriate decoding mitigate this; they do not prove every transcript is grounded. Compare latency, alignment assumptions, robustness, and language-model integration.
@@ 763|gcn dl d2l optim scikit
Repeated graph propagation averages neighboring representations. On a connected graph, repeated multiplication by a normalized propagation operator can suppress differences and push nodes toward a low-dimensional stationary subspace. Then different classes become hard to distinguish. [1]

$$H^{(l+1)}=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W_l).$$

There is no universal failure threshold at four or five layers; graph structure, spectral properties, features, and architecture determine it. Residual/jumping connections, normalization, controlled propagation depth, or decoupled feature transformation can help. Distinguish over-smoothing from over-squashing, where many distant signals are compressed through narrow graph bottlenecks. Measure pairwise representation distances and task performance as depth grows rather than diagnosing either problem from depth alone.
''')
