from author import write
write(r'''
@@ 881|librosa whisper ctc rnnt scipy
Decode PCM into floating amplitudes, resample to the model's expected rate, and apply the intended channel handling. Split into overlapping frames, multiply by a Hann window, take an STFT, convert magnitudes to power if required, apply triangular mel filters, then log-compress with a finite floor. [1]

$$S_{m,t}=\log\left(\epsilon+\sum_kH_{m,k}|X_{k,t}|^2\right).$$

Window length controls frequency/time resolution; hop controls frame rate; mel filters approximate a perceptual frequency scale. The exact log base, normalization, centering, padding, and power convention must match training. A spectrogram is not simply an image with arbitrary resizing: changing its axes changes time/frequency meaning. Preserve silence and clipping diagnostics rather than normalizing away every audio-quality problem.
@@ 882|whisper ctc metrics rnnt mltest
WER counts substitutions, deletions, and insertions equally relative to reference word count. It can be low while a crucial entity, negation, dosage, or ticker is wrong; common filler words may dominate the denominator. [1]

$$WER=(S+D+I)/N.$$

Report ordinary WER for comparability, plus entity/number accuracy, negation errors, and task-specific weighted consequences. Keep normalization rules explicit: punctuation, casing, scripts, and number expansion can change scores substantially. Evaluate noisy, accented, and code-switched slices, with human review for consequential errors. Semantic similarity alone is insufficient because “fifteen” and “fifty” can be embedding-near but operationally very different. Separate transcription fidelity from downstream summarization or decision accuracy.
@@ 883|ctc whisper irl scipy metrics
Use edit-distance dynamic programming over reference tokens $r_1\ldots r_n$ and hypothesis tokens $h_1\ldots h_m$. Initialize $D_{i,0}=i,D_{0,j}=j$ and compute [1]

$$D_{i,j}=\min(D_{i-1,j}+1,D_{i,j-1}+1,D_{i-1,j-1}+\mathbf1[r_i\ne h_j]).$$

Backtracking counts insertions, deletions, and substitutions; divide their sum by reference length for WER, or use characters for CER. Standard LCS is not identical because it typically models insertions/deletions without unit-cost substitution. Handle empty references explicitly; many insertions can make WER exceed 100%. Fix tokenization and normalization before comparing systems, and specify tie-breaking if reporting the individual error-type counts.
@@ 884|whisper sentencepiece ctc rnnt metrics
Code-switched audio may have multiple acceptable scripts/transliterations, while a multilingual decoder's language/task token can bias output toward one script or toward translation. Training imbalance and an overly strong language prior can replace unusual mixed-language speech with more familiar text. [1] [2]

Use the intended transcription task and language hints without forcing every segment into one language. Evaluate with explicitly permitted alternate references or transliteration-normalized scores, while retaining a strict score for names/numbers where script matters. Semantic embeddings can complement surface metrics but may miss negation, quantities, and hallucinations.

Audit audio-grounded errors with bilingual reviewers. A fluent translation is not a faithful transcript when the requested output is verbatim code-switched speech; keep those tasks and metrics separate.
@@ 885|rnnt ctc whisper attention scaling
CTC assumes frame-label conditional independence given the encoded audio and sums blank/repetition alignments. AED generates text autoregressively with attention to audio. RNN-T combines an audio encoder, label-history predictor, and joint network, allowing emissions as audio arrives. [1] [2] [3]

Transducers fit streaming/on-device requirements because they can align variable-rate audio and labels without waiting for a full utterance, but latency also depends on encoder lookahead, chunking, and endpointing. “Industry standard” is not an exclusive rule: streaming CTC and chunked attention systems are also viable.

Compare first-token/finalization delay, WER, memory, battery, vocabulary biasing, and noise robustness. An offline architecture can score well on WER while failing the product's interactive latency budget.
@@ 886|ctc rnnt whisper sentencepiece d2l
CTC aligns a short label sequence with a longer frame sequence. Its collapse map first merges adjacent repeated labels and then removes blanks. A blank separates consecutive instances of the same character. [1]

For example, frame path `l l` collapses to `l`, whereas `l blank l` collapses to `ll`. Thus “to” and “too” can differ through blank-separated repeated o labels. The blank represents no emitted label at that frame; it is not necessarily physical silence.

Training sums probabilities over all valid paths using dynamic programming rather than choosing one alignment. Feasible alignments require enough frames, especially for repeated labels. Greedy best-path decoding can differ from the highest-probability collapsed transcript because many paths may map to the same output.
@@ 887|rnnt ctc whisper irl agents
Shallow fusion combines acoustic/transducer scores with an external language-model or contextual-bias score during decoding. A lexicon/WFST or phrase trie can reward likely hotword paths; contextual adapters provide learned conditioning when supported. [1] [2]

$$score(y)=\log p_{ASR}(y\mid x)+\lambda\log p_{LM}(y)+\beta\,bias(y).$$

Tune weights on held-out audio: excessive hotword bias inserts terms that were never spoken. Include phonetic variants and negative examples with similar sounds. LLM post-correction can use context but should stay constrained to plausible N-best/audio evidence and preserve uncertainty. User-specific vocabulary must be scoped and versioned. Measure rare-term recall alongside false hotword insertions, latency, and ordinary WER; improving one list of names can damage unrelated speech.
@@ 888|whisper rnnt ctc judge mltest
A post-corrector can turn an uncertain transcript into a confidently wrong sentence, change numbers/negation, or insert contextually plausible words absent from the audio. Added model calls also increase latency and privacy exposure. [1]

Constrain candidates using N-best hypotheses, aligned timestamps, phonetic similarity, and a small edit budget. Preserve high-confidence spans and expose unresolved uncertainty instead of inventing completion. Require evidence for entity/number changes and retain the original transcript for audit.

Evaluate against human transcripts with a metric that separates beneficial corrections from harmful edits, especially on rare names and silence. An LLM judge alone may reward fluency over fidelity. A lower text-only language-model loss is not proof that the corrected words were spoken.
@@ 889|fastspeech vits whisper flow codec
Acoustic-model pipelines predict features such as mel spectrograms, then synthesize waveform with a vocoder. FastSpeech-style duration modeling enables parallel generation; VITS combines variational modeling, flows, and adversarial waveform generation, so it is not simply the same architecture. [1] [2]

Codec language models generate discrete audio-code sequences and decode them to waveform; some use autoregression, others masked parallel refinement. Flow-matching TTS learns a conditional velocity field and integrates from noise to speech representations. These categories can overlap.

Compare intelligibility, prosody, speaker similarity, first-audio latency, real-time factor, and robustness to long or unusual text. Fewer autoregressive steps do not guarantee lower end-to-end latency if iterative refinement or vocoding is expensive. Evaluate consent and intended speaker controls when using voice-conditioning data.
@@ 890|codec prml dl attention flow
An audio encoder produces continuous latent vectors. The first codebook approximates a vector; each subsequent codebook quantizes the residual left by previous stages. The decoder reconstructs audio from the sum of selected code vectors. [1]

$$r_0=z,\quad q_k=\arg\min_{e\in C_k}\|r_{k-1}-e\|^2,\quad r_k=r_{k-1}-q_k,\quad \hat z=\sum_kq_k.$$

The discrete indices form several synchronized token streams. More codebooks increase bitrate and usually fidelity; dropping later stages offers a rate/quality tradeoff when training supports it. Token rate, codebook size, and number of stages determine raw bitrate before entropy coding. Coarse-to-fine residual structure is not guaranteed to map cleanly to human concepts such as words versus timbre.
@@ 891|donut teds vit llava blip
A pipeline detects text regions, recognizes characters, then reconstructs reading order/layout. It exposes intermediate boxes and confidence for debugging but propagates detection/OCR errors. An OCR-free VLM maps document pixels directly to text/structure, simplifying interfaces and using broader context but potentially hallucinating plausible content. [1]

Dense tables are especially sensitive to resolution: downsampling can erase decimal points, minus signs, and superscripts. Tiling preserves detail but must retain page coordinates and cross-tile relationships. Evaluate cell text, row/column structure, numeric fidelity, and evidence localization separately.

For financial documents, verify totals and units against image regions rather than trusting fluent generated HTML. Hybrid systems can combine high-resolution OCR with a layout-aware model and explicit validation; OCR-free does not mean evidence-free.
@@ 892|teds donut vit detr irl
Spanning cells break a simple one-box-per-grid-slot assumption; borderless tables require alignment cues; rotation changes geometry; mixed scripts complicate recognition and reading order. Box-plus-graph methods predict cells and relations such as same-row/column or adjacency, preserving explicit geometry. Sequence methods generate HTML or a compact structural grammar such as OTSL. [1]

Graph methods need consistent relation decoding; sequence methods can produce syntactically valid but geometrically inconsistent tables or drift after one token error. Separate structure from text recognition when diagnosing failures. Validate rowspan/colspan coverage, nonoverlapping cell occupancy, row lengths, and reading order.

Use high-resolution crops with global page context and evaluate difficult table categories separately. A table that looks plausible visually may still attach a value to the wrong header, which is often the consequential error.
@@ 893|teds donut metrics irl prml
TEDS compares predicted and reference table trees using normalized edit cost, reflecting structural changes such as missing cells or wrong spans and, in content-aware variants, cell text differences. A common form is [1]

$$TEDS=1-\frac{EditDist(T_p,T_r)}{\max(|T_p|,|T_r|)}.$$

Bounding-box IoU measures spatial overlap but may assign a good score to a cell attached to the wrong row/header or with an incorrect span. Conversely, tree similarity alone may hide small numeric transcription errors with large business impact. Report structure-only and content-aware variants, text/numeric accuracy, and geometric metrics together. Canonicalize equivalent HTML consistently and specify the edit-cost implementation; raw HTML-string similarity is not the same as DOM-tree correctness.
@@ 894|vit llava donut blip teds
A full page compressed to 224×224 can reduce small characters to a few pixels or less; information lost in resampling cannot be recovered reliably by a language prior. Fixed-size patches further limit spatial granularity. [1]

Dynamic high-resolution systems encode several detailed tiles plus a global thumbnail. The thumbnail supplies layout context, while tiles preserve small text. Retain tile coordinates and overlap so the model can connect lines and cells across boundaries. More tiles increase visual tokens, memory, and language-model cost.

Evaluate text-size and page-density slices, not only natural-image benchmarks. Tiling can duplicate or split content and confuse reading order; explicit document reconstruction or targeted crops may outperform indiscriminate full-page tiling for dense tables.
@@ 895|clip siglip vit blip llava
CLIP trains image and text encoders to align paired examples and separate mismatched ones. At zero-shot classification, text prompts for candidate classes become comparison vectors. [1]

$$L_{i\to t}=-\log\frac{e^{s(i,t_i)/\tau}}{\sum_j e^{s(i,t_j)/\tau}}.$$

A global similarity objective can succeed by recognizing salient objects without precisely encoding counts, relations, or character strings. “Red cube above blue sphere” may share words/objects with an incorrect composition. Training distribution and resolution also matter. It is too strong to say CLIP always fails these tasks, but such weaknesses require direct evaluation. Use region/token-level supervision, higher resolution, or specialized OCR/counting components when exact structure is required.
@@ 896|siglip clip scaling vit optim
CLIP normalizes each image/text score against batch alternatives with softmax. SigLIP trains matched and mismatched pairs using independent sigmoid binary losses, with a learned scale/bias convention. [1]

$$L=-\sum_{ij}\log\sigma(y_{ij}(s_{ij}+b)),\qquad y_{ij}\in\{-1,+1\}.$$

Pairwise losses avoid requiring a global softmax denominator and allow flexible distribution of pair computations. Negative-pair count, sampling, and weighting still matter; sigmoid does not make the full all-pairs cost disappear. Reported advantages at particular batch sizes are empirical, not a guarantee of stability for every configuration. Compare matched compute, data, negative coverage, and downstream retrieval/classification; distributed communication depends on how embeddings and pair blocks are exchanged.
@@ 897|dino mae clip vit reg
SimCLR contrasts augmented views against negatives; MoCo maintains a momentum encoder/queue for negatives. BYOL and DINO use teacher–student consistency with mechanisms that discourage collapse, without explicit negative pairs. MAE reconstructs masked image patches from visible ones. [1] [2]

DINO-style features can organize object-related regions because augmentations, patch interactions, and self-distillation encourage shared semantic structure. Attention maps can highlight objects, but they are not guaranteed segmentation masks or causal explanations. DINOv2 also depends on data curation and its specific training objectives.

Compare linear probing, dense prediction, robustness, and transfer with appropriate heads. A representation good for classification may not preserve all detail required for reconstruction. Evaluate the trained features rather than inferring capability from whether the loss uses negatives.
@@ 898|lora dino sam reg ewc
Freezing a visual backbone preserves its stored weights, while LoRA learns a restricted update or a separate task head. This can reduce overfitting and training memory when downstream data are limited. It does **not** guarantee that the adapted model preserves spatial priors better than carefully regularized full fine-tuning. [1] [2]

Super-resolution needs fine detail, segmentation needs localization, and defect grading may require tiny texture cues; a single adaptation recipe may not suit all three. Compare frozen probes, selected-layer adaptation, LoRA ranks/targets, and full fine-tuning on held-out domains. Track both task quality and retained general features. Keep an untouched base checkpoint and evaluate resolution/normalization compatibility before attributing a poor result to adaptation capacity.
@@ 899|calib vit prml scipy d2l
For rectified pinhole cameras with parallel image planes, $x_L=fX/Z$ and $x_R=f(X-B)/Z$. Subtracting gives disparity $d=fB/Z$, hence depth $Z=fB/d$. Use consistent units: if disparity is in pixels, focal length must be in pixels. [1]

$$\left|\frac{dZ}{dd}\right|=\frac{fB}{d^2}=\frac{Z^2}{fB},\quad \sigma_Z\approx\frac{Z^2}{fB}\sigma_d.$$

Far objects have small disparity, so a fixed pixel error creates rapidly growing depth uncertainty. Calibration error, occlusion, textureless regions, and correspondence mistakes add errors beyond this local approximation. Larger baseline improves depth sensitivity but increases occlusions and matching difficulty. Report uncertainty rather than treating every disparity as equally reliable.
@@ 900|sam detr unet focal calib
Mask R-CNN uses region proposals and per-region mask prediction, offering strong instance localization with proposal overhead. YOLO-style segmentation emphasizes fast single-stage detection/masks. SAM is promptable segmentation with broad transfer, but still needs prompts or an automatic mask proposal/selection pipeline and may be costly at required resolution. [1] [2]

For industrial use, compare end-to-end latency, tiny-defect recall, instance separation, mask-boundary accuracy, and robustness to lighting/material changes on the target hardware. Model-family names do not establish a universal speed ranking.

Pixel masks alone do not yield physical size. Calibrate optics and scale, correct perspective, and use depth or a known measurement plane. Evaluate measurement error in physical units and reject uncertain/occluded parts rather than reporting precise-looking dimensions from unreliable masks.
@@ 901|calib vit prml scipy d2l
Intrinsics map camera-frame rays to image coordinates; extrinsics transform world coordinates into the camera frame. Lens-distortion coefficients model departures from ideal pinhole projection. [1]

$$s\begin{pmatrix}u\\v\\1\end{pmatrix}=K[R\mid t]\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix},\quad K=\begin{pmatrix}f_x&s_k&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}.$$

Radial distortion changes position with radius from the optical axis; tangential distortion captures decentering effects. Calibration fits these using known geometry and multiple views, then checks reprojection error and held-out views. Coordinate conventions, units, resizing/cropping, and rolling shutter matter. A low average reprojection error does not guarantee reliable extrapolation to image corners or a different focus/zoom setting; retain calibration conditions and uncertainty.
@@ 902|bev calib vit detr sam
Early fusion combines sensor measurements/features before final detection; late fusion combines independently predicted objects; BEV fusion projects or attends sensor features into a common ground-plane representation. Camera images offer texture/semantics, while LiDAR supplies geometric range. [1] [2]

A BEV representation supports spatial and temporal reasoning, but needs accurate calibration, timestamps, ego-motion compensation, and uncertainty handling. BEVFormer itself is a multi-camera temporal/spatial attention design, not proof that every BEV model fuses LiDAR. Distinguish it from explicit camera–LiDAR fusion variants.

Test missing sensors, weather, occlusion, range, and timing errors. Late fusion offers modularity but may discard complementary low-level evidence; early/feature fusion can exploit it while becoming more sensitive to alignment and sensor-domain shifts.
''')
