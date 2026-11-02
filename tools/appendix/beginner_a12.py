from expand import write
write(r'''
@@ 881
**A waveform tells you amplitude over time; a spectrogram tells you which frequencies are present over time.** Speech contains rapidly changing frequency patterns. Computing one Fourier transform for an entire sentence would lose when those patterns occurred, so the short-time Fourier transform analyzes overlapping windows. [1]

At a 16,000 Hz sample rate, an illustrative 25 ms window contains 400 samples and a 10 ms hop advances 160 samples. Multiply each frame by a window such as Hann to soften its boundaries, then compute its Fourier transform. Magnitude or squared magnitude gives a frequency-energy representation; the expected convention must match the model.

**Mel filters combine nearby frequency bins.** Their spacing allocates frequency resolution according to a perceptually motivated scale. Each filter forms a weighted sum of spectral power. Taking a logarithm compresses large energy differences, and a small floor prevents log(0). The result has mel bands on one axis and frame time on the other.

Longer windows improve frequency resolution while blurring rapid time changes; shorter hops provide more densely sampled frames but more computation. Resampling, channel mixing, padding, normalization, and log conventions are part of the model's input contract. Resizing the result like an arbitrary photograph can change the meaning of both axes. Validate the waveform and preprocessing before blaming the recognizer for a bad transcript.
@@ 882
**WER measures transcription edits, not their consequences.** In a 100-word reference, changing one ordinary filler word and changing the word “not” each count as one substitution. Both give 1% WER, even though removing a negation can reverse the sentence's meaning. [1]

The numerator adds substitutions, deletions, and insertions; the denominator is the number of reference words. This makes WER useful and reproducible for general comparison, but it weights all word errors equally. A transcript with excellent WER can still get the account number, person's name, or crucial quantity wrong.

**Use a complementary scorecard.** Keep ordinary WER, then separately measure named-entity accuracy, number accuracy, negation preservation, and errors that affect the downstream task. For an illustrative appointment sentence, confusing “fifteen” and “fifty” may be much more consequential than omitting “please.” A semantic similarity score can also miss this because the two sentences remain linguistically similar.

Fix normalization rules before scoring. Converting spoken numbers to digits, removing punctuation, changing case, and segmenting words can alter the count. Evaluate accents, noise, and mixed-language speech separately so aggregate performance does not hide a weak slice. Finally, distinguish faithful transcription from a useful summary: a summary may intentionally omit words, while a transcript should represent what was spoken.
@@ 883
**Find the cheapest sequence of edits that turns the reference into the hypothesis.** Dynamic programming avoids trying every possible alignment independently. A cell D(i,j) stores the minimum edits required for the first i reference words and first j hypothesis words. Each cell chooses among deleting a reference word, inserting a hypothesis word, or matching/substituting the two current words. [1]

For reference “the cat sat” and hypothesis “the small cat sits,” one optimal alignment inserts “small” and substitutes “sits” for “sat.” There are two edits over three reference words, so WER is 2/3, approximately 66.7%. Backtracking through the table explains which edits produced the score.

The first row and column represent transforming an empty sequence: j insertions or i deletions. A matching diagonal move costs zero; a substitution costs one. Longest common subsequence alone is not the same objective because its usual formulation does not assign a single unit cost to substitution.

**WER is not bounded by 100%.** A hypothesis can insert many more words than the reference contains. An empty reference needs an explicit convention rather than division by zero. CER uses the same alignment idea on characters, which can be useful when word boundaries are ambiguous. Normalize both texts consistently, and document tie-breaking if separate insertion/deletion/substitution counts matter.
@@ 884
**Code-switching mixes languages within the same utterance; transcription should preserve that choice.** A speaker may use one language's grammar, an English technical term, and a name written in another script. A multilingual recognizer can instead translate the whole phrase or force it into the dominant script, producing fluent text that is not the requested transcript. [1] [2]

Part of the difficulty is acoustic ambiguity, and part is the decoder's language prior. A rare mixed-language phrase may be replaced with a more frequent single-language sequence. Language and task hints can help, but a hint that is too restrictive can suppress the very switching you need to capture.

**Evaluation must decide what counts as equivalent.** If two accepted scripts or transliterations represent the same spoken name, a single-reference WER can overstate the error. Use documented alternative references or a complementary transliteration-normalized score. Also retain strict checks when spelling, script, numbers, or identifiers are part of the required output.

A semantic embedding score may reward a good translation even though verbatim transcription was requested. Compare against the audio with bilingual review on difficult cases, and measure language-switch boundaries, rare terms, and hallucinations separately. Do not silently normalize away meaningful mistakes just to improve the metric. The task definition should specify whether the desired result is verbatim mixed-language text, transliteration, or translation.
@@ 885
**Speech frames and written tokens do not arrive one-for-one.** A vowel may last many frames, and words have variable duration. An ASR architecture must learn which audio regions correspond to which labels while deciding how much future audio it needs. [1] [2] [3]

CTC predicts frame-level label distributions and sums over alignments containing blanks and repetitions. Given the encoded audio, its output factorization is comparatively simple. An attention encoder-decoder instead emits text autoregressively while attending to encoded audio, making label history explicit but often relying on substantial context.

RNN-T combines an audio encoder with a prediction network representing previous labels and a joint network that decides the next emission or blank. The lattice can advance in audio time or emit labels, accommodating variable alignment. This makes transducers a natural fit for incremental recognition.

**Architecture alone does not determine latency.** An encoder that waits for extensive future audio can make a nominally streaming decoder slow. Chunk size, lookahead, endpoint detection, and stabilization of partial transcripts all affect the user's experience. Compare first visible text, finalization delay, accuracy, memory, and energy on the target device. Streaming CTC and chunked attention can also be practical; “industry standard” does not mean there is only one viable architecture.
@@ 886
**CTC separates frame predictions from the final transcript.** Many audio frames can correspond to one written symbol. CTC therefore defines a collapse rule: merge adjacent repeated labels, then remove blank symbols. Multiple frame paths can collapse to the same output, and training adds their probabilities. [1]

For example, the path `t t blank o o` collapses to “to.” The path `t blank o blank o` collapses to “too.” The blank between the two o labels prevents them from merging. Without that separator, a longer run of o predictions still represents one o, not two.

**Blank does not mean acoustic silence.** It means no output label is emitted at that position in the alignment. The network may use blanks during speech while deciding when to emit a token. Physical pauses and alignment blanks are related only indirectly.

Dynamic programming efficiently sums valid alignments instead of requiring exact frame-by-frame labels. Repeated target characters need enough time steps to insert separating blanks, which affects alignment feasibility. Greedy decoding selects the most likely label at each frame and collapses that one path. The most probable transcript can differ because several individually weaker paths may sum to greater probability. This explains why beam-style decoding can improve results even without changing the acoustic model.
@@ 887
**A hotword list tells the recognizer which rare phrases are plausible, not which words were definitely spoken.** A general model may rarely have seen a person's surname or a product identifier. Contextual bias gives those phrases extra support during decoding, especially when their acoustics resemble more common alternatives. [1] [2]

Shallow fusion adds an external language-model score to the ASR score. A phrase trie or finite-state structure can track partial matches and reward completing a relevant term. If the candidate is “Aster Labs,” the decoder can recognize that “Aster” begins a currently relevant phrase rather than treating every word independently.

**Bias has a false-positive cost.** Turning the weight up too far can cause “Aster Labs” to appear when a similar ordinary phrase was spoken. Evaluate negative examples containing phonetic confounders, silence, and unrelated speech. Rare-term recall alone rewards over-insertion; pair it with false hotword insertion rate and ordinary WER.

Use scoped vocabulary for the current user or task, with appropriate updates and expiration. LLM post-correction can exploit context but should stay tied to audio-supported candidates or N-best hypotheses. A contextually likely term is not acoustic evidence. Tune weights on held-out recordings and include end-to-end latency, since a better rare-word score may not justify a much slower interactive experience.
@@ 888
**A language model can improve grammar while making a transcript less faithful.** If the audio says an unusual name, a post-corrector may replace it with a familiar one. If the recognizer is uncertain about a number, the corrector may choose a plausible value with unjustified confidence. The goal is recovering spoken content, not writing the most natural sentence. [1]

Suppose the original transcript says “ship fifteen units” with low confidence on the number. A context-only corrector sees that previous orders were fifty and changes it to “fifty.” The result reads well but may be wrong. Without audio or a supported alternative hypothesis, the correction has no decisive evidence.

Constrain edits using N-best candidates, phonetic similarity, timestamps, and confidence. Preserve high-confidence spans and require stronger evidence for names, numbers, and negation. Keep the original transcript and mark unresolved uncertainty rather than filling gaps with fluent guesses.

**Measure harmful edits explicitly.** Count initially correct spans made incorrect, not only initially incorrect spans repaired. Evaluate silence and noisy audio, where hallucinated content is especially revealing. Compare against human audio-grounded references rather than relying solely on a text judge that may favor fluency. Added model calls also affect latency and data handling. A lower language-model loss is not proof of a better transcript.
@@ 889
**Text-to-speech must decide both what sounds to produce and how to realize them as a waveform.** A traditional neural pipeline predicts acoustic features such as a mel spectrogram, then a vocoder synthesizes samples. Duration modeling determines how long phonemes last; prosody controls rhythm, emphasis, and pitch. [1] [2]

FastSpeech-style systems use explicit duration information to generate many acoustic frames in parallel. VITS combines different ingredients, including latent-variable modeling, flows, and adversarial waveform synthesis; it should not be treated as merely another identical mel-plus-vocoder pipeline.

Codec language models generate discrete audio codes that a codec decoder turns into sound. This makes audio generation resemble sequence modeling over a learned vocabulary. Flow-matching systems instead learn a velocity field that transports noise toward a speech representation, then integrate that field during synthesis. These categories can overlap in how representations are encoded and decoded.

**Compare the whole listening experience.** A system can have a good average real-time factor yet delay the first audible chunk. Another can start quickly but mishandle long text, punctuation, or rare names. Measure intelligibility, natural prosody, speaker similarity where relevant, first-audio delay, and sustained generation speed. Count iterative refinement and vocoder cost rather than assuming fewer autoregressive tokens automatically means lower latency. Speaker-conditioned generation should use appropriate consent and speaker controls.
@@ 890
**Residual vector quantization improves an approximation in stages.** An encoder first maps audio into continuous latent vectors. The first codebook chooses a nearby vector. The next codebook approximates the error that remains, and later codebooks continue refining that residual. The decoder receives the sum of the chosen vectors. [1]

A one-dimensional illustration makes the idea concrete. Suppose the latent value is 2.7. The first stage chooses 2.0, leaving 0.7. The second chooses 0.5, leaving 0.2. The third chooses 0.2, reconstructing 2.7. Real codebooks contain multidimensional vectors, and the learned choices need not resemble these simple decimal increments.

Only codebook indices need to be transmitted or modeled. If there are eight codebooks, 1024 entries per codebook, and 50 latent frames per second, the raw index rate is $8\times\log_2(1024)\times50=4000$ bits/s, before entropy coding and other overhead. This is an illustrative configuration, not a universal codec specification.

More stages generally permit better reconstruction at greater bitrate and modeling cost. Dropping later stages supports variable quality only when the system is trained and designed for it. Earlier and later codebooks are coarse-to-fine in residual reconstruction, but that does not guarantee a clean semantic division such as words in one stage and speaker identity in another.
@@ 891
**Document understanding requires both reading symbols and attaching them to the right structure.** A conventional pipeline detects text regions, recognizes their contents, and reconstructs layout or tables. Each intermediate result is inspectable, but a missed region can propagate into every later stage. An OCR-free model generates text or structure directly from document pixels. [1]

Consider a financial table with “1.25” in one cell and “125” in another. A low-resolution image can erase the decimal point. A language model may then produce a plausible value based on context, but plausibility cannot recover information that the image no longer contains.

**Separate the error types.** Was the region missed? Were characters misread? Was correct text assigned to the wrong row or header? Direct generation can simplify the interface while hiding these distinctions, so retain evidence localization and high-resolution crops where possible. Hybrid systems can use OCR for fine text and a layout-aware model for relationships.

Evaluate cell text, numeric fidelity, row/column structure, and source grounding separately. Check totals and units as consistency tests, while recognizing that a table can be internally consistent and still copied incorrectly. For consequential documents, the output should be traceable to the relevant image region. “OCR-free” describes the architecture; it does not remove the need for visual evidence or validation.
@@ 892
**A table is a relationship structure, not just a collection of text boxes.** The value “17” is meaningful only when associated with the correct row label and column header. Merged headers, missing borders, rotated text, and multiple scripts make those relationships difficult to infer. [1]

A box-and-graph system predicts cells and relations such as adjacency or shared row/column membership. Its strength is explicit geometry. Its challenge is turning possibly inconsistent pairwise relations into one coherent grid. A sequence system instead emits HTML or a compact table grammar, representing spans and order as tokens.

**One small error can change many meanings.** If a header spans three columns but the model predicts a span of two, later values can be attached to the wrong category even when every character is recognized correctly. A sequence decoder can produce valid syntax yet an incorrect layout; a graph model can find good boxes yet connect them wrongly.

Validate cell occupancy, spans, row lengths, and reading order. Use high-resolution crops while retaining enough page context to identify titles and headers. Report results by difficult table type rather than only a pooled average. When diagnosing a failure, separate structure reconstruction from text recognition so you improve the component that actually caused the wrong association.
@@ 893
**TEDS asks whether the predicted table has the right tree structure and, depending on the variant, the right cell contents.** HTML represents a table as nested elements: table, rows, and cells. Tree edit distance measures the cost of transforming one such structure into another. Normalizing that cost gives a similarity score. [1]

A predicted cell can overlap the correct image region almost perfectly while claiming the wrong row span. Bounding-box IoU would look good, but the table's structure would be wrong. Conversely, two structurally identical tables can differ in one digit that changes a reported value by a factor of ten.

**A numerical illustration.** Under a chosen edit-cost implementation, suppose the larger tree has 20 nodes and the edit distance is 3. The normalized similarity is $1-3/20=0.85$. The actual edit rules matter: replacing cell text may be scored differently from adding or deleting a structural node.

Use structure-only TEDS to isolate layout errors and a content-aware version to include text. Add exact or task-specific numerical accuracy when small character errors have large consequences. Canonicalize equivalent markup consistently, since different HTML strings can describe the same table. No single score captures geometry, structure, and semantic correctness completely; choose complementary metrics that reveal which aspect failed.
@@ 894
**Downsampling can destroy the evidence before the model sees it.** If a 2200-pixel-high page is reduced to 224 pixels, a character originally 20 pixels high becomes roughly two pixels high. A decimal point or minus sign may vanish entirely. More language-model reasoning cannot reliably reconstruct an erased distinction. [1]

Dynamic high-resolution encoding addresses this by supplying detailed tiles along with a global view. The thumbnail shows where columns and sections lie; tiles preserve small characters. Coordinates and overlap help reconnect text that crosses tile boundaries.

**Detail creates its own costs.** More tiles produce more visual tokens, memory use, and downstream attention work. Overlap can duplicate text; separate crops can obscure reading order or attach a value to a header in another tile. A system needs a reconstruction strategy, not just a pile of enlarged patches.

Evaluate by original text size, page density, and table complexity. Test numeric punctuation and superscripts explicitly. For a question about one row, targeted high-resolution crops may be more efficient than encoding the entire page at maximum detail. Preserve source coordinates so extracted answers can be checked against the image. Good natural-image classification performance does not establish reliable small-text document reading.
@@ 895
**CLIP learns which image and text descriptions belong together.** It brings matched image/text embeddings closer and separates mismatched pairs. At zero-shot classification time, descriptions such as “a photo of a dog” become class vectors, and the image is assigned to a similar text vector. [1]

This objective can reward recognizing prominent objects without requiring exact compositional reasoning. An image containing a red cube and a blue sphere may resemble both “red cube above blue sphere” and “blue sphere above red cube.” The objects and words are present in both descriptions; the distinguishing relationship is harder.

**Counting and OCR require fine distinctions.** Two versus three small objects may barely change a global embedding. A single character in a sign can determine the answer while occupying very few pixels. If training rarely demands those distinctions, the representation has limited incentive to preserve them precisely.

These are tendencies to test, not statements that every CLIP model always fails. Evaluate paired examples differing only in count, relation, or text. Higher resolution, region-level representations, targeted supervision, or a specialized OCR/counting component can help when exactness matters. A strong retrieval score over broad semantic categories does not automatically imply reliable reasoning about every detail within the image.
@@ 896
**Softmax contrastive learning asks which candidate is the match; sigmoid learning asks whether each pair matches.** In CLIP's batch objective, the correct text competes against other texts in a shared normalization. Increasing one candidate's probability changes the normalized probabilities of the others. SigLIP applies a binary match/non-match loss to each evaluated pair. [1]

For an image with one matched caption and many mismatched captions, sigmoid training pushes the matched pair's score upward and mismatched scores downward independently through their binary losses. A learned scale and bias help set the score range and decision baseline. The absence of a global softmax denominator makes the pairwise computation easier to partition in some implementations.

**Independent losses do not eliminate negative-pair cost.** A batch of b images and b texts still has $b^2$ possible pairs. Computing all of them requires work, while sampling a subset changes coverage and weighting. With many more negatives than positives, loss normalization and bias are important parts of the method.

Compare the objectives under matched data, compute, and evaluation. Reported batch-size advantages are empirical findings for particular training setups, not a theorem that sigmoid always converges better. Distributed performance also depends on where embeddings and pair blocks are placed. The useful distinction is the training comparison being made, not simply whether the formula contains a softmax or a sigmoid.
@@ 897
**Self-supervised vision creates a learning problem from images without needing a class label for each one.** SimCLR treats two augmentations of the same image as a positive pair and other images as negatives. MoCo maintains a momentum encoder and a queue to provide a large, more stable set of comparison features. [1] [2]

BYOL and DINO use a student that matches a teacher's representation, with architectural and optimization mechanisms that discourage every image receiving the same output. The teacher is typically updated more slowly than the student. MAE instead hides many image patches and learns to reconstruct them from the visible patches, emphasizing prediction of missing content.

**Why object-like regions can emerge.** Different crops of an image may share an object while backgrounds vary. Learning consistent features across those views can favor representations organized around recurring visual entities. Patch-level interactions can then produce maps that highlight an object even without a segmentation label for every pixel.

An attractive attention map is not automatically a precise segmentation mask or a causal explanation. Evaluate the representation through downstream tasks such as linear classification, dense prediction, and transfer to new domains. Reconstruction, instance discrimination, and self-distillation encourage different information to be preserved. Whether a loss uses explicit negatives is only one design choice, not a complete explanation of representation quality.
@@ 898
**Small task datasets can tempt a large backbone to memorize incidental details.** Freezing the backbone and learning a task head limits how much the representation can change. LoRA permits a restricted low-rank update, offering an intermediate choice between a fixed representation and unrestricted fine-tuning. [1] [2]

Imagine adapting an image model to defects on one factory's product. Full fine-tuning may learn the factory's lighting or camera background as a shortcut. A constrained update can reduce this risk, but it may also lack the flexibility needed to detect a genuinely new microscopic texture.

**The right adaptation depends on what must be preserved.** Segmentation needs spatial localization; super-resolution needs recoverable fine detail; defect grading may depend on tiny frequency patterns. A backbone good at image-level semantics may discard exactly the information one of these tasks requires. Freezing it does not create missing detail.

Compare a frozen probe, selected-layer tuning, several LoRA ranks and target layers, and regularized full tuning. Evaluate new cameras, materials, and lighting, not only random splits of nearly identical frames. Track retained general capabilities as well as task quality. Keeping base weights unchanged makes rollback easy, but the active adapter can still degrade the deployed representation. “Preserves spatial priors better” is a hypothesis to test, not an automatic consequence of low rank.
@@ 899
**Stereo depth comes from how much the same point shifts between two camera views.** For rectified cameras, corresponding points lie on the same image row. A nearby point shifts more horizontally than a distant point. That horizontal difference is disparity. [1]

With focal length f in pixels, baseline B in meters, and disparity d in pixels, depth is $Z=fB/d$. If f=800 pixels, B=0.1 m, and d=40 pixels, depth is 2 m. If disparity falls to 4 pixels, depth becomes 20 m.

**The same pixel error becomes much worse far away.** Differentiating gives $|dZ/dd|=fB/d^2$. A one-pixel disparity uncertainty is about 0.05 m at d=40 but about 5 m at d=4 under the local approximation. The far-range estimate can therefore look numerically precise while being physically very uncertain.

A larger baseline increases disparity for the same depth and improves sensitivity, but also increases viewpoint differences and occlusions, making correspondence harder. Textureless surfaces, reflections, and calibration errors cause additional failures beyond the derivative calculation. Keep focal length and disparity in consistent pixel units, especially after resizing images. Report depth confidence and reject unreliable matches instead of treating every computed disparity as equally trustworthy.
@@ 900
**Segmentation models answer different interface and latency needs.** Mask R-CNN first localizes candidate objects and predicts masks for those regions. YOLO-style segmentation emphasizes efficient detection and mask prediction in a single-stage family. SAM is designed for promptable segmentation, so a deployment also needs a way to supply prompts or generate and select masks automatically. [1] [2]

For an industrial part, a large clean mask may be easy while a one-pixel crack is decisive. Evaluate tiny-defect recall, boundary accuracy, separation of touching instances, and stability under lighting changes. Family names alone do not determine speed: input resolution, implementation, hardware, and prompt-generation overhead can dominate.

**Pixels are not millimeters.** A mask's area or width becomes a physical measurement only with calibrated geometry. If the object lies on a known plane, a calibrated mapping can convert image coordinates. If depth varies, perspective and distance must be accounted for; the same pixel width can represent different physical sizes.

Measure end-to-end error in physical units on reference objects, including segmentation and calibration uncertainty. Occlusion, lens distortion, and reflections can bias measurements even when masks look plausible. Choose the smallest system that meets both quality and latency requirements, and allow uncertain cases to be rejected rather than displaying unsupported precision.
@@ 901
**Extrinsics answer where the camera is; intrinsics answer how its rays become pixels.** Extrinsic rotation and translation transform a world point into camera coordinates. Intrinsics then scale the projected coordinates by focal lengths and shift them by the principal point. Lens distortion adjusts the ideal pinhole result to match real optics. [1]

For a camera-frame point (X,Y,Z), the undistorted horizontal coordinate is approximately $u=f_xX/Z+c_x$ when skew is zero. If X=1 m, Z=5 m, $f_x=1000$ pixels, and $c_x=640$, then u=840 pixels. Moving the point twice as far away with the same X halves its offset from the principal point.

Radial distortion changes displacement with distance from the optical center; tangential distortion captures effects such as lens decentering. Calibration estimates these parameters using known geometry observed from several viewpoints. Held-out views help reveal whether the fitted model generalizes beyond the calibration images.

**Coordinate conventions are part of the answer.** State whether the transform maps world to camera or the reverse, and use consistent units. Cropping shifts the principal point; resizing changes focal lengths expressed in pixels. Focus or zoom changes may invalidate a previous calibration. Inspect residuals across the image, especially corners, rather than trusting only one average reprojection error.
@@ 902
**Camera and LiDAR provide complementary evidence.** Cameras capture texture, color, and rich semantic cues; LiDAR measures geometric range at sampled directions. Fusion tries to combine those strengths, but only if measurements refer to the same place and time. [1] [2]

Early or feature fusion combines measurements or learned features before the final prediction. It can exploit complementary local evidence but is sensitive to alignment. Late fusion combines independently detected objects, making components modular while potentially discarding information that would have resolved an earlier ambiguity.

**Bird's-eye view provides a shared spatial workspace.** Features are represented over a ground-plane grid so objects from different cameras or times can be reasoned about together. Moving from image pixels into that space requires geometry, learned depth or attention, and appropriate transformations. BEVFormer is specifically a multi-camera temporal/spatial design; not every BEV model includes LiDAR.

A timestamp error can place a moving car's camera appearance beside its LiDAR points instead of on them. Incorrect extrinsics produce similar misalignment even for static objects. Evaluate timing offsets, calibration errors, missing sensors, weather, and range. Temporal fusion also needs ego-motion compensation. The representation makes spatial combination convenient; it does not automatically solve synchronization or make uncertain sensor evidence exact.
''')
