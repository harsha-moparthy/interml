from author import write
write(r'''
@@ 843|testtime chinchilla r1 prm scaling
Pretraining scaling allocates a one-time compute budget between parameters and training tokens. Test-time scaling allocates per-request compute to additional samples, search, verification, or longer generated reasoning. The two are complementary, not replacements. [1] [2]

$$C_{total}=C_{train}+Q\,C_{request}.$$

For a hard but solvable prompt, a capable smaller model plus a reliable verifier can use extra attempts more effectively than a modest parameter increase. For problems beyond the model's proposal capability or a weak verifier, extra inference may just repeat errors. Compare matched total FLOPs, latency, query volume, and difficulty. The “often beats doubling” claim is conditional on benchmark and budget; at very large serving volume, repeated inference costs can outweigh one-time training savings.
@@ 844|r1 ppo prm star testtime
RL rewards successful completions, so behavior patterns correlated with success can become more frequent without step-by-step human labels. A model pretrained on reasoning text already has candidate patterns such as checking or revising; reward optimization can select and extend them. [1]

This does not prove human-like introspection or that every visible “wait” indicates a meaningful correction. Distinguish genuine improvements on independently verified tasks from longer text or reward-format exploitation. Rule-based rewards work best where correctness can be checked reliably, such as executable code or math with a known answer. Training details differ across model families; unpublished internals should not be inferred from surface behavior. Evaluate accuracy, compute, and robustness under independent tests, not narrative appearance alone.
@@ 845|prm r1 ppo testtime drgrpo
Tree search amplifies whatever its value/process scorer favors. If a scorer rewards plausible-looking but incorrect steps, search discovers those weaknesses; changing policies also shift the distribution on which the value model must be accurate. Fine-grained labeling and long branching searches add substantial cost. [1] [2]

Verifiable outcome rewards avoid some learned-judge ambiguity, and group-relative baselines remove a separately trained critic. They still suffer sparse rewards, test incompleteness, and reward hacking. This is not proof that MCTS/PRMs universally fail: they can help when intermediate verification and search structure are strong. Compare correctness at equal total compute and audit the verifier itself. A broken test suite gives a precisely optimized but wrong target regardless of the RL algorithm.
@@ 846|drgrpo r1 ppo simpo prm
For rewards $r_1,\ldots,r_G$, group-standardized advantages are $A_i=(r_i-\bar r)/(s_r+\epsilon)$. All-equal rewards produce no comparative signal. In common GRPO objectives, averaging each response's token losses by its own length changes the per-token weight of long versus short responses; reward-standard-deviation scaling separately changes weighting across prompt difficulties. [1]

Dr. GRPO analyzes these as distinct biases and uses a constant length normalization and removes group standard-deviation scaling in its proposed correction. “Token-level normalization fixes it” is too vague: averaging across all batch tokens and normalizing every response independently are different objectives.

Track correct/incorrect response lengths, truncation, reward variance, and gradient contribution by prompt. The exact loss reduction is part of the algorithm, not an innocuous implementation detail. [2]
@@ 847|star r1 prm ppo dpo
Rejection-sampling fine-tuning generates candidate solutions, keeps those passing a correctness check, and trains with supervised likelihood on the retained examples. STaR iterates generation and training, including rationalization mechanisms to bootstrap useful reasoning. [1]

$$\mathcal L_{RFT}=-\sum_{(x,y):verify(y)=1}\log\pi_\theta(y\mid x).$$

RFT is simple when enough correct diverse samples exist and verification is strong. It discards unsuccessful trajectories and can narrow diversity or reinforce flawed rationales that happen to reach correct answers. Online RL explicitly uses relative reward and exploration from the current policy, but has more optimization and rollout complexity. Start with verified supervised data when feasible; use RL when the objective benefits from ongoing exploration and the reward signal withstands exploitation.
@@ 848|lost yarn rag ragas testtime
A model's advertised context capacity says how many tokens it accepts, not how reliably it uses them. Relevant evidence can be harder to retrieve when surrounded by distractors or placed in the middle; conflicting instructions and accumulated conversational noise can further degrade use. [1]

A single needle test mainly asks whether one explicit fact can be recovered. It does not test linking several distant facts, resolving contradictions, aggregating many records, or citing every supporting source. Build long-context evaluations varying evidence position, number of hops, distractor similarity, answerability, and document order.

Compare retrieval-plus-short-context with full-context input at equal cost/latency. Summaries reduce length but can omit crucial qualifiers; retain source provenance and allow targeted re-reading rather than assuming compression preserves every fact.
@@ 849|react retry agents sre xgrammar
Represent tool calls and results as typed state transitions. Validate arguments before execution; return actionable structured errors for schema mistakes; distinguish transient timeouts/5xx errors from permanent invalid requests and valid empty results. [1] [2]

Use per-call deadlines, bounded retries with backoff/jitter, and a total task budget. Side-effecting calls need idempotency keys or status checks before retrying. An empty search result should trigger a changed query or an explicit information gap, not the same request forever.

Store attempted calls and outcomes so repeated failures are detectable. Escalate or stop when the remaining budget cannot support a useful alternative. Tool error text is untrusted data, not authority to alter permissions. Successful recovery means a verified task result, not simply suppressing exceptions.
@@ 850|xgrammar react agents torch attention
Constrained decoding tracks valid prefixes under a grammar or schema and masks tokens that would make completion invalid. The remaining logits are renormalized before sampling. JSON syntax can require stack-like state, so not every useful schema is representable by a simple finite-state machine. [1]

$$p'(t)=\frac{p(t)\mathbf1[t\in A(prefix)]}{\sum_{u\in A(prefix)}p(u)}.$$

This can guarantee supported syntactic constraints, not semantic correctness or “zero accuracy loss.” Masking changes the distribution and may exclude a model's preferred reasoning/format path. Tokenization, escapes, numeric ranges, and schema support need testing. Validate business rules after parsing and authorize tool actions independently; a schema-valid request can still be wrong or inappropriate.
@@ 851|xgrammar testtime prm react agents
A strict output schema can force premature commitment: the model may have to emit a final field before it has resolved dependencies or uncertainty. Grammar validity only restricts possible strings; it does not supply the missing computation. [1]

One design separates a bounded internal analysis/verification stage from final structured serialization. Another asks for explicit intermediate evidence fields only when useful and appropriate for the product. A free-form “thinking” field is not guaranteed to improve accuracy and should not be treated as proof of reasoning quality.

Compare task correctness, schema validity, latency, and output privacy under controlled prompts. External checks—calculations, type validation, or source retrieval—often provide more dependable verification than a longer self-explanation. Keep final output requirements clear and avoid schema order that conflicts with genuine data dependencies.
@@ 852|agents react rag sre mltest
Keep a compact working state containing the goal, constraints, current hypothesis, relevant file locations, and recent bounded command output. Store episodic records of attempted fixes, observed failures, and why approaches were rejected. Maintain a version-aware semantic/lexical/AST index for finding code, but retrieve exact current files before editing. [1] [2]

Every memory entry should record provenance and repository revision; stale summaries must not override current code or user instructions. Save large logs as artifacts and retrieve targeted sections instead of repeatedly inserting them into context. Checkpoints should include remaining work and validation results.

Separate planning from tool authorization, run code in an isolated environment, and require tests or inspection to verify completion. More memory is useful only when relevant, current, and distinguishable from untrusted repository content.
@@ 853|agents react sre mltest retry
Direct prompt injection places adversarial instructions in a user's input; indirect injection places them in material the agent retrieves, such as a webpage, email, or source file. The central failure is promoting untrusted content into authority over goals or tools. [1]

Enforce authorization outside the model: least-privilege tools, scoped credentials, filesystem/network isolation, validated destinations, and separate read/write capabilities. Treat retrieved instructions as data unless explicitly authorized. Sensitive side effects should require a valid user intent and policy check independent of the page's wording.

Test attacks that attempt tool misuse, data exfiltration, and instruction override in an isolated harness. Input filtering and a warning prompt alone are insufficient; constrain what the system can actually do, log decisions, and limit damage when a model misinterprets content.
@@ 854|ewc lora ties reg qlora
Narrow fine-tuning can overwrite behavior useful on the original distribution. Replay mixes representative old-task examples into training. LoRA limits the update subspace and leaves base weights intact, making adapter removal easy, but an active adapter can still degrade general behavior. EWC penalizes changes to parameters estimated important for prior tasks. [1] [2]

$$L=L_{new}+\frac\lambda2\sum_iF_i(\theta_i-\theta_i^*)^2.$$

Model merging combines updates or checkpoints; TIES/DARE/SLERP use different heuristics and do not guarantee retention. Evaluate old and new tasks, calibration, and worst slices across tuning strengths. Replay quality and legal data availability matter, and approximate importance estimates can become stale. Retaining a frozen base checkpoint is a recovery mechanism, not proof of no forgetting.
@@ 855|ties ewc lora optim reg
Fine-tuned models from the same initialization often preserve parameter alignment, making interpolation or task-vector addition more meaningful than averaging independently initialized networks. Define $\Delta_i=\theta_i-\theta_0$ and merge $\theta=\theta_0+\sum_i\alpha_i\Delta_i$. [1]

Ordinary averaging combines weights directly; task arithmetic combines deltas; SLERP interpolates directions on a sphere; TIES trims small deltas and resolves sign conflicts before merging. These are empirical methods, not guarantees of a low-loss path. Different architectures/tokenizers or neuron permutations can make merging invalid.

Tune coefficients on a held-out multi-task suite and test interference, retention, and safety-relevant behaviors. Do not optimize the merge on the final test set. A merged model can fail tasks that every source checkpoint handled well.
@@ 856|collapse reg r1 star mltest
Repeatedly replacing real data with a model's outputs can amplify its errors and underrepresent rare events. Filtering only for fluent or high-confidence outputs further narrows the distribution. Synthetic data can still help when it adds verified diversity and remains anchored to trustworthy evidence. [1] [4]

Retain a versioned real-data core, track synthetic provenance and generation model, deduplicate, deliberately sample rare cases, and verify labels with independent tools or reviewers. Avoid training/evaluating on near-identical generated examples. Measure tail coverage and subgroup quality, not just aggregate loss on another synthetic set.

A stronger generator is not sufficient if filtering discards hard cases or downstream training overweights duplicates. Compare against adding equally many real examples when possible and audit how mixture weights change over generations.
@@ 857|caiai judge rlhf dpo prm
Constitutional AI uses written principles to guide critique/revision and preference judgments; RLAIF uses AI-generated feedback in training rather than relying entirely on human labels. This scales supervision but also transfers the judge's blind spots. [1]

To discourage sycophancy, include paired examples where a user confidently asserts a false premise and the correct response respectfully corrects it. Score factual support independently from politeness, verbosity, and agreement. Blind model identity and calibrate judges against human/executable checks. [2]

A written principle is not self-enforcing: monitor whether reward optimization learns superficial phrases rather than behavior. Use adversarial evaluations, independent reference evidence, and disagreement review. Keep the distinction between being helpful to a user and agreeing with every claim the user makes.
@@ 858|unlearn reg prml dl mltest
Knowledge is distributed across shared parameters, so increasing loss on a forget set can damage neighboring useful capabilities and still fail to remove all recoverable information. Paraphrases, alternate prompts, and memorized fragments may survive. [1] [2]

Evaluate unlearning against a retrained-without-data reference where feasible, using held-out related facts, extraction attempts, membership tests with known limitations, and retained-task utility. Combine targeted updates with retention objectives or localized mechanisms, but do not equate refusal at inference with deletion from weights.

Track training-data provenance and exact scope of the removal request. Gradient ascent is an optimization operation, not a certificate of legal or technical erasure. Strong claims require defined adversary/access assumptions and evidence beyond a small set of prompted examples.
@@ 859|sentencepiece attention bert t5 scaling
BPE repeatedly merges frequent symbol pairs. WordPiece selects subwords using a likelihood-related training criterion and commonly decodes via longest matching pieces. Unigram tokenization starts from a candidate vocabulary and prunes it under a probabilistic segmentation model; SentencePiece is a toolkit supporting multiple algorithms, not one algorithm itself. [1]

Byte-level/token-free models avoid a fixed subword vocabulary but face longer raw sequences; latent patching can group bytes adaptively. A larger vocabulary can encode frequent multilingual/code substrings more compactly, at the cost of larger embedding/output matrices and fewer observations per rare token.

Compression depends on training corpus and vocabulary allocation, not size alone. Evaluate bytes or characters per token by language, normalization reversibility, rare scripts, latency, and downstream accuracy.
@@ 860|sentencepiece attention testtime torch d2l
Subword tokens hide character boundaries: a word may be one token or several chunks, so counting letters requires recovering and operating on its spelling rather than counting input tokens. Arithmetic similarly depends on digit alignment, carries, and place value, which arbitrary multi-digit chunks can obscure. [1]

Single-digit tokenization exposes consistent units; right-to-left grouping can align least-significant digits and carries. These inductive biases may help but do not guarantee algorithmic generalization to longer numbers. “Strawberry” has three r's regardless of its tokenizer representation.

For exact tasks, execute a deterministic string/counting or arithmetic operation and validate its input. Benchmark across lengths and unusual characters, including Unicode normalization; apparent gains on familiar number lengths can conceal failure on distribution shifts.
@@ 861|sae dl prml reg lora
Activation steering adds a chosen direction to hidden states, often derived from contrasting example sets. It can change behavior without updating weights, but effects depend on layer, strength, prompt, and off-target changes. [1]

A sparse autoencoder reconstructs activations through an overcomplete sparse code:

$$z=\operatorname{ReLU}(W_ex+b_e),\quad \hat x=W_dz+b_d,\quad L=\|x-\hat x\|^2+\lambda\|z\|_1.$$

Sparsity encourages features that activate selectively, helping separate concepts superposed in neuron coordinates. “Monosemantic” is an interpretive goal, not a guarantee that each feature has one complete human-readable meaning. Validate with interventions and held-out examples; correlated activation labels alone do not establish causal control or exhaustive understanding of the model.
@@ 862|agents react testtime judge sre
Extra agents help when work can be decomposed, contexts differ usefully, and outputs are independently verifiable. They hurt when handoffs lose constraints, agents share correlated errors, reviewers reward persuasive text, or coordination consumes more time than the work itself. [1]

If a serial workflow has independent stage success probabilities $p_i$, end-to-end success is $\prod_i p_i$; five 95%-reliable stages give about 77.4%. Independence is only an illustrative assumption—shared model errors can be worse. Parallel redundancy helps only when selection/verification is reliable.

Use typed artifacts, one source of task state, clear ownership, bounded messages, and executable acceptance tests. Compare with a single-agent baseline at equal compute and latency. The number of agents is not a quality metric.
''')
