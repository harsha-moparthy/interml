"""Curated, beginner-friendly readings and deterministic topic matching.

The catalog favors blog posts, visual essays, and tutorials that explain an idea
without needing the original paper. A question receives the two highest-scoring
distinct readings; this makes 1,100-question coverage reproducible and testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Reading:
    title: str
    url: str
    publisher: str
    tags: str
    why: str
    access: str = "Free to read"


def R(title, url, publisher, tags, why, access="Free to read"):
    return Reading(title, url, publisher, tags, why, access)


# Tags include exact phrases where a resource is unusually strong. Each entry is
# a readable explainer; reuse happens only where one post genuinely covers
# several related questions.
CATALOG = [
    R("A Visual Intro to NumPy and Data Representation", "https://jalammar.github.io/visual-numpy/", "Jay Alammar", "numpy array indexing slicing view copy shape vectorization tensor", "Pictures of arrays, axes, and shapes make indexing and vectorization easy to see."),
    R("Broadcasting in NumPy, explained", "https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html", "Jake VanderPlas", "numpy broadcasting shape vectorization pairwise outer", "Walks through broadcasting rules with small shape diagrams and runnable examples."),
    R("Fancy Indexing in NumPy", "https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html", "Jake VanderPlas", "numpy vectorization array indexing boolean mask gather advanced", "Explains boolean and integer-array indexing with short runnable examples."),
    R("Aggregation and Grouping in pandas", "https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html", "Jake VanderPlas", "pandas groupby aggregate transform apply split combine", "A step-by-step tutorial on split-apply-combine with clear examples."),
    R("Combining Datasets: Merge and Join in pandas", "https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html", "Jake VanderPlas", "pandas merge join concat dataframe combine", "Explains each join type with small tables you can follow by eye."),
    R("Explained Visually: Eigenvectors and Eigenvalues", "https://setosa.io/ev/eigenvectors-and-eigenvalues/", "Setosa", "linear algebra eigen eigenvalue eigenvector matrix transformation svd", "Interactive diagrams show what a matrix does to space."),
    R("The Matrix Calculus You Need for Deep Learning", "https://explained.ai/matrix-calculus/", "Terence Parr and Jeremy Howard", "calculus derivative gradient jacobian hessian matrix backprop chain rule", "Builds matrix derivatives up from single-variable calculus with worked examples."),
    R("An Intuitive (and Short) Explanation of Bayes' Theorem", "https://betterexplained.com/articles/an-intuitive-and-short-explanation-of-bayes-theorem/", "BetterExplained", "bayes bayesian posterior prior likelihood conditional probability", "A plain-language story that makes Bayes' rule feel obvious."),
    R("Visual Information Theory", "https://colah.github.io/posts/2015-09-Visual-Information/", "Chris Olah", "entropy cross entropy kl divergence information theory mutual information loss", "Draws entropy, cross-entropy, and KL divergence as areas you can compare."),
    R("How to Explain Gradient Boosting", "https://explained.ai/gradient-boosting/", "Terence Parr and Jeremy Howard", "gradient boosting xgboost residual ensemble trees loss", "Builds boosting from a tiny example with clear plots."),
    R("Data Leakage in Machine Learning", "https://machinelearningmastery.com/data-leakage-machine-learning/", "Machine Learning Mastery", "data leakage target leakage train test contamination pipeline", "Plain examples of how leakage sneaks in and how to prevent it."),
    R("Visualizing K-Means Clustering", "https://naftaliharris.com/blog/visualizing-k-means-clustering/", "Naftali Harris", "kmeans clustering centroid initialization unsupervised", "Click to place points and watch k-means converge."),
    R("How to Use t-SNE Effectively", "https://distill.pub/2016/misread-tsne/", "Distill", "tsne umap embedding visualization dimensionality reduction perplexity", "Interactive demos of how t-SNE settings change the picture."),
    R("Calibration of probabilities (scikit-learn guide)", "https://scikit-learn.org/stable/modules/calibration.html", "scikit-learn", "probability calibration reliability diagram platt isotonic brier temperature scaling ece", "Plots and short code explain reliability curves, Platt scaling, and isotonic regression."),
    R("Interpretable ML: SHAP", "https://christophm.github.io/interpretable-ml-book/shap.html", "Christoph Molnar", "interpretability shap lime permutation feature importance shapley", "A friendly chapter that explains Shapley values with examples."),
    R("Causal Inference for the Brave and True", "https://matheusfacure.github.io/python-causality-handbook/landing-page.html", "Matheus Facure", "causal inference confounder collider propensity difference regression discontinuity instrumental uplift treatment", "A humorous, code-first guide to causal thinking."),
    R("How Not To Run an A/B Test", "https://www.evanmiller.org/how-not-to-run-an-ab-test.html", "Evan Miller", "ab test experiment significance peeking sample size hypothesis", "A short classic on the most common A/B testing mistake."),
    R("Calculus on Computational Graphs: Backpropagation", "https://colah.github.io/posts/2015-08-Backprop/", "Chris Olah", "backprop backpropagation chain rule computational graph autograd gradient", "Explains backprop as passing derivatives along a graph."),
    R("Understanding LSTM Networks", "https://colah.github.io/posts/2015-08-Understanding-LSTMs/", "Chris Olah", "lstm gru rnn recurrent sequence gate", "The standard picture-by-picture explanation of recurrent gates."),
    R("The Unreasonable Effectiveness of Recurrent Neural Networks", "https://karpathy.github.io/2015/05/21/rnn-effectiveness/", "Andrej Karpathy", "rnn recurrent character language model sequence", "A story-driven tour of what RNNs learn."),
    R("Conv Nets: A Modular Perspective", "https://colah.github.io/posts/2014-07-Conv-Nets-Modular/", "Chris Olah", "convolution cnn convolutional neural network layer pooling", "Builds a convolutional network from simple pieces."),
    R("Understanding Convolutions", "https://colah.github.io/posts/2014-07-Understanding-Convolutions/", "Chris Olah", "convolution kernel filter image signal", "Explains what a convolution is with falling-ball pictures."),
    R("A Recipe for Training Neural Networks", "https://karpathy.github.io/2019/04/25/recipe/", "Andrej Karpathy", "debug training nan loss overfit batch gradient initialization pipeline", "A practical, step-by-step method for debugging training."),
    R("An Overview of Gradient Descent Optimization Algorithms", "https://ruder.io/optimizing-gradient-descent/", "Sebastian Ruder", "optimizer sgd momentum adam adagrad rmsprop adadelta adamax nesterov learning rate", "Compares SGD, momentum, Adam and friends with clear intuition."),
    R("Why Momentum Really Works", "https://distill.pub/2017/momentum/", "Distill", "momentum optimization gradient descent convergence learning rate", "Interactive demos show how momentum speeds up descent."),
    R("Initializing Neural Networks", "https://www.deeplearning.ai/ai-notes/initialization/index.html", "DeepLearning.AI", "initialization xavier he weights vanishing exploding gradient batchnorm", "Interactive notes on why starting weights matter."),
    R("A Gentle Introduction to torch.autograd", "https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html", "PyTorch", "pytorch autograd backward gradient detach computation graph", "A short tutorial that follows one gradient through the graph."),
    R("Object Detection for Dummies, Part 1", "https://lilianweng.github.io/posts/2017-10-29-object-recognition-part-1/", "Lilian Weng", "object detection bounding box iou nms sliding window hog image gradient", "Explains boxes, IoU, and basic detection ideas step by step."),
    R("Object Detection for Dummies, Part 3: R-CNN Family", "https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/", "Lilian Weng", "object detection rcnn fast rcnn faster rcnn mask rcnn region proposal segmentation anchor", "Walks through R-CNN to Mask R-CNN with diagrams."),
    R("Object Detection for Dummies, Part 4: Fast Detection Models", "https://lilianweng.github.io/posts/2018-12-27-object-recognition-part-4/", "Lilian Weng", "object detection yolo ssd retinanet one stage detector", "Explains YOLO, SSD, and RetinaNet in plain steps."),
    R("Non-Maximum Suppression for Object Detection in Python", "https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/", "PyImageSearch", "nms non maximum suppression iou bounding box overlap", "Builds NMS with code and pictures."),
    R("The Illustrated Transformer", "https://jalammar.github.io/illustrated-transformer/", "Jay Alammar", "transformer encoder decoder cross attention self attention positional encoding multi head", "A precise visual walkthrough that keeps tensor flow tied to the architecture."),
    R("The Illustrated GPT-2", "https://jalammar.github.io/illustrated-gpt2/", "Jay Alammar", "gpt decoder causal attention mask autoregressive language model", "Explains masked self-attention and generation step by step."),
    R("Visualizing Seq2seq Models With Attention", "https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/", "Jay Alammar", "attention seq2seq encoder decoder alignment translation", "Animated pictures show why attention was invented."),
    R("Attention? Attention!", "https://lilianweng.github.io/posts/2018-06-24-attention/", "Lilian Weng", "attention self attention cross attention multi head memory", "A guided tour of attention variants."),
    R("The Transformer Family Version 2.0", "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/", "Lilian Weng", "transformer efficient attention sparse linear attention memory architecture variants moe", "A readable map of Transformer improvements."),
    R("The Annotated Transformer", "https://nlp.seas.harvard.edu/annotated-transformer/", "Harvard NLP", "transformer attention scaled dot product multi head implementation pytorch code", "The Transformer paper rewritten as line-by-line runnable code."),
    R("Transformers From Scratch", "https://peterbloem.nl/blog/transformers", "Peter Bloem", "transformer self attention scaled dot product softmax sqrt query key value", "Builds self-attention from simple matrix steps and explains each design choice."),
    R("Transformer Explainer", "https://poloclub.github.io/transformer-explainer/", "Georgia Tech Polo Club", "transformer gpt attention softmax temperature embedding interactive", "Run GPT-2 in the browser and watch every internal step."),
    R("Rotary Embeddings: A Relative Revolution", "https://blog.eleuther.ai/rotary-embeddings/", "EleutherAI", "rope rotary position embedding relative rotation positional", "Explains why rotating query and key vectors encodes relative position."),
    R("FlashAttention-2 (Stanford CRFM blog)", "https://crfm.stanford.edu/2023/07/17/flash2.html", "Stanford CRFM", "flashattention 2 attention gpu memory speed", "The authors explain why attention is memory-bound and how FlashAttention fixes it."),
    R("Making Deep Learning Go Brrrr From First Principles", "https://horace.io/brrr_intro.html", "Horace He", "roofline arithmetic intensity compute bound memory bandwidth overhead gpu performance fusion", "The best intuition for compute, memory, and overhead limits."),
    R("KV Caching Explained", "https://huggingface.co/blog/not-lain/kv-caching", "Hugging Face", "kv cache decoding autoregressive inference attention speed", "Shows what is cached and why generation gets faster."),
    R("vLLM and PagedAttention (vLLM blog)", "https://blog.vllm.ai/2023/06/20/vllm.html", "vLLM team", "vllm pagedattention kv cache continuous batching serving fragmentation prefix cache", "The authors explain paged KV memory with diagrams."),
    R("How to Generate Text: Decoding Methods", "https://huggingface.co/blog/how-to-generate", "Hugging Face", "decoding greedy beam search sampling top k top p temperature generation", "Compares greedy, beam, and sampling with examples."),
    R("Mixture of Experts Explained", "https://huggingface.co/blog/moe", "Hugging Face", "mixture of experts moe router gating sparse expert load balancing", "Covers routing, load balancing, and serving of MoE models."),
    R("A Visual Guide to Mamba and State Space Models", "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state", "Maarten Grootendorst", "mamba state space model ssm linear recurrence sequence", "Many diagrams build up state-space models from RNNs."),
    R("Byte-Pair Encoding Tokenization", "https://huggingface.co/learn/llm-course/chapter6/5", "Hugging Face", "tokenization tokenizer bpe byte pair vocabulary subword wordpiece", "Trains a BPE tokenizer by hand on a tiny corpus."),
    R("The Illustrated Word2vec", "https://jalammar.github.io/illustrated-word2vec/", "Jay Alammar", "embedding word2vec negative sampling skipgram nlp vector", "Builds intuition for learned embeddings while retaining the training objective."),
    R("The Illustrated BERT, ELMo, and co.", "https://jalammar.github.io/illustrated-bert/", "Jay Alammar", "bert encoder masked language model pretraining transfer learning classification", "Explains how pretrained encoders are trained and used."),
    R("Practical Tips for Finetuning LLMs Using LoRA", "https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms", "Sebastian Raschka", "lora low rank adapter peft fine tuning rank alpha", "Experiment-backed advice that also explains how LoRA works."),
    R("Parameter-Efficient LLM Finetuning With LoRA", "https://www.sebastianraschka.com/blog/2023/llm-finetuning-lora.html", "Sebastian Raschka", "lora low rank adapter peft fine tuning merge weight update", "Builds LoRA from the low-rank idea with simple code."),
    R("Making LLMs Even More Accessible: bitsandbytes, 4-bit and QLoRA", "https://huggingface.co/blog/4bit-transformers-bitsandbytes", "Hugging Face", "qlora nf4 quantization 4bit double quantization fine tuning memory", "Explains 4-bit NormalFloat and QLoRA with usable code."),
    R("Illustrating Reinforcement Learning from Human Feedback (RLHF)", "https://huggingface.co/blog/rlhf", "Hugging Face", "rlhf reward model preference alignment ppo kl", "A clear three-stage picture of how chat models are aligned."),
    R("Preference Tuning LLMs with DPO and Friends", "https://huggingface.co/blog/pref-tuning", "Hugging Face", "dpo preference alignment chosen rejected ipo kto", "Compares DPO-style methods in practice."),
    R("Proximal Policy Optimization (PPO) (HF Deep RL Course)", "https://huggingface.co/blog/deep-rl-ppo", "Hugging Face", "ppo reinforcement learning clipped objective policy gradient advantage", "Explains the clipped objective with intuition and diagrams."),
    R("The 37 Implementation Details of PPO", "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/", "ICLR Blog Track", "ppo implementation details advantage normalization gae value clipping", "Lists what really matters when coding PPO."),
    R("Policy Gradient Algorithms", "https://lilianweng.github.io/posts/2018-04-08-policy-gradient/", "Lilian Weng", "reinforcement learning policy gradient actor critic reinforce gae reward", "Bridges REINFORCE to actor-critic methods."),
    R("Hugging Face Deep Reinforcement Learning Course", "https://huggingface.co/learn/deep-rl-course/unit0/introduction", "Hugging Face", "reinforcement learning q learning dqn policy environment reward exploration", "A hands-on course that explains RL ideas before the math."),
    R("Tutorial: What Is a Variational Autoencoder?", "https://jaan.io/what-is-variational-autoencoder-vae-tutorial/", "Jaan Altosaar", "vae variational autoencoder elbo reparameterization latent generative", "Explains the VAE with pictures before the equations."),
    R("From Autoencoder to Beta-VAE", "https://lilianweng.github.io/posts/2018-08-12-vae/", "Lilian Weng", "vae elbo reparameterization latent autoencoder", "A deeper follow-up covering VAE variants."),
    R("The Illustrated Stable Diffusion", "https://jalammar.github.io/illustrated-stable-diffusion/", "Jay Alammar", "diffusion stable diffusion latent denoising unet text image generative classifier free guidance", "Shows how noise becomes an image step by step."),
    R("The Annotated Diffusion Model", "https://huggingface.co/blog/annotated-diffusion", "Hugging Face", "diffusion ddpm noise schedule denoising unet training sampling code", "Implements DDPM in short, explained code."),
    R("What Are Diffusion Models?", "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/", "Lilian Weng", "diffusion ddpm ddim score matching classifier free guidance", "A detailed second read on forward and reverse diffusion."),
    R("Contrastive Representation Learning", "https://lilianweng.github.io/posts/2021-05-31-contrastive/", "Lilian Weng", "contrastive infonce simclr clip self supervised negative triplet", "Shows how contrastive losses fit together."),
    R("The Illustrated SimCLR Framework", "https://amitness.com/posts/simclr.html", "Amit Chaudhary", "simclr contrastive self supervised augmentation nt-xent", "Pictures of each SimCLR step."),
    R("Vision Transformer (ViT) Explained", "https://amaarora.github.io/posts/2021-01-18-ViT.html", "Aman Arora", "vision transformer vit patch embedding image attention", "Walks through patch embedding and the ViT forward pass with code."),
    R("DINO: Self-Supervised Vision Transformers (Meta AI blog)", "https://ai.meta.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/", "Meta AI", "dino self supervised distillation vision transformer teacher student", "A short authors' overview with example attention maps."),
    R("Practical BM25, Part 2", "https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables", "Elastic", "bm25 tf idf lexical retrieval inverted index ranking term frequency", "Explains each BM25 term with worked numbers."),
    R("An Overview of Late Interaction Retrieval Models", "https://weaviate.io/blog/late-interaction-overview", "Weaviate", "colbert late interaction maxsim retrieval multi vector rerank", "Compares ColBERT-style retrieval with other designs."),
    R("What Is ColBERT and Late Interaction?", "https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/", "Jina AI", "colbert late interaction maxsim token embedding", "A short explanation of MaxSim with diagrams."),
    R("SPLADE for Sparse Vector Search Explained", "https://www.pinecone.io/learn/splade/", "Pinecone", "splade sparse lexical expansion retrieval learned sparse", "Shows how a language model turns text into sparse weights."),
    R("Hierarchical Navigable Small Worlds (HNSW)", "https://www.pinecone.io/learn/series/faiss/hnsw/", "Pinecone", "hnsw approximate nearest neighbor vector index recall latency graph", "Explains layered graph search with diagrams."),
    R("Product Quantization", "https://www.pinecone.io/learn/series/faiss/product-quantization/", "Pinecone", "product quantization ivf vector compression nearest neighbor index faiss", "Shows how vectors are compressed into short codes."),
    R("Retrieval Augmented Generation", "https://www.pinecone.io/learn/retrieval-augmented-generation/", "Pinecone", "rag retrieval augmented generation embedding chunk vector database", "Explains the full RAG pipeline in plain steps."),
    R("System Design for Recommendations and Search", "https://eugeneyan.com/writing/system-design-for-discovery/", "Eugene Yan", "recommendation search ranking retrieval system design two stage candidate ads", "Compares real-world designs from large companies."),
    R("Rules of Machine Learning", "https://developers.google.com/machine-learning/guides/rules-of-ml", "Google", "ml system design production monitoring pipeline features baseline launch", "Practical rules for shipping ML without surprises."),
    R("Data Distribution Shifts and Monitoring", "https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html", "Chip Huyen", "drift data distribution shift monitoring production covariate concept", "A clear taxonomy of drift and how to detect it."),
    R("Real-Time Machine Learning: Challenges and Solutions", "https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html", "Chip Huyen", "streaming online inference feature store serving latency real time train serve", "Explains streaming features and online prediction trade-offs."),
    R("Continuous Delivery for Machine Learning", "https://martinfowler.com/articles/cd4ml.html", "Martin Fowler", "mlops testing deployment pipeline versioning continuous delivery reproducibility", "Shows how to make ML releases repeatable."),
    R("Made With ML", "https://madewithml.com/", "Made With ML", "mlops testing deployment monitoring data quality experiment tracking production", "An end-to-end guide to building reliable ML applications."),
    R("LLM Powered Autonomous Agents", "https://lilianweng.github.io/posts/2023-06-23-agent/", "Lilian Weng", "agent tool planning memory reflection workflow reasoning", "A readable survey of planning, memory, and tool use."),
    R("Building Effective Agents", "https://www.anthropic.com/research/building-effective-agents", "Anthropic", "agent workflow tool use routing evaluator orchestrator parallel", "Practical patterns for building reliable agents."),
    R("An Even Easier Introduction to CUDA", "https://developer.nvidia.com/blog/even-easier-introduction-cuda/", "NVIDIA", "cuda gpu kernel thread block grid parallel", "A gentle first kernel with threads and blocks explained."),
    R("Using Shared Memory in CUDA C/C++", "https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/", "NVIDIA", "cuda shared memory coalescing tiling bank conflicts transpose", "Shows why and how to stage data in shared memory."),
    R("How to Optimize a CUDA Matmul Kernel", "https://siboehm.com/articles/22/CUDA-MMM", "Simon Boehm", "cuda matmul tiling occupancy coalescing warp performance gpu optimization", "Improves one kernel step by step with clear explanations."),
    R("How to Think About GPUs and Rooflines (How to Scale Your Model)", "https://jax-ml.github.io/scaling-book/roofline/", "Google DeepMind", "roofline arithmetic intensity compute bound memory bandwidth flops", "Explains roofline analysis with worked numbers."),
    R("How to Train Really Large Models on Many GPUs?", "https://lilianweng.github.io/posts/2021-09-25-train-large/", "Lilian Weng", "distributed ddp fsdp tensor parallel pipeline parallel data parallel zero all reduce", "Compares every parallelism scheme with diagrams."),
    R("ZeRO and DeepSpeed (Microsoft Research blog)", "https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/", "Microsoft Research", "zero deepspeed optimizer state sharding memory fsdp", "The authors explain how sharding removes memory duplication."),
    R("Introducing PyTorch Fully Sharded Data Parallel (FSDP)", "https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/", "PyTorch", "fsdp fully sharded data parallel distributed memory sharding", "Explains what FSDP shards and when it helps."),
    R("A Visual Guide to Quantization", "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization", "Maarten Grootendorst", "quantization int8 int4 fp8 gptq smoothquant calibration outlier post training", "Over 50 diagrams that build quantization from numbers up."),
    R("Making LLMs Lighter With GPTQ", "https://huggingface.co/blog/gptq-integration", "Hugging Face", "gptq quantization int4 post training weight only compression", "Explains GPTQ with practical code."),
    R("A Friendly Introduction to Differential Privacy", "https://desfontain.es/blog/friendly-intro-to-differential-privacy.html", "Damien Desfontaines", "differential privacy epsilon delta noise laplace gaussian mechanism composition", "A multi-part series that needs no background."),
    R("Opacus: Training With Differential Privacy", "https://opacus.ai/tutorials/intro_to_advanced_features", "Meta / Opacus", "differential privacy dp sgd clipping noise accountant pytorch", "Shows DP-SGD clipping and noise in working code."),
    R("Forecasting: Principles and Practice", "https://otexts.com/fpp3/", "Hyndman and Athanasopoulos", "time series forecasting arima ets cross validation hierarchy reconciliation mape mase seasonality", "A free, friendly online textbook with plots on every page."),
    R("A Gentle Introduction to Graph Neural Networks", "https://distill.pub/2021/gnn-intro/", "Distill", "graph neural network gnn message passing node link prediction", "Interactive pictures of message passing."),
    R("Understanding Convolutions on Graphs", "https://distill.pub/2021/understanding-gnns/", "Distill", "graph convolution gcn gnn message passing", "Compares graph convolution variants visually."),
    R("Machine Learning Crash Course", "https://developers.google.com/machine-learning/crash-course", "Google", "machine learning basics regression classification overfitting embedding neural network", "Short lessons with exercises for core ML ideas."),
    R('FlashAttention: Making Attention I/O-Aware (Hugging Face blog)', 'https://huggingface.co/blog/garg-aayush/flash-attention', 'Aayush Garg', 'flashattention online softmax tiling recomputation logsumexp hbm sram attention', 'Derives tiling, recomputation, and online softmax step by step.'),
    R('Online Softmax, Derived', 'https://jinseok-moon.github.io/p/online-softmax/', 'Jinseok Moon', 'online softmax safe softmax streaming running max normalizer single pass', 'A short derivation of the one-pass update from safe softmax.'),
    R('The Log-Sum-Exp Trick', 'https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/', 'Gregory Gundersen', 'log-sum-exp logsumexp overflow underflow numerical stability', 'Explains why naive exp overflows and how shifting by the max fixes it.'),
    R('Softmax and Numerical Stability', 'https://theorempath.com/topics/softmax-and-numerical-stability', 'TheoremPath', 'softmax log-softmax numerical stability overflow log-sum-exp cross entropy stable', 'Shows stable softmax and log-softmax and why the shift is exact.'),
    R('The Softmax Function and Its Derivative', 'https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/', 'Eli Bendersky', 'softmax derivative jacobian cross entropy gradient', 'Derives the softmax Jacobian and the cross-entropy gradient.'),
    R('CS231n: Linear Classification (SVM and Softmax)', 'https://cs231n.github.io/linear-classify/', 'Stanford CS231n', 'hinge loss svm softmax cross entropy margin linear classifier regularization', 'Compares hinge loss and softmax loss with gradients and worked numbers.'),
    R('CS231n: Neural Networks Part 2 (Initialization, Regularization)', 'https://cs231n.github.io/neural-networks-2/', 'Stanford CS231n', 'dropout regularization initialization batchnorm weight decay preprocessing', 'Explains initialization, dropout, and regularization in detail.'),
    R('CS231n: k-Nearest Neighbor', 'https://cs231n.github.io/classification/', 'Stanford CS231n', 'knn nearest neighbor distance metric l1 l2 cross validation classification', 'Explains kNN, distance choices, and validation of k.'),
    R('Data Indexing and Selection (pandas)', 'https://jakevdp.github.io/PythonDataScienceHandbook/03.02-data-indexing-and-selection.html', 'Jake VanderPlas', 'pandas loc iloc indexing selection series dataframe boolean mask', 'Explains loc, iloc, and implicit versus explicit indexing with examples.'),
    R('Introducing pandas Objects', 'https://jakevdp.github.io/PythonDataScienceHandbook/03.01-introducing-pandas-objects.html', 'Jake VanderPlas', 'pandas series dataframe index object dtype construct', 'What Series, DataFrame, and Index are and how to build them.'),
    R('Handling Missing Data (pandas)', 'https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html', 'Jake VanderPlas', 'pandas missing nan null fillna dropna dtype astype', 'How pandas represents, detects, and fills missing values.'),
    R('Hierarchical Indexing, stack and unstack (pandas)', 'https://jakevdp.github.io/PythonDataScienceHandbook/03.05-hierarchical-indexing.html', 'Jake VanderPlas', 'pandas multiindex stack unstack hierarchical reshape pivot melt', 'Shows MultiIndex, stack, and unstack on small tables.'),
    R('Pivot Tables (pandas)', 'https://jakevdp.github.io/PythonDataScienceHandbook/03.09-pivot-tables.html', 'Jake VanderPlas', 'pandas pivot table crosstab aggregate groupby margins', 'Builds pivot tables from groupby with a real dataset.'),
    R('Aggregations: Min, Max, and Everything In Between (NumPy)', 'https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html', 'Jake VanderPlas', 'numpy aggregate min max sum mean std axis argmin percentile', 'Explains axis-wise reductions with examples.'),
    R('Feature Engineering (scikit-learn)', 'https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html', 'Jake VanderPlas', 'feature engineering one-hot categorical imputation derived features pipeline scaling', 'Covers categorical encoding, derived features, and imputation with code.'),
    R('In Depth: Linear Regression', 'https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html', 'Jake VanderPlas', 'linear regression basis functions polynomial regularization ridge lasso', 'From simple lines to basis functions and regularization.'),
    R('In Depth: Decision Trees and Random Forests', 'https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html', 'Jake VanderPlas', 'decision tree random forest bagging overfitting ensemble', 'Shows how trees overfit and how bagging fixes it.'),
    R('Hyperparameters and Model Validation', 'https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html', 'Jake VanderPlas', 'cross validation bias variance validation curve learning curve overfit holdout', 'Explains validation curves, learning curves, and the bias-variance trade-off.'),
    R('In Depth: Naive Bayes', 'https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html', 'Jake VanderPlas', 'naive bayes gaussian multinomial generative classifier', 'Derives Naive Bayes as a generative classifier with examples.'),
    R('In Depth: k-Means Clustering', 'https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html', 'Jake VanderPlas', 'kmeans clustering expectation maximization centroid', 'Explains k-means as expectation-maximization.'),
    R('In Depth: Gaussian Mixture Models', 'https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html', 'Jake VanderPlas', 'gaussian mixture gmm em expectation maximization responsibilities bic aic', 'Explains EM for GMMs and model selection with AIC and BIC.'),
    R('In Depth: Support Vector Machines', 'https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html', 'Jake VanderPlas', 'svm support vector kernel margin soft margin rbf', 'Margins, support vectors, and the kernel trick with plots.'),
    R('In Depth: Principal Component Analysis', 'https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html', 'Jake VanderPlas', 'pca principal component reconstruction explained variance dimensionality reduction', 'Covers PCA for visualization, noise filtering, and reconstruction.'),
    R('Principal Component Analysis in 3 Simple Steps', 'https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html', 'Sebastian Raschka', 'pca eigenvector eigenvalue covariance matrix projection scatter', 'Computes PCA by hand from the covariance matrix.'),
    R('Kernel Tricks and Nonlinear Dimensionality Reduction via RBF Kernel PCA', 'https://sebastianraschka.com/Articles/2014_kernel_pca.html', 'Sebastian Raschka', 'kernel pca rbf kernel nonlinear dimensionality reduction gram matrix centering', 'Implements RBF kernel PCA step by step.'),
    R('Linear and Quadratic Discriminant Analysis (scikit-learn)', 'https://scikit-learn.org/stable/modules/lda_qda.html', 'scikit-learn', 'lda qda discriminant analysis gaussian class covariance generative', 'Gives the math of LDA and QDA with plots.'),
    R('t-SNE: Mathematical Formulation (scikit-learn)', 'https://scikit-learn.org/stable/modules/manifold.html', 'scikit-learn', 'tsne t-sne gradient kl divergence perplexity student t manifold isomap', 'Defines t-SNE affinities, objective, and optimization.'),
    R('Mahalanobis Distance Explained', 'https://www.machinelearningplus.com/statistics/mahalanobis-distance/', 'Machine Learning Plus', 'mahalanobis distance covariance outlier multivariate', 'Builds Mahalanobis distance from Euclidean distance with code.'),
    R('Learning Word Embedding', 'https://lilianweng.github.io/posts/2017-10-15-word-embedding/', 'Lilian Weng', 'word2vec skip-gram cbow glove fasttext embedding negative sampling hierarchical softmax', 'Covers skip-gram, CBOW, GloVe, and negative sampling.'),
    R('Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss', 'https://gombru.github.io/2019/04/03/ranking_loss/', 'Raul Gomez', 'triplet loss margin ranking hinge contrastive pairwise siamese', 'Untangles the many names for margin-based losses.'),
    R('Focal Loss Explained', 'https://amaarora.github.io/posts/2020-06-29-FocalLoss.html', 'Aman Arora', 'focal loss class imbalance cross entropy gamma retinanet', 'Derives focal loss from cross entropy with code.'),
    R('An Overview of Semantic Image Segmentation', 'https://jeremyjordan.me/semantic-segmentation/', 'Jeremy Jordan', 'segmentation unet u-net dice loss encoder decoder skip connection pixel', 'Explains U-Net and the Dice loss for segmentation.'),
    R('U-Net: Understanding and Implementing', 'https://amaarora.github.io/posts/2020-09-13-unet.html', 'Aman Arora', 'unet u-net encoder decoder skip connection segmentation implementation', 'Builds U-Net in PyTorch and explains each block.'),
    R('Depthwise Separable Convolutions in PyTorch', 'https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/', 'Paepper', 'depthwise separable convolution pointwise mobilenet parameters', 'Counts parameters and operations saved by depthwise separable convolutions.'),
    R('Deconvolution and Checkerboard Artifacts', 'https://distill.pub/2016/deconv-checkerboard/', 'Distill', 'upsampling deconvolution transposed convolution checkerboard pixel shuffle resize', 'Explains why upsampling layers cause artifacts and how to avoid them.'),
    R('Deriving the Batch Norm Backward Pass', 'https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html', 'Chris Yeh', 'batchnorm batch normalization backprop gradient derivation', 'A full derivation of the batch norm gradient.'),
    R('Normalization in Deep Learning (BN, LN, IN, GN)', 'https://theaisummer.com/normalization/', 'AI Summer', 'batch normalization layer normalization instance normalization group normalization rmsnorm', 'Compares the four normalization layers with diagrams and code.'),
    R('RMSNorm vs LayerNorm', 'https://sebastianraschka.com/faq/docs/rmsnorm-vs-layernorm.html', 'Sebastian Raschka', 'rmsnorm layernorm normalization llm root mean square', 'Explains exactly what RMSNorm drops from LayerNorm.'),
    R('Setting the Learning Rate of Your Neural Network', 'https://www.jeremyjordan.me/nn-learning-rate/', 'Jeremy Jordan', 'learning rate scheduler cosine annealing warm restarts step decay cyclical warmup', 'Explains schedules, cosine annealing, and restarts.'),
    R('What Is SwiGLU?', 'https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it', 'Aziz Belaweid', 'swiglu glu swish gated feed forward transformer activation', 'Derives SwiGLU and shows the implementation.'),
    R('Dive into Deep Learning: Residual Networks (ResNet)', 'https://d2l.ai/chapter_convolutional-modern/resnet.html', 'Dive into Deep Learning', 'residual connection skip resnet identity block', 'Explains residual blocks and why they ease training.'),
    R('Neural Networks and Deep Learning, Chapter 1', 'http://neuralnetworksanddeeplearning.com/chap1.html', 'Michael Nielsen', 'neural network sigmoid neuron gradient descent mnist perceptron', 'Builds a digit classifier from scratch.'),
    R('Neural Networks and Deep Learning, Chapter 2 (Backpropagation)', 'http://neuralnetworksanddeeplearning.com/chap2.html', 'Michael Nielsen', 'backpropagation chain rule gradient hadamard error delta', 'The clearest derivation of the four backprop equations.'),
    R('Neural Networks and Deep Learning, Chapter 3', 'http://neuralnetworksanddeeplearning.com/chap3.html', 'Michael Nielsen', 'cross entropy regularization dropout initialization softmax overfitting weight decay', 'Explains why cross-entropy helps and how regularization works.'),
    R('GRPO in the Hugging Face LLM Course', 'https://huggingface.co/learn/llm-course/chapter12/3b', 'Hugging Face', 'grpo group relative policy optimization advantage reward deepseek', "Walks through GRPO's group advantage and objective."),
    R('GRPO Explained (Snorkel)', 'https://snorkel.ai/grpo/', 'Snorkel AI', 'grpo advantage group baseline critic ppo', 'A short explanation of why GRPO removes the critic.'),
    R('Knowledge Distillation (Distiller docs)', 'https://intellabs.github.io/distiller/knowledge_distillation.html', 'Intel Labs', 'knowledge distillation teacher student temperature soft targets kl loss', 'Explains soft targets, temperature, and the distillation loss.'),
    R('Assisted Generation: a New Direction Toward Low-Latency Text Generation', 'https://huggingface.co/blog/assisted-generation', 'Hugging Face', 'speculative decoding assisted generation draft assistant model verification latency', 'Explains draft-and-verify decoding with timing diagrams.'),
    R('Medusa: Simple Framework for Accelerating LLM Generation', 'https://www.together.ai/blog/medusa', 'Together AI', 'medusa speculative decoding heads tree attention eagle draft', 'Explains Medusa heads and tree verification.'),
    R('Looking Back at Speculative Decoding', 'https://research.google/blog/looking-back-at-speculative-decoding/', 'Google Research', 'speculative decoding draft target acceptance sampling latency', 'The inventors explain the idea and its history.'),
    R('Evaluating the Effectiveness of LLM-Evaluators', 'https://eugeneyan.com/writing/llm-evaluators/', 'Eugene Yan', 'llm judge llm as a judge evaluator bias pairwise rubric agreement', 'A survey of LLM-judge methods, biases, and tips.'),
    R('Using LLM-as-a-Judge (Hugging Face Cookbook)', 'https://huggingface.co/learn/cookbook/en/llm_judge', 'Hugging Face', 'llm judge llm as a judge evaluation prompt rubric', 'Builds and improves an LLM judge in code.'),
    R('The RAG Triad', 'https://www.trulens.org/getting_started/core_concepts/rag_triad/', 'TruLens', 'rag triad groundedness context relevance answer relevance evaluation', 'Defines the three RAG quality checks.'),
    R('Mastering LLM Techniques: Inference Optimization', 'https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/', 'NVIDIA', 'prefill decode kv cache batching inference latency throughput memory bound', 'Explains prefill versus decode and the main speedups.'),
    R('How Continuous Batching Enables 23x Throughput', 'https://www.anyscale.com/blog/continuous-batching-llm-inference', 'Anyscale', 'continuous batching dynamic batching iteration level scheduling throughput serving', 'Compares static, dynamic, and continuous batching.'),
    R('mixup: Data-Dependent Data Augmentation', 'https://www.inference.vc/mixup-data-dependent-data-augmentation/', 'Ferenc Huszar', 'mixup cutmix data augmentation vicinal risk interpolation labels', 'Explains why mixing examples and labels regularizes.'),
    R('Reciprocal Rank Fusion (OpenSearch)', 'https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/', 'OpenSearch', 'reciprocal rank fusion rrf hybrid search rank score fusion', 'Shows RRF with a worked example.'),
    R('Learning to Rank: RankNet, LambdaRank, LambdaMART', 'https://everdark.github.io/k9/notebooks/ml/learning_to_rank/learning_to_rank.html', 'Kyle Chung', 'learning to rank ranknet lambdarank lambdamart pairwise listwise ndcg', 'Derives RankNet and LambdaRank with code.'),
    R('Speech Processing for Machine Learning: Filter Banks, MFCCs', 'https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html', 'Haytham Fayek', 'mel spectrogram log mel mfcc filter bank stft audio speech', 'Builds log-mel features step by step.'),
    R('The Benjamini-Hochberg Procedure Explained', 'https://www.r-bloggers.com/2023/07/the-benjamini-hochberg-procedure-fdr-and-p-value-adjusted-explained/', 'R-bloggers', 'multiple testing benjamini hochberg false discovery rate fdr bonferroni p value', 'Computes adjusted p-values by hand.'),
    R('Frechet Inception Distance (FID)', 'https://apxml.com/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/frechet-inception-distance-fid', 'ApX Machine Learning', 'fid frechet inception distance gan evaluation covariance gaussian features', 'Defines FID and its covariance term.'),
    R('Mutual Information and Jensen-Shannon Divergence', 'https://hackmd.io/@mljc/H1Biuwlru', 'ML Journal Club', 'mutual information jensen shannon divergence entropy kl', 'Links mutual information to JSD with an urn example.'),
    R('QR Decomposition (Sage Linear Algebra)', 'https://linear.pugetsound.edu/scla/html/section-QR.html', 'Univ. of Puget Sound', 'qr decomposition gram schmidt orthogonal orthonormal least squares', 'Builds QR from Gram-Schmidt with worked examples.'),
    R('Householder QR (LAFF)', 'https://www.cs.utexas.edu/~flame/laff/alaff/chapter03-what-you-will-learn.html', 'UT Austin LAFF', 'householder qr gram schmidt reflection orthogonal', 'Explains Householder reflections for stable QR.'),
    R('Split Conformal Prediction', 'https://r-statistics.co/Split-Conformal-Prediction.html', 'r-statistics.co', 'conformal prediction coverage calibration interval split', 'A hands-on build of split conformal intervals.'),
    R('MAPIE: Conformal Regression Theory', 'https://mapie.readthedocs.io/en/latest/theoretical_description_regression.html', 'MAPIE', 'conformal prediction jackknife cv plus interval regression', 'Defines the conformal regression methods.'),
    R('Second-Order Optimization Methods', 'https://theorempath.com/topics/second-order-optimization-methods', 'TheoremPath', 'newton method hessian second order optimization quasi newton bfgs curvature', "Explains Newton's method, its cost, and quasi-Newton fixes."),
    R('Deep Learning Book: Numerical Computation', 'https://www.deeplearningbook.org/contents/numerical.html', 'Goodfellow, Bengio, Courville', 'overflow underflow softmax conditioning gradient hessian newton', "Covers numerical stability and Newton's method."),
    R('Parallelism Methods (Hugging Face docs)', 'https://huggingface.co/docs/transformers/perf_train_gpu_many', 'Hugging Face', 'data parallel tensor parallel pipeline parallel 3d parallelism zero', 'A compact guide to combining parallelism strategies.'),
    R('Training Larger Models With 3D Parallelism (UvA notebook)', 'https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/3d_parallelism.html', 'UvA DL Course', '3d parallelism tensor parallel pipeline parallel data parallel device mesh', 'Builds data, tensor, and pipeline parallelism in code.'),
    R('Triton: Fused Softmax', 'https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html', 'Triton', 'triton fused softmax kernel reduction memory bandwidth', 'Writes a fused softmax kernel and explains the memory savings.'),
    R('Triton: Vector Addition', 'https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html', 'Triton', 'triton kernel program id block vector add masking', 'The first Triton kernel, explained line by line.'),
    R('Triton: Matrix Multiplication', 'https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html', 'Triton', 'triton matmul tiling l2 cache blocked matrix multiplication autotune', 'Blocked matmul with L2 cache reuse.'),
    R('Matrix Factorization (Google Recommendation course)', 'https://developers.google.com/machine-learning/recommendation/collaborative/matrix', 'Google', 'matrix factorization recommendation collaborative filtering embedding als', 'Explains matrix factorization with a movie example.'),
    R('Candidate Generation (Google Recommendation course)', 'https://developers.google.com/machine-learning/recommendation/overview/candidate-generation', 'Google', 'candidate generation recommendation retrieval content based collaborative', 'Introduces candidate generation and ranking stages.'),
    R('Classification: ROC and AUC (Google)', 'https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc', 'Google', 'roc auc threshold true positive rate false positive rate', 'Explains ROC and AUC step by step.'),
    R('Classification: Accuracy, Precision, Recall (Google)', 'https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall', 'Google', 'precision recall accuracy f1 imbalance confusion matrix', 'Defines each metric with a worked confusion matrix.'),
    R('Fairness: Equality of Opportunity (Google)', 'https://developers.google.com/machine-learning/crash-course/fairness/equality-of-opportunity', 'Google', 'fairness equality of opportunity equalized odds demographic parity', 'Defines equality of opportunity with examples.'),
    R('Regularization (Google ML Crash Course)', 'https://developers.google.com/machine-learning/crash-course/overfitting/regularization', 'Google', 'regularization l1 l2 weight decay overfitting lambda', 'Explains L1 and L2 regularization.'),
    R('Interpretable ML: Logistic Regression', 'https://christophm.github.io/interpretable-ml-book/logistic.html', 'Christoph Molnar', 'logistic regression odds ratio sigmoid log odds interpretation', 'Explains logistic regression and odds ratios with examples.'),
    R('Interpretable ML: Decision Trees', 'https://christophm.github.io/interpretable-ml-book/tree.html', 'Christoph Molnar', 'decision tree split cart interpretation feature importance', 'Explains CART splits with examples.'),
    R('Multi-Armed Bandit Problem and Its Solutions', 'https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/', 'Lilian Weng', 'bandit thompson sampling ucb epsilon greedy exploration contextual', 'Explains epsilon-greedy, UCB, and Thompson sampling.'),
    R('A (Long) Peek into Reinforcement Learning', 'https://lilianweng.github.io/posts/2018-02-19-rl-overview/', 'Lilian Weng', 'reinforcement learning monte carlo temporal difference td q-learning sarsa dqn bellman mdp', 'Covers MDPs, Monte Carlo, TD, Q-learning, and DQN.'),
    R('Q-Learning (HF Deep RL Course)', 'https://huggingface.co/learn/deep-rl-course/unit2/q-learning', 'Hugging Face', 'q-learning q table epsilon greedy bellman update', 'Explains the Q-learning update with a worked example.'),
    R('Monte Carlo vs Temporal Difference Learning (HF)', 'https://huggingface.co/learn/deep-rl-course/unit2/mc-vs-td', 'Hugging Face', 'monte carlo temporal difference td value return bootstrap', 'Compares the two with worked numbers.'),
    R('Deep Q-Network (HF Deep RL Course)', 'https://huggingface.co/learn/deep-rl-course/unit3/deep-q-network', 'Hugging Face', 'dqn deep q network replay buffer target network', 'Explains the DQN loss, replay buffer, and target network.'),
    R('Thompson Sampling for Bernoulli Bandits', 'https://gdmarmerola.github.io/ts-for-bernoulli-bandit/', 'Guilherme Marmerola', 'thompson sampling beta bernoulli bandit posterior exploration', 'A visual, code-first build of Thompson sampling.'),
]


STOP = set("the a an and or of to in for with using use implement explain why how what compare from given into as by on is are be does do".split())


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9+.-]*", text.lower()) if len(w) > 2 and w not in STOP}


# (pattern, two titles, priority). Broad words get low priority so a narrow topic wins.
RULES = [
    (r"positional (embedding|encoding)|sinusoidal", ["The Illustrated Transformer", "Rotary Embeddings: A Relative Revolution"], 90),
    (r"\bclip\b", ["Contrastive Representation Learning", "The Illustrated SimCLR Framework"], 104),
    (r"torch\.compile|kernel fusion|operator fusion|tensorrt|onnx|torchscript", ["Making Deep Learning Go Brrrr From First Principles", "Triton: Fused Softmax"], 120),
    (r"precision metric|recall metric|implement precision|implement recall", ["Classification: Accuracy, Precision, Recall (Google)", "Classification: ROC and AUC (Google)"], 92),
    (r"autograd\.function|custom autograd|micrograd", ["A Gentle Introduction to torch.autograd", "Neural Networks and Deep Learning, Chapter 2 (Backpropagation)"], 100),
    (r"online softmax|online normalizer|streaming softmax", ["FlashAttention: Making Attention I/O-Aware (Hugging Face blog)", "Online Softmax, Derived"], 140),
    (r"flashattention|flash attention|tiled attention", ["FlashAttention: Making Attention I/O-Aware (Hugging Face blog)", "FlashAttention-2 (Stanford CRFM blog)"], 130),
    (r"log-?softmax|stable softmax|stable log|log[- ]sum[- ]exp|logsumexp", ["The Log-Sum-Exp Trick", "Softmax and Numerical Stability"], 125),
    (r"softmax", ["The Softmax Function and Its Derivative", "CS231n: Linear Classification (SVM and Softmax)"], 75),
    (r"hinge", ["CS231n: Linear Classification (SVM and Softmax)", "Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss"], 110),
    (r"triplet|margin ranking|contrastive loss", ["Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss", "Contrastive Representation Learning"], 110),
    (r"focal loss", ["Focal Loss Explained", "Object Detection for Dummies, Part 4: Fast Detection Models"], 115),
    (r"\bsvm\b|support vector|kernel trick|pegasos|rbf kernel|polynomial kernel|kernelized", ["In Depth: Support Vector Machines", "CS231n: Linear Classification (SVM and Softmax)"], 105),
    (r"kernel pca", ["Kernel Tricks and Nonlinear Dimensionality Reduction via RBF Kernel PCA", "In Depth: Principal Component Analysis"], 125),
    (r"\bpca\b|principal component", ["In Depth: Principal Component Analysis", "Principal Component Analysis in 3 Simple Steps"], 80),
    (r"eigen|\bsvd\b|singular value", ["Explained Visually: Eigenvectors and Eigenvalues", "In Depth: Principal Component Analysis"], 60),
    (r"\bqr\b|gram[- ]schmidt|householder", ["QR Decomposition (Sage Linear Algebra)", "Householder QR (LAFF)"], 100),
    (r"newton'?s method|newton method|quasi-newton|\bbfgs\b", ["Second-Order Optimization Methods", "Deep Learning Book: Numerical Computation"], 90),
    (r"\bknn\b|k-nearest|nearest neighbou?rs?\b", ["CS231n: k-Nearest Neighbor", "In Depth: k-Means Clustering"], 100),
    (r"\bqda\b|\blda\b|discriminant analysis", ["Linear and Quadratic Discriminant Analysis (scikit-learn)", "In Depth: Naive Bayes"], 100),
    (r"naive bayes", ["In Depth: Naive Bayes", "An Intuitive (and Short) Explanation of Bayes' Theorem"], 100),
    (r"gaussian mixture|\bgmm\b|expectation[- ]maximization|\bem algorithm\b|\be-step\b|\bm-step\b|\bbic\b|\baic\b", ["In Depth: Gaussian Mixture Models", "In Depth: k-Means Clustering"], 128),
    (r"mahalanobis", ["Mahalanobis Distance Explained", "Explained Visually: Eigenvectors and Eigenvalues"], 105),
    (r"mutual information|jensen[- ]shannon|\bjsd\b", ["Mutual Information and Jensen-Shannon Divergence", "Visual Information Theory"], 105),
    (r"cross[- ]entropy|entropy|kl[- ]divergence|information theory", ["Visual Information Theory", "Neural Networks and Deep Learning, Chapter 3"], 70),
    (r"multiple testing|bonferroni|benjamini|false discovery", ["The Benjamini-Hochberg Procedure Explained", "How Not To Run an A/B Test"], 105),
    (r"conformal", ["Split Conformal Prediction", "Calibration of probabilities (scikit-learn guide)"], 110),
    (r"a/b test|ab test|hypothesis test|p[- ]value", ["How Not To Run an A/B Test", "The Benjamini-Hochberg Procedure Explained"], 95),
    (r"thompson|contextual bandit|linucb|\bbandit", ["Thompson Sampling for Bernoulli Bandits", "Multi-Armed Bandit Problem and Its Solutions"], 110),
    (r"q-learning|\bmdp\b", ["Q-Learning (HF Deep RL Course)", "A (Long) Peek into Reinforcement Learning"], 110),
    (r"monte carlo (prediction|control|return|policy)|temporal[- ]difference learning|\btd\(|td-lambda|eligibility trace", ["Monte Carlo vs Temporal Difference Learning (HF)", "A (Long) Peek into Reinforcement Learning"], 108),
    (r"\bdqn\b|deep q", ["Deep Q-Network (HF Deep RL Course)", "A (Long) Peek into Reinforcement Learning"], 112),
    (r"broadcast|outer product|outer sum|pairwise (diff|dist)", ["Broadcasting in NumPy, explained", "A Visual Intro to NumPy and Data Representation"], 70),
    (r"\bloc\b|\biloc\b|\.loc|\.iloc", ["Data Indexing and Selection (pandas)", "Introducing pandas Objects"], 80),
    (r"crosstab|cross tab|pivot_table|pivot table|\.pivot\(", ["Pivot Tables (pandas)", "Aggregation and Grouping in pandas"], 80),
    (r"unstack|\.stack\(|multiindex|\.melt\(|\bmelt\b|wide[- ]to[- ]long", ["Hierarchical Indexing, stack and unstack (pandas)", "Pivot Tables (pandas)"], 78),
    (r"groupby|group by|split-apply", ["Aggregation and Grouping in pandas", "Pivot Tables (pandas)"], 80),
    (r"pd\.merge|merge dataframes?|\bmerge\b|(inner|left|right|outer) join|pd\.concat|concatenat(e|ing) dataframes?", ["Combining Datasets: Merge and Join in pandas", "Introducing pandas Objects"], 75),
    (r"missing (value|data)|fillna|dropna|astype|change data types?", ["Handling Missing Data (pandas)", "Introducing pandas Objects"], 40),
    (r"pandas|dataframe", ["Introducing pandas Objects", "Data Indexing and Selection (pandas)"], 70),
    (r"descriptive statistic|row extreme|argmax|argmin|percentile|aggregat", ["Aggregations: Min, Max, and Everything In Between (NumPy)", "A Visual Intro to NumPy and Data Representation"], 30),
    (r"numpy|ndarray|advanced indexing|slicing|vectoriz", ["A Visual Intro to NumPy and Data Representation", "Fancy Indexing in NumPy"], 10),
    (r"cross[- ]attention", ["The Illustrated Transformer", "Visualizing Seq2seq Models With Attention"], 120),
    (r"kv[- ]?cach|pagedattention|prefix cach", ["KV Caching Explained", "vLLM and PagedAttention (vLLM blog)"], 120),
    (r"prefill|decode phase|continuous batching|dynamic batching", ["Mastering LLM Techniques: Inference Optimization", "How Continuous Batching Enables 23x Throughput"], 115),
    (r"rotary|\brope\b", ["Rotary Embeddings: A Relative Revolution", "Transformers From Scratch"], 120),
    (r"scaled dot[- ]product attention|divide by.*sqrt", ["Transformers From Scratch", "Transformer Explainer"], 120),
    (r"multi[- ]head attention|grouped[- ]query|multi[- ]query attention|self[- ]attention", ["The Illustrated Transformer", "Transformers From Scratch"], 110),
    (r"mixture of experts|\bmoe\b|expert parallel", ["Mixture of Experts Explained", "The Transformer Family Version 2.0"], 120),
    (r"mamba|state[- ]space|\bssm\b", ["A Visual Guide to Mamba and State Space Models", "The Transformer Family Version 2.0"], 120),
    (r"speculative|medusa|\beagle\b|draft model", ["Assisted Generation: a New Direction Toward Low-Latency Text Generation", "Medusa: Simple Framework for Accelerating LLM Generation"], 118),
    (r"\bqlora\b|normalfloat|\bnf4\b", ["Making LLMs Even More Accessible: bitsandbytes, 4-bit and QLoRA", "Practical Tips for Finetuning LLMs Using LoRA"], 125),
    (r"\blora\b|low[- ]rank adaptation", ["Parameter-Efficient LLM Finetuning With LoRA", "Practical Tips for Finetuning LLMs Using LoRA"], 120),
    (r"direct preference|\bdpo\b|preference optimization|\bipo\b|\bkto\b|simpo", ["Preference Tuning LLMs with DPO and Friends", "Illustrating Reinforcement Learning from Human Feedback (RLHF)"], 120),
    (r"\bgrpo\b|group relative", ["GRPO in the Hugging Face LLM Course", "GRPO Explained (Snorkel)"], 125),
    (r"rlhf|reward model", ["Illustrating Reinforcement Learning from Human Feedback (RLHF)", "Preference Tuning LLMs with DPO and Friends"], 115),
    (r"\bppo\b|actor.critic|generalized advantage|\bgae\b", ["Proximal Policy Optimization (PPO) (HF Deep RL Course)", "The 37 Implementation Details of PPO"], 120),
    (r"policy gradient|reinforce\b", ["Policy Gradient Algorithms", "Hugging Face Deep Reinforcement Learning Course"], 105),
    (r"reinforcement learning", ["Hugging Face Deep Reinforcement Learning Course", "A (Long) Peek into Reinforcement Learning"], 90),
    (r"distillation|teacher[- ]student", ["Knowledge Distillation (Distiller docs)", "DINO: Self-Supervised Vision Transformers (Meta AI blog)"], 116),
    (r"llm[- ]as[- ]a[- ]judge|llm judge|\bjudge\b", ["Evaluating the Effectiveness of LLM-Evaluators", "Using LLM-as-a-Judge (Hugging Face Cookbook)"], 110),
    (r"rag triad|groundedness", ["The RAG Triad", "Retrieval Augmented Generation"], 115),
    (r"diffusion|\bddpm\b|\bddim\b|classifier[- ]free guidance|flow matching", ["The Illustrated Stable Diffusion", "The Annotated Diffusion Model"], 100),
    (r"\bfid\b|frechet|fr\u00e9chet", ["Frechet Inception Distance (FID)", "The Illustrated Stable Diffusion"], 105),
    (r"variational autoencoder|\bvae\b|\belbo\b|reparameterization", ["Tutorial: What Is a Variational Autoencoder?", "From Autoencoder to Beta-VAE"], 100),
    (r"simclr|infonce", ["The Illustrated SimCLR Framework", "Contrastive Representation Learning"], 100),
    (r"\bdino\b", ["DINO: Self-Supervised Vision Transformers (Meta AI blog)", "Contrastive Representation Learning"], 100),
    (r"self[- ]supervised", ["Contrastive Representation Learning", "DINO: Self-Supervised Vision Transformers (Meta AI blog)"], 95),
    (r"vision transformer|\bvit\b|patch embedding|patchify", ["Vision Transformer (ViT) Explained", "The Illustrated Transformer"], 100),
    (r"\bnms\b|non[- ]max", ["Non-Maximum Suppression for Object Detection in Python", "Object Detection for Dummies, Part 1"], 110),
    (r"yolo|\bssd\b|retinanet", ["Object Detection for Dummies, Part 4: Fast Detection Models", "Focal Loss Explained"], 105),
    (r"object detection|bounding box|\biou\b|giou|diou|ciou|box (en|de)code|mask r-cnn|r-cnn", ["Object Detection for Dummies, Part 3: R-CNN Family", "Object Detection for Dummies, Part 1"], 100),
    (r"\bu-?net\b|\bdice\b|semantic segmentation|image segmentation", ["U-Net: Understanding and Implementing", "An Overview of Semantic Image Segmentation"], 100),
    (r"depthwise|separable conv|mobilenet", ["Depthwise Separable Convolutions in PyTorch", "Understanding Convolutions"], 105),
    (r"pixel ?shuffle|transposed conv|upsampl", ["Deconvolution and Checkerboard Artifacts", "Understanding Convolutions"], 100),
    (r"convolution|\bcnn\b|conv2d|pooling|receptive field", ["Understanding Convolutions", "Conv Nets: A Modular Perspective"], 60),
    (r"\bgru\b|\blstm\b|\brnn\b|recurrent", ["Understanding LSTM Networks", "The Unreasonable Effectiveness of Recurrent Neural Networks"], 90),
    (r"mel[- ]spectrogram|log-mel|spectrogram|\bmfcc\b", ["Speech Processing for Machine Learning: Filter Banks, MFCCs", "Normalization in Deep Learning (BN, LN, IN, GN)"], 100),
    (r"bm25|tf[- ]?idf|inverted index|lexical retrieval", ["Practical BM25, Part 2", "Retrieval Augmented Generation"], 100),
    (r"colbert|late interaction|maxsim", ["What Is ColBERT and Late Interaction?", "An Overview of Late Interaction Retrieval Models"], 120),
    (r"splade|learned sparse|sparse retrieval", ["SPLADE for Sparse Vector Search Explained", "Practical BM25, Part 2"], 120),
    (r"hnsw|approximate nearest|\bann\b|vector index|\bfaiss\b", ["Hierarchical Navigable Small Worlds (HNSW)", "Product Quantization"], 120),
    (r"product quantization|\bivf\b", ["Product Quantization", "Hierarchical Navigable Small Worlds (HNSW)"], 120),
    (r"reciprocal[- ]rank|\brrf\b|rank fusion", ["Reciprocal Rank Fusion (OpenSearch)", "Practical BM25, Part 2"], 118),
    (r"ranknet|lambdarank|lambdamart|learning to rank|listwise|pairwise rank", ["Learning to Rank: RankNet, LambdaRank, LambdaMART", "System Design for Recommendations and Search"], 115),
    (r"\brag\b|retrieval[- ]augmented", ["Retrieval Augmented Generation", "System Design for Recommendations and Search"], 100),
    (r"matrix factorization|collaborative filtering", ["Matrix Factorization (Google Recommendation course)", "System Design for Recommendations and Search"], 112),
    (r"recommend|two[- ]tower|candidate generation", ["Candidate Generation (Google Recommendation course)", "System Design for Recommendations and Search"], 108),
    (r"retrieval|ranking|\bmrr\b|\bndcg\b|mean average precision", ["System Design for Recommendations and Search", "Practical BM25, Part 2"], 90),
    (r"calibrat|platt scaling|temperature scaling|expected calibration", ["Calibration of probabilities (scikit-learn guide)", "Classification: Accuracy, Precision, Recall (Google)"], 100),
    (r"data leakage|leakage", ["Data Leakage in Machine Learning", "Hyperparameters and Model Validation"], 105),
    (r"cross[- ]validation|train.*test split|stratified|walk[- ]forward", ["Hyperparameters and Model Validation", "Data Leakage in Machine Learning"], 100),
    (r"\broc\b|\bauc\b", ["Classification: ROC and AUC (Google)", "Classification: Accuracy, Precision, Recall (Google)"], 95),
    (r"precision[- ]recall|\brecall\b|f1[- ]score|confusion matrix|multilabel|hamming|jaccard|balanced accuracy", ["Classification: Accuracy, Precision, Recall (Google)", "Classification: ROC and AUC (Google)"], 85),
    (r"causal inference|causal graph|confounder|collider|propensity|uplift|treatment effect", ["Causal Inference for the Brave and True", "How Not To Run an A/B Test"], 100),
    (r"time[- ]series|forecast|arima|mape|mase|seasonal", ["Forecasting: Principles and Practice", "Hyperparameters and Model Validation"], 100),
    (r"triton", ["Triton: Fused Softmax", "Triton: Matrix Multiplication"], 130),
    (r"shared memory|coalesc|bank conflict", ["Using Shared Memory in CUDA C/C++", "How to Optimize a CUDA Matmul Kernel"], 105),
    (r"\bcuda\b|gpu kernel|\bwarp\b|occupancy|atomic", ["An Even Easier Introduction to CUDA", "How to Optimize a CUDA Matmul Kernel"], 100),
    (r"roofline|arithmetic intensity|memory.bandwidth|compute.bound|flops", ["Making Deep Learning Go Brrrr From First Principles", "How to Think About GPUs and Rooflines (How to Scale Your Model)"], 100),
    (r"3d parallel|4d parallel|tensor parallel|pipeline parallel", ["Training Larger Models With 3D Parallelism (UvA notebook)", "Parallelism Methods (Hugging Face docs)"], 112),
    (r"zero[- ]?[123]\b|zero redundancy|deepspeed", ["ZeRO and DeepSpeed (Microsoft Research blog)", "How to Train Really Large Models on Many GPUs?"], 110),
    (r"\bfsdp\b", ["Introducing PyTorch Fully Sharded Data Parallel (FSDP)", "How to Train Really Large Models on Many GPUs?"], 110),
    (r"distributed (training|data)|\bddp\b|all[- ]reduce|data parallel", ["How to Train Really Large Models on Many GPUs?", "Introducing PyTorch Fully Sharded Data Parallel (FSDP)"], 100),
    (r"gptq", ["Making LLMs Lighter With GPTQ", "A Visual Guide to Quantization"], 120),
    (r"smoothquant|quantiz|\bint8\b|\bint4\b|\bfp8\b", ["A Visual Guide to Quantization", "Making LLMs Lighter With GPTQ"], 100),
    (r"differential privacy|dp[- ]sgd|privacy accountant", ["A Friendly Introduction to Differential Privacy", "Opacus: Training With Differential Privacy"], 100),
    (r"fairness|demographic parity|equalized odds|equality of opportunity|bias audit", ["Fairness: Equality of Opportunity (Google)", "Interpretable ML: SHAP"], 100),
    (r"graph neural|\bgnn\b|message passing|node2vec|link prediction", ["A Gentle Introduction to Graph Neural Networks", "Understanding Convolutions on Graphs"], 100),
    (r"\bagent|tool calling|tree[- ]of[- ]thought|program repair|verifier|durable workflow", ["Building Effective Agents", "LLM Powered Autonomous Agents"], 100),
    (r"\bdrift|monitoring", ["Data Distribution Shifts and Monitoring", "Rules of Machine Learning"], 85),
    (r"streaming|real[- ]time|feature store|train.serve", ["Real-Time Machine Learning: Challenges and Solutions", "Rules of Machine Learning"], 85),
    (r"mlops|production|deployment|system design", ["Rules of Machine Learning", "Continuous Delivery for Machine Learning"], 10),
    (r"tokeniz|\bbpe\b|wordpiece|vocabulary", ["Byte-Pair Encoding Tokenization", "The Illustrated Word2vec"], 90),
    (r"word2vec|glove|fasttext", ["The Illustrated Word2vec", "Learning Word Embedding"], 100),
    (r"embedding", ["The Illustrated Word2vec", "The Illustrated BERT, ELMo, and co."], 40),
    (r"\bbert\b|masked language", ["The Illustrated BERT, ELMo, and co.", "The Illustrated Transformer"], 90),
    (r"beam search|top[- ]k|top[- ]p|nucleus|temperature|decoding|greedy", ["How to Generate Text: Decoding Methods", "Transformer Explainer"], 102),
    (r"swiglu|geglu|\bglu\b|swish|\bsilu\b|gelu", ["What Is SwiGLU?", "The Transformer Family Version 2.0"], 100),
    (r"rmsnorm|rms norm", ["RMSNorm vs LayerNorm", "Normalization in Deep Learning (BN, LN, IN, GN)"], 112),
    (r"batch ?norm", ["Deriving the Batch Norm Backward Pass", "Normalization in Deep Learning (BN, LN, IN, GN)"], 105),
    (r"layer ?norm|group ?norm|instance ?norm|normalization", ["Normalization in Deep Learning (BN, LN, IN, GN)", "RMSNorm vs LayerNorm"], 95),
    (r"dropout", ["CS231n: Neural Networks Part 2 (Initialization, Regularization)", "Neural Networks and Deep Learning, Chapter 3"], 100),
    (r"residual|skip connection", ["Dive into Deep Learning: Residual Networks (ResNet)", "Understanding LSTM Networks"], 85),
    (r"learning rate (scheduler|schedule)|steplr|exponentiallr|cosine annealing|warm restart|sgdr|warmup", ["Setting the Learning Rate of Your Neural Network", "An Overview of Gradient Descent Optimization Algorithms"], 108),
    (r"transformer|encoder[- ]decoder|decoder[- ]only|causal attention|\bgpt\b", ["The Illustrated Transformer", "The Illustrated GPT-2"], 20),
    (r"\badam\b|\bsgd\b|momentum|adagrad|rmsprop|optimizer|learning rate", ["An Overview of Gradient Descent Optimization Algorithms", "Why Momentum Really Works"], 85),
    (r"initializ|xavier|kaiming|vanishing gradient|exploding gradient", ["Initializing Neural Networks", "A Recipe for Training Neural Networks"], 85),
    (r"autograd|computation graph|backprop|micrograd", ["Neural Networks and Deep Learning, Chapter 2 (Backpropagation)", "Calculus on Computational Graphs: Backpropagation"], 75),
    (r"pytorch|nn\.module|dataloader|nn\.sequential", ["A Gentle Introduction to torch.autograd", "A Recipe for Training Neural Networks"], 30),
    (r"gradient|jacobian|chain rule", ["The Matrix Calculus You Need for Deep Learning", "Calculus on Computational Graphs: Backpropagation"], 20),
    (r"huber|\bmse\b|mean squared|regression loss|\bmae\b", ["In Depth: Linear Regression", "Regularization (Google ML Crash Course)"], 45),
    (r"sigmoid", ["Neural Networks and Deep Learning, Chapter 1", "Interpretable ML: Logistic Regression"], 50),
    (r"k[- ]means|clustering|\bdbscan\b", ["In Depth: k-Means Clustering", "Visualizing K-Means Clustering"], 70),
    (r"t-?sne|\bumap\b", ["How to Use t-SNE Effectively", "t-SNE: Mathematical Formulation (scikit-learn)"], 70),
    (r"boost|xgboost|lightgbm|adaboost", ["How to Explain Gradient Boosting", "In Depth: Decision Trees and Random Forests"], 90),
    (r"random forest|bagging", ["In Depth: Decision Trees and Random Forests", "Interpretable ML: Decision Trees"], 90),
    (r"decision tree|gini|information gain", ["Interpretable ML: Decision Trees", "In Depth: Decision Trees and Random Forests"], 85),
    (r"logistic regression", ["Interpretable ML: Logistic Regression", "CS231n: Linear Classification (SVM and Softmax)"], 60),
    (r"linear regression|least squares|ridge|lasso|regulariz|elasticnet", ["In Depth: Linear Regression", "Regularization (Google ML Crash Course)"], 60),
    (r"overfit|underfit|bias[- ]variance|double descent", ["Hyperparameters and Model Validation", "In Depth: Linear Regression"], 60),
    (r"\bshap\b|shapley|\blime\b|interpretab|feature importance", ["Interpretable ML: SHAP", "Machine Learning Crash Course"], 60),
    (r"mixup|cutmix", ["mixup: Data-Dependent Data Augmentation", "A Recipe for Training Neural Networks"], 100),
    (r"\bpipeline\b|one-hot encod|standardiz|imputation|feature engineering|target encoding|scaler|columntransformer|column transformer|frequency encoding|cyclic encoding|robust scal|box-cox|yeo-johnson", ["Feature Engineering (scikit-learn)", "Data Leakage in Machine Learning"], 40),
    (r"bayes|posterior|\bprior\b", ["An Intuitive (and Short) Explanation of Bayes' Theorem", "In Depth: Naive Bayes"], 30),
    (r"covariance|correlation", ["Mahalanobis Distance Explained", "Explained Visually: Eigenvectors and Eigenvalues"], 35),
    (r"neural network|activation|relu", ["Neural Networks and Deep Learning, Chapter 1", "A Recipe for Training Neural Networks"], 10),
    (r"probability|distribution|expectation|variance", ["An Intuitive (and Short) Explanation of Bayes' Theorem", "Visual Information Theory"], 10),
]


def select_readings(text: str, chapter: str = "") -> list[Reading]:
    hay = f"{chapter} {text}".lower()
    words = _tokens(hay)
    by_title = {r.title: r for r in CATALOG}
    body = text.lower()
    matches = [(prio, -pos, titles) for pos, (pattern, titles, prio) in enumerate(RULES)
               if re.search(pattern, body)]
    forced = []
    if matches:
        _, _, titles = max(matches)
        forced.extend(by_title[t] for t in titles)
    scored = []
    for pos, item in enumerate(CATALOG):
        tags = _tokens(item.tags)
        overlap = len(words & tags)
        phrase_bonus = 0
        title_bonus = sum(2 for w in _tokens(item.title) if w in words)
        scored.append((overlap * 4 + phrase_bonus + title_bonus, -pos, item))
    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    chosen = list(dict.fromkeys(forced))
    for score, _, item in scored:
        if len(chosen) == 2:
            break
        if score >= 8 and item not in chosen:
            chosen.append(item)
    if len(chosen) < 2 and re.search(r"time[- ]series|forecast|demand|price", hay):
        chosen.extend(x for x in (by_title["Forecasting: Principles and Practice"], by_title["Hyperparameters and Model Validation"]) if x not in chosen)
    if len(chosen) < 2 and re.search(r"retriev|ranking|recommend", hay):
        chosen.extend(x for x in (by_title["System Design for Recommendations and Search"], by_title["Practical BM25, Part 2"]) if x not in chosen)
    for fallback in (by_title["Machine Learning Crash Course"], by_title["Neural Networks and Deep Learning, Chapter 1"]):
        if len(chosen) == 2:
            break
        if fallback not in chosen:
            chosen.append(fallback)
    return chosen[:2]


def answer_guide(prompt: str, readings: list[Reading], kind: str) -> str:
    """Create a compact study-answer contract, not a fabricated long-form answer.

    The linked sources carry the detailed derivation. The guide tells the learner
    exactly what a complete interview answer or implementation must establish.
    """
    concepts = re.findall(r"\*\*([^*]+)\*\*", prompt)
    concepts = list(dict.fromkeys(re.sub(r"`([^`]+)`", r"\1", c).strip() for c in concepts))[:7]
    focus = ", ".join(concepts) if concepts else "the named method, its assumptions, and its failure modes"
    if kind == "coding":
        body = (
            "A complete solution must define the input and output shapes, implement the core operation without hidden "
            "library shortcuts, handle the edge cases named in the prompt, and verify the result with a small deterministic "
            "example plus at least one boundary case. State time and memory complexity."
        )
    elif kind == "system-design":
        body = (
            "A complete answer must start with measurable quality, latency, throughput, cost, privacy, and freshness targets. "
            "Then trace data from collection through training, offline evaluation, serving, monitoring, feedback, and rollback. "
            "Quantify the main bottleneck and explain the tradeoff behind each component instead of listing technologies."
        )
    else:
        body = (
            "A complete answer must state the definition and assumptions, derive or explain the central mechanism, compare the "
            "alternatives named in the question on a common axis, and finish with one failure mode or boundary condition."
        )
    return (
        f"Answer focus: {focus}.\n\n{body}\n\n"
        f"Read “{readings[0].title}” first for the intuition. Then use “{readings[1].title}” to check implementation details, "
        "tradeoffs, or failure modes."
    )
