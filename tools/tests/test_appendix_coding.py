"""Numerical and boundary checks for every authored A01 implementation."""

import copy
import itertools
import math
from pathlib import Path
import runpy
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[2] / "app/content/solutions"
MODULES = {n: runpy.run_path(str(ROOT / f"q{n}.py")) for n in range(601, 653)}
torch.set_num_threads(2)


def close(actual, expected, **kwargs):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().numpy()
    if isinstance(expected, torch.Tensor):
        expected = expected.detach().numpy()
    np.testing.assert_allclose(
        actual,
        expected,
        atol=kwargs.pop("atol", 1e-7),
        rtol=kwargs.pop("rtol", 1e-6),
        **kwargs,
    )


class AppendixCodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.rng = np.random.default_rng(4)

    def test_601_attention_masks_and_gradients(self):
        fn = MODULES[601]["scaled_dot_product_attention"]
        q = torch.randn(2, 3, 4, 6, dtype=torch.double, requires_grad=True)
        mask = torch.ones(2, 1, 4, 4, dtype=torch.bool)
        mask[:, :, 0] = False
        actual = fn(q, q, q, mask=mask, causal=True)
        combined = mask & torch.ones(4, 4, dtype=torch.bool).tril()
        close(actual, F.scaled_dot_product_attention(q, q, q, attn_mask=combined))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        close(actual[:, :, 0], torch.zeros(2, 3, 6))

    def test_602_fused_mha(self):
        model = MODULES[602]["MultiHeadSelfAttention"](8, 2).double()
        x = torch.randn(2, 3, 8, dtype=torch.double)
        q, k, v = model.qkv(x).reshape(2, 3, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
        expected = model.output(
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
            .transpose(1, 2)
            .reshape(2, 3, 8)
        )
        close(model(x, causal=True), expected)

    def test_603_gqa_and_mqa(self):
        for kv_heads in (1, 2, 4):
            q = torch.randn(2, 4, 3, 5, dtype=torch.double)
            k = torch.randn(2, kv_heads, 6, 5, dtype=torch.double)
            v = torch.randn_like(k)
            result = MODULES[603]["grouped_query_attention"](q, k, v)
            expected = F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(4 // kv_heads, 1),
                v.repeat_interleave(4 // kv_heads, 1),
            )
            close(result, expected)

    def test_604_cached_equals_full_causal_decoder(self):
        model = MODULES[604]["CachedDecoderLayer"](8, 2).double().eval()
        x = torch.randn(2, 5, 8, dtype=torch.double)
        full, _ = model(x)
        first, cache = model(x[:, :3])
        second, cache = model(x[:, 3:], cache)
        close(torch.cat((first, second), 1), full)
        self.assertEqual(cache[0].shape, (2, 2, 5, 4))

    def test_605_rope_norms_and_relative_shift(self):
        q = torch.randn(2, 3, 5, 6, dtype=torch.double)
        k = torch.randn_like(q)
        fn = MODULES[605]["apply_rope"]
        qr, kr = fn(q, k, torch.arange(5))
        close(qr.square().sum(-1), q.square().sum(-1))
        qs, ks = fn(q, k, torch.arange(5) + 11)
        close(qr @ kr.transpose(-1, -2), qs @ ks.transpose(-1, -2))

    def test_606_sinusoidal_odd_width_and_offset(self):
        fn = MODULES[606]["sinusoidal_encoding"]
        full = fn(6, 5, dtype=torch.double)
        close(full[0], [0, 1, 0, 1, 0])
        close(fn(2, 5, offset=3, dtype=torch.double), full[3:5])

    def test_607_alibi_nonpower_heads_and_cache(self):
        fn = MODULES[607]["alibi_bias"]
        full = fn(3, 5, 5)
        self.assertEqual(full.shape, (1, 3, 5, 5))
        close(fn(3, 2, 5, query_offset=3), full[:, :, 3:])
        self.assertTrue(torch.isneginf(full[0, 0, 0, 1]))
        close(full[0, :, 2, 2], torch.zeros(3))

    def test_608_lora_merge_and_gradients(self):
        model = MODULES[608]["LoRALinear"](torch.nn.Linear(4, 3).double(), rank=2)
        x = torch.randn(5, 4, dtype=torch.double)
        close(model(x), model.base(x))
        model(x).sum().backward()
        self.assertIsNone(model.base.weight.grad)
        self.assertGreater(model.b.grad.abs().sum(), 0)
        with torch.no_grad():
            model.b.add_(0.2)
        close(model(x), model.merge_weights()(x))
        close(model.merge_weights()(x), model.merge_weights()(x))

    def test_609_rmsnorm(self):
        x = torch.randn(2, 3, 6, dtype=torch.double)
        for centered in (True, False):
            model = MODULES[609]["RMSNorm"](6, zero_centered_weight=centered).double()
            close(model(x), F.rms_norm(x, (6,), eps=1e-6))

    def test_610_gated_ffns(self):
        x = torch.randn(2, 3, 4)
        for name, activation in [("swiglu", F.silu), ("geglu", F.gelu)]:
            model = MODULES[610]["GatedFeedForward"](4, 7, activation=name)
            gate, up = model.gate_up(x).chunk(2, -1)
            close(model(x), model.down(activation(gate) * up))

    def test_611_router_weights_balance_and_gradient(self):
        logits = torch.randn(12, 5, requires_grad=True)
        ids, weights, loss = MODULES[611]["top_k_router"](logits, 2)
        close(weights.sum(-1), torch.ones(12))
        self.assertEqual(ids.shape, (12, 2))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_612_sampling_filters(self):
        fn = MODULES[612]["sample_tokens"]
        logits = torch.tensor([[0.0, 9.0, 1.0]])
        self.assertEqual(fn(logits, top_k=1).item(), 1)
        self.assertEqual(fn(logits, temperature=0).item(), 1)
        self.assertEqual(fn(logits, top_p=0.01).item(), 1)

    def test_613_beam_eos_stops(self):
        fn = MODULES[613]["beam_search"]
        tokens, score = fn(lambda _: torch.tensor([-10.0, 10.0]), [0], 1, width=1)
        self.assertEqual(tokens, [0, 1])
        self.assertTrue(math.isfinite(score))

    def test_614_speculative_acceptance_and_residual(self):
        fn = MODULES[614]["verify_draft"]
        emitted, accepted = fn([0], [[1.0, 0.0]], [[0.0, 1.0], [0.5, 0.5]])
        self.assertEqual((emitted, accepted), ([1], 0))
        emitted, accepted = fn([0], [[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual((emitted, accepted), ([0, 1], 1))
        # Empirical marginal of the first verified token must match p, not q.
        p = np.array([0.25, 0.75])
        q = np.array([0.8, 0.2])
        counts = np.zeros(2)
        for seed in range(4000):
            proposal = np.random.default_rng(seed + 9000).choice(2, p=q)
            output, _ = fn([proposal], [q], [p, p], seed=seed)
            counts[output[0]] += 1
        close(counts / counts.sum(), p, atol=0.025)

    def test_615_bpe_training_and_inference(self):
        m = MODULES[615]
        merges = m["train_bpe"](["low", "low", "lower"], 5)
        self.assertEqual(merges, m["train_bpe"](["low", "low", "lower"], 5))
        self.assertEqual("".join(m["encode_bpe"]("lowest", merges)), "lowest</w>")
        self.assertEqual(m["train_bpe"]([], 2), [])

    def test_616_dpo_reference_detached(self):
        m = MODULES[616]
        x = torch.zeros(3, requires_grad=True)
        ref = torch.zeros(3, requires_grad=True)
        loss = m["dpo_loss"](x, x, ref, ref)
        close(loss, math.log(2))
        loss.backward()
        self.assertIsNone(ref.grad)

    def test_617_window_boundaries(self):
        mask = MODULES[617]["sliding_window_mask"](2, 5, 2, offset=3)
        close(mask, [[0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])

    def test_618_online_softmax_and_attention(self):
        m = MODULES[618]
        x = np.array([1000, 1001, -1000, 998.0])
        close(m["online_softmax"](x, 2), torch.tensor(x).softmax(0))
        q = self.rng.normal(size=(3, 4))
        k = self.rng.normal(size=(7, 4))
        v = self.rng.normal(size=(7, 2))
        expected = F.scaled_dot_product_attention(
            torch.tensor(q), torch.tensor(k), torch.tensor(v)
        )
        close(m["online_attention"](q, k, v, 2), expected)

    def test_619_batchnorm_running_stats_and_eval(self):
        for shape in [(4, 3), (4, 3, 5), (4, 3, 5, 6)]:
            custom = MODULES[619]["BatchNorm"](3).double()
            reference = (
                torch.nn.BatchNorm2d(3) if len(shape) == 4 else torch.nn.BatchNorm1d(3)
            ).double()
            x = torch.randn(*shape, dtype=torch.double)
            close(custom(x), reference(x))
            close(custom.running_var, reference.running_var)
            custom.eval()
            reference.eval()
            close(custom(x), reference(x))

    def test_620_layernorm_forward_backward(self):
        model = MODULES[620]["LayerNorm"](4)
        x = self.rng.normal(size=(2, 3, 4))
        up = self.rng.normal(size=x.shape)
        tensor = torch.tensor(x, requires_grad=True)
        weight = torch.ones(4, dtype=torch.double, requires_grad=True)
        bias = torch.zeros(4, dtype=torch.double, requires_grad=True)
        expected = F.layer_norm(tensor, (4,), weight, bias, 1e-5)
        close(model.forward(x), expected)
        expected.backward(torch.tensor(up))
        for a, b in zip(model.backward(up), (tensor.grad, weight.grad, bias.grad)):
            close(a, b)

    def test_621_dropout_and_droppath(self):
        fn = MODULES[621]["inverted_dropout"]
        x = torch.ones(10, 4, 3)
        close(fn(x, 1), torch.zeros_like(x))
        self.assertIs(fn(x, 0.5, training=False), x)
        result = fn(x, 0.5, drop_path=True)
        self.assertTrue(all(len(torch.unique(row)) == 1 for row in result))

    def test_622_conv_against_torch(self):
        x = self.rng.normal(size=(2, 3, 6, 7))
        w = self.rng.normal(size=(4, 3, 3, 2))
        b = self.rng.normal(size=4)
        close(
            MODULES[622]["conv2d"](x, w, b, stride=2, padding=1),
            F.conv2d(
                torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=2, padding=1
            ),
        )

    def test_623_pool_backward_overlapping(self):
        x = self.rng.normal(size=(2, 3, 5, 5))
        m = MODULES[623]
        result, cache = m["max_pool2d"](x, 3, 1)
        tensor = torch.tensor(x, requires_grad=True)
        expected = F.max_pool2d(tensor, 3, 1)
        close(result, expected)
        expected.sum().backward()
        close(m["max_pool2d_backward"](np.ones_like(result), cache), tensor.grad)

    def test_624_bce_extreme_values(self):
        x = np.array([-1000.0, 0, 1000])
        y = np.array([0.0, 1.0, 0.0])
        close(
            MODULES[624]["binary_cross_entropy_with_logits"](x, y),
            F.binary_cross_entropy_with_logits(torch.tensor(x), torch.tensor(y)),
        )

    def test_625_label_smoothing(self):
        x = torch.randn(5, 4, dtype=torch.double)
        y = torch.tensor([0, 1, 2, 3, 0])
        close(
            MODULES[625]["label_smoothed_cross_entropy"](x, y, 0.2),
            F.cross_entropy(x, y, label_smoothing=0.2),
        )

    def test_626_focal_reduces_to_weighted_bce(self):
        x = torch.randn(5, dtype=torch.double)
        y = torch.tensor([0.0, 1, 0, 1, 1], dtype=torch.double)
        expected = (
            F.binary_cross_entropy_with_logits(x, y, reduction="none")
            * torch.where(y.bool(), 0.25, 0.75)
        ).mean()
        close(MODULES[626]["focal_loss"](x, y, gamma=0), expected)

    def test_627_contrastive_two_variants(self):
        fn = MODULES[627]["contrastive_loss"]
        x = torch.eye(4)
        for mode in ("clip", "simclr"):
            self.assertLess(fn(x, x, mode=mode), fn(x, x.flip(0), mode=mode))

    def test_628_triplet_missing_classes(self):
        fn = MODULES[628]["batch_hard_triplet_loss"]
        x = torch.tensor([[0.0], [1.0], [10.0], [11.0]], requires_grad=True)
        close(fn(x, torch.tensor([0, 0, 1, 1])), 0)
        zero = fn(x, torch.arange(4))
        zero.backward()
        close(x.grad, torch.zeros_like(x))

    def test_629_schedule_endpoints(self):
        fn = MODULES[629]["learning_rate"]
        kw = dict(total_steps=6, warmup_steps=2, peak_lr=1.0, minimum_lr=0.1)
        close([fn(i, **kw) for i in (0, 1, 2, 5, 6)], [0.5, 1, 1, 0.1, 0.1])

    def test_630_optimizers_match_torch(self):
        for name, reference in [
            ("MomentumSGD", torch.optim.SGD),
            ("AdamW", torch.optim.AdamW),
        ]:
            kwargs = (
                dict(lr=0.02, momentum=0.8)
                if name == "MomentumSGD"
                else dict(lr=0.02, weight_decay=0.03)
            )
            a = torch.nn.Parameter(torch.randn(3, dtype=torch.double))
            b = torch.nn.Parameter(a.detach().clone())
            custom = MODULES[630][name]([a], **kwargs)
            standard = reference([b], **kwargs)
            for _ in range(4):
                gradient = torch.randn_like(a)
                a.grad = gradient.clone()
                b.grad = gradient.clone()
                custom.step()
                standard.step()
                close(a, b)
            state = copy.deepcopy(custom.state_dict())
            custom.load_state_dict(state)

    def test_631_global_norm(self):
        a = torch.nn.Parameter(torch.zeros(2))
        b = torch.nn.Parameter(torch.zeros(1))
        a.grad = torch.tensor([3.0, 0])
        b.grad = torch.tensor([4.0])
        norm = MODULES[631]["clip_grad_norm"]([a, b], 2.5)
        close(norm, 5.0)
        close(torch.cat((a.grad, b.grad)), [1.5, 0, 2.0], atol=1e-6)

    def test_632_accumulation_partial_group(self):
        model = torch.nn.Linear(3, 2)
        reference = copy.deepcopy(model)
        x = torch.randn(7, 3)
        y = torch.tensor([0, 1, 0, 1, 1, 0, 1])
        batches = [(x[:2], y[:2]), (x[2:5], y[2:5]), (x[5:], y[5:])]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss, _ = MODULES[632]["train_epoch"](
            model, batches, optimizer, accumulation_steps=2, max_norm=None
        )
        self.assertTrue(math.isfinite(loss))
        standard = torch.optim.SGD(reference.parameters(), lr=0.01)
        for features, labels in [(x[:5], y[:5]), (x[5:], y[5:])]:
            standard.zero_grad()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                summed = F.cross_entropy(reference(features), labels)
            summed.backward()
            standard.step()
        for a, b in zip(model.parameters(), reference.parameters()):
            close(a, b, atol=3e-5)

    def test_633_silu_gradcheck(self):
        fn = MODULES[633]["silu"]
        x = torch.randn(5, dtype=torch.double, requires_grad=True)
        close(fn(x), F.silu(x))
        self.assertTrue(torch.autograd.gradcheck(fn, (x,)))

    def test_634_initialization_variance(self):
        for method, variance in [("kaiming", 2 / 500), ("xavier", 2 / 1500)]:
            w = torch.empty(1000, 500)
            MODULES[634]["initialize_linear"](w, method=method)
            self.assertLess(abs(float(w.var()) - variance), variance * 0.02)

    def test_635_quantization_one_sided_and_zero(self):
        for x in (
            torch.zeros(4),
            torch.full((4,), 3.0),
            torch.full((4,), -3.0),
            torch.linspace(-2, 5, 30),
        ):
            codes, scale, zero = MODULES[635]["quantize_int8"](x)
            self.assertLessEqual(
                float(((codes.float() - zero) * scale - x).abs().max()),
                scale * 0.501 + 1e-6,
            )

    def test_636_kmeans_duplicate_points(self):
        fn = MODULES[636]["kmeans"]
        centers, labels = fn([[1.0, 1.0]] * 5, 3)
        self.assertTrue(np.isfinite(centers).all())
        close(centers, np.ones((3, 2)))
        centers, labels = fn([[0.0], [0.1], [9.9], [10.0]], 2)
        close(np.sort(centers[:, 0]), [0.05, 9.95])

    def test_637_gmm_responsibilities(self):
        x = np.r_[self.rng.normal(-3, 0.1, (100, 2)), self.rng.normal(3, 0.1, (100, 2))]
        weights, means, variances, responsibilities = MODULES[637]["gaussian_mixture"](
            x, 2
        )
        close(weights.sum(), 1)
        close(responsibilities.sum(1), np.ones(len(x)))
        self.assertTrue((variances > 0).all())
        close(np.sort(means[:, 0]), [-3, 3], atol=0.05)

    def test_638_knn_both_tasks(self):
        fn = MODULES[638]["knn_predict"]
        close(fn([[0], [2], [10]], [1, 3, 9], [[1]], 2), [2])
        close(fn([[0], [2], [10]], [1, 0, 9], [[1]], 2, classification=True), [0])

    def test_639_regression_closed_and_minibatch(self):
        x = self.rng.normal(size=(100, 2))
        y = 2 + x @ np.array([3.0, -1.0])
        fn = MODULES[639]["linear_regression"]
        for method in ("closed", "sgd"):
            intercept, weight = fn(x, y, method=method, epochs=300, learning_rate=0.05)
            close(weight, [3, -1], atol=1e-5)
            close(intercept, 2, atol=1e-5)

    def test_640_proximal_logistic(self):
        x = np.array([[-2, 0], [-1, 0], [1, 0], [2, 0]])
        w, b = MODULES[640]["logistic_regression"](x, [0, 0, 1, 1], l1=0.01)
        self.assertEqual(w[1], 0)
        close(x @ w + b > 0, [0, 0, 1, 1])

    def test_641_cart_continuous_labels(self):
        model = MODULES[641]["DecisionTree"](max_depth=3).fit(
            [[0], [1], [2], [3]], ["a", "a", "b", "b"]
        )
        self.assertEqual(model.predict([[0.5], [2.5]]).tolist(), ["a", "b"])

    def test_642_forest_reproducibility(self):
        cls = MODULES[642]["RandomForest"]
        x = np.arange(30.0)[:, None]
        y = (x[:, 0] >= 15).astype(int)
        first = cls(trees=20).fit(x, y)
        second = cls(trees=20).fit(x, y)
        close(first.predict(x), second.predict(x))
        self.assertGreater((first.predict(x) == y).mean(), 0.9)

    def test_643_boosting_reduces_squared_error(self):
        m = MODULES[643]
        x = np.linspace(-1, 1, 50)[:, None]
        y = x[:, 0] ** 2
        model = m["fit_gradient_boosting"](x, y, stages=100)
        self.assertLess(
            np.mean((m["predict_gradient_boosting"](x, model) - y) ** 2),
            np.var(y) * 0.2,
        )

    def test_644_pca_subspace(self):
        x = self.rng.normal(size=(20, 4))
        fn = MODULES[644]["pca"]
        a = fn(x, 2, method="svd")
        b = fn(x, 2, method="eigh")
        close(a[1].T @ a[1], b[1].T @ b[1])
        close(a[2], b[2])

    def test_645_linear_svm_separator(self):
        x = np.array([[-2], [-1], [1], [2]])
        w, b = MODULES[645]["linear_svm"](x, [-1, -1, 1, 1])
        close(np.sign(x @ w + b), [-1, -1, 1, 1])

    def test_646_naive_bayes_variants(self):
        cls = MODULES[646]["NaiveBayes"]
        x = np.array([[2, 0], [3, 0], [0, 2], [0, 3]])
        for kind in ("gaussian", "multinomial"):
            model = cls(kind).fit(x, [0, 0, 1, 1])
            close(model.predict(x), [0, 0, 1, 1])

    def test_647_auc_ties(self):
        y = [0, 1, 1, 0, 1]
        scores = [0.1, 0.4, 0.4, 0.8, 0.9]
        roc, pr, ap = MODULES[647]["binary_ranking_metrics"](y, scores)
        close(roc, metrics.roc_auc_score(y, scores))
        close(ap, metrics.average_precision_score(y, scores))
        precision, recall, _ = metrics.precision_recall_curve(y, scores)
        close(pr, metrics.auc(recall, precision))

    def test_648_ranking_zero_relevance(self):
        ndcg, mrr = MODULES[648]["ranking_metrics"]([[3, 2, 0], [0, 0, 0]], 2)
        close([ndcg, mrr], [0.5, 0.5])

    def test_649_stratified_partition(self):
        labels = np.array([0] * 11 + [1] * 13)
        splits = list(MODULES[649]["stratified_kfold"](labels, 4))
        close(np.sort(np.concatenate([test for _, test in splits])), np.arange(24))
        for train, test in splits:
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(np.unique(labels[test])), 2)

    def test_650_lexical_scores(self):
        model = MODULES[650]["LexicalIndex"](["cat cat dog", "fish", ""])
        for method in ("tfidf", "bm25"):
            scores = getattr(model, method)("cat")
            self.assertGreater(scores[0], scores[1])
            close(scores[1:], 0)
            close(getattr(model, method)("unknown"), [0, 0, 0])

    def test_651_iou_nms(self):
        m = MODULES[651]
        boxes = np.array([[0, 0, 2, 2], [0, 0, 2, 2], [2, 2, 3, 3]])
        close(m["pairwise_iou"](boxes, boxes), [[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        close(m["nms"](boxes, [0.8, 0.8, 0.7]), [0, 2])

    def test_652_viterbi_matches_enumeration(self):
        initial = np.log([0.6, 0.4])
        transition = np.log([[0.7, 0.3], [0.2, 0.8]])
        emission = np.log([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        path, score = MODULES[652]["viterbi"](initial, transition, emission)

        def score_path(p):
            return (
                initial[p[0]]
                + sum(emission[i, s] for i, s in enumerate(p))
                + sum(transition[a, b] for a, b in zip(p, p[1:]))
            )

        best = max(itertools.product(range(2), repeat=3), key=score_path)
        close(path, best)
        close(score, score_path(best))
