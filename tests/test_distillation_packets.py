"""Tests for the shared-chart connection-packet core.

Covers the analytic fixtures demanded by PREDICTIVE_CONNECTION_DISTILLATION.md
Section 13: exact affine-head and Bernoulli closed forms, estimator
superiority and underflow robustness of the score-moment cubic, linear
chart-change covariance, the first-kind/metric-derivative identity, zero
self-distillation defects, numerical domination of the Section 11.2 and 11.3
bounds, the Sobolev grid audit, the KL-only oscillatory escape, acceptance
gates, serialization, the Jacobian/Gram distinction, the sufficiency
decomposition, and the context-shuffled cubic control.
"""

import json
import math
import unittest
from dataclasses import replace

import numpy as np

from predictive_geometry.distillation import (
    ConnectionPacket,
    PacketProvenance,
    alpha_connection,
    build_packet,
    build_packet_from_logits,
    centered_logit_jacobian_loss,
    context_shuffled_cubics,
    levi_civita_loss,
    metric_relative_loss,
    packet_connection_bound,
    packet_from_dict,
    packet_to_dict,
    packet_transport_bound,
    quantization_error,
    raised_cubic_loss,
    sobolev_grid_audit,
    sqrt_jacobian_loss,
    sufficiency_decomposition,
    tensor_operator_norm,
)


def softmax(logits):
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / weights.sum()


def affine_family(unembedding):
    """Return ``z -> softmax(W z)`` for an (N, m) unembedding."""

    def q_fn(z):
        return softmax(unembedding @ np.atleast_1d(z))

    return q_fn


def affine_exact_moments(unembedding, z):
    """Exact (q, G, L, C) for an affine softmax family; L = C / 2."""

    probabilities = softmax(unembedding @ np.atleast_1d(z))
    centered = unembedding - probabilities @ unembedding
    metric = np.einsum("a,ai,aj->ij", probabilities, centered, centered)
    cubic = np.einsum("a,ai,aj,ak->ijk", probabilities, centered, centered, centered)
    return probabilities, metric, 0.5 * cubic, cubic


def with_test_provenance(packet):
    dimension = packet.z.size
    return replace(
        packet,
        provenance=PacketProvenance(
            teacher_model_id="test/model",
            teacher_revision="0123456789abcdef",
            tokenizer_hash="sha256:test-tokenizer",
            outcome_map_hash="sha256:test-outcomes",
            chart_id="test-chart",
            chart_bounds=tuple((-1.0, 1.0) for _ in range(dimension)),
            context_id="test-context",
            inference_dtype="float64",
            reproducible=True,
        ),
    )


def bernoulli_family():
    """The paper's boundary family ``theta -> (sin^2, cos^2)(theta/2)``."""

    def q_fn(z):
        theta = float(np.atleast_1d(z)[0])
        return np.array(
            [math.sin(theta / 2.0) ** 2, math.cos(theta / 2.0) ** 2]
        )

    return q_fn


class TestPacketIdentities(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.unembedding = rng.normal(size=(6, 2))
        self.z = np.array([0.3, -0.2])
        self.q_fn = affine_family(self.unembedding)

    def test_affine_packet_recovers_exact_moments(self):
        packet = build_packet(self.q_fn, self.z, 1e-3)
        _, metric, first_kind, cubic = affine_exact_moments(self.unembedding, self.z)
        self.assertTrue(packet.accepted)
        np.testing.assert_allclose(packet.metric, metric, atol=1e-9)
        np.testing.assert_allclose(packet.cubic, cubic, atol=1e-9)
        np.testing.assert_allclose(packet.first_kind, first_kind, atol=1e-7)

    def test_affine_exponential_connection_vanishes(self):
        packet = build_packet(self.q_fn, self.z, 1e-3)
        gamma = alpha_connection(packet.metric, packet.first_kind, packet.cubic, 1.0)
        self.assertLess(tensor_operator_norm(gamma), 1e-6)

    def test_bernoulli_closed_forms(self):
        theta = 1.0
        packet = build_packet(bernoulli_family(), [theta], 1e-4)
        # The exact first-kind tensor vanishes for this family. The gate
        # compares raised tensors in the Fisher norm against a dimensionless
        # floor, so a structurally zero coefficient is treated as numerically
        # indistinguishable from zero rather than divided by its own roundoff.
        self.assertTrue(packet.accepted)
        self.assertEqual(packet.rejection_reason, "")
        np.testing.assert_allclose(packet.metric, [[1.0]], atol=1e-8)
        self.assertLess(abs(packet.first_kind[0, 0, 0]), 1e-6)
        self.assertAlmostEqual(
            packet.cubic[0, 0, 0], 2.0 / math.tan(theta), places=6
        )
        for alpha in (-1.0, 0.0, 1.0):
            gamma = alpha_connection(
                packet.metric, packet.first_kind, packet.cubic, alpha
            )
            self.assertAlmostEqual(
                gamma[0, 0, 0], -alpha / math.tan(theta), places=5
            )

    def test_first_kind_matches_metric_derivative(self):
        rng = np.random.default_rng(3)
        unembedding = rng.normal(size=(5, 1))
        q_fn = affine_family(unembedding)
        delta = 1e-4
        center = build_packet(q_fn, [0.2], 1e-3)
        plus = build_packet(q_fn, [0.2 + delta], 1e-3)
        minus = build_packet(q_fn, [0.2 - delta], 1e-3)
        metric_derivative = (plus.metric[0, 0] - minus.metric[0, 0]) / (2 * delta)
        self.assertAlmostEqual(
            center.first_kind[0, 0, 0], 0.5 * metric_derivative, places=4
        )

    def test_linear_chart_change_covariance(self):
        change = np.array([[1.2, 0.3], [-0.4, 0.9]])
        pulled = affine_family(self.unembedding @ change)
        z_tilde = np.linalg.solve(change, self.z)
        original = build_packet(self.q_fn, self.z, 1e-3)
        transformed = build_packet(pulled, z_tilde, 1e-3)
        np.testing.assert_allclose(
            transformed.metric, change.T @ original.metric @ change, atol=1e-6
        )
        np.testing.assert_allclose(
            transformed.cubic,
            np.einsum("abc,ai,bj,ck->ijk", original.cubic, change, change, change),
            atol=1e-6,
        )
        gamma = alpha_connection(
            original.metric, original.first_kind, original.cubic, -1.0
        )
        gamma_tilde = alpha_connection(
            transformed.metric, transformed.first_kind, transformed.cubic, -1.0
        )
        inverse = np.linalg.inv(change)
        np.testing.assert_allclose(
            gamma_tilde,
            np.einsum("lc,cab,ai,bj->lij", inverse, gamma, change, change),
            atol=1e-5,
        )


class TestCubicEstimators(unittest.TestCase):
    def test_score_moment_exact_where_probability_form_truncates(self):
        rng = np.random.default_rng(11)
        unembedding = rng.normal(size=(5, 1))
        q_fn = affine_family(unembedding)
        z = np.array([0.4])
        _, _, _, exact_cubic = affine_exact_moments(unembedding, z)
        packet = build_packet(q_fn, z, 5e-2)
        score_error = abs(packet.cubic[0, 0, 0] - exact_cubic[0, 0, 0])
        audit_error = abs(packet.cubic_audit[0, 0, 0] - exact_cubic[0, 0, 0])
        self.assertLess(score_error, 1e-10)
        self.assertGreater(audit_error, 1e-6)
        self.assertGreater(audit_error, 1e4 * score_error)

    def test_logit_path_prevents_silent_probability_underflow(self):
        logit_offsets = np.array([0.0, -1.0, -2.0, -120.0])
        slopes = np.array([1.0, 0.5, -0.5, 2.0])

        def q_fn(z):
            value = float(np.atleast_1d(z)[0])
            logits = logit_offsets + slopes * value
            exact = softmax(logits)
            return np.float64(np.float32(exact))

        with self.assertRaisesRegex(ValueError, "open simplex"):
            build_packet(q_fn, [0.1], 1e-2)

        def logits_fn(z):
            value = float(np.atleast_1d(z)[0])
            return np.asarray(logit_offsets + slopes * value, dtype=np.float64)

        packet = build_packet_from_logits(logits_fn, [0.1], 1e-2)
        self.assertTrue(packet.accepted)
        self.assertTrue(np.all(np.isfinite(packet.cubic)))
        probabilities = softmax(logit_offsets + slopes * 0.1)
        centered = slopes - probabilities @ slopes
        exact_cubic = probabilities @ np.power(centered, 3)
        self.assertAlmostEqual(packet.cubic[0, 0, 0], exact_cubic, places=11)
        self.assertTrue(np.all(np.isfinite(packet.cubic_audit)))

    def test_logit_path_rejects_posthoc_float32_cast(self):
        def logits_fn(z):
            return np.array([float(z[0]), 0.0], dtype=np.float32)

        with self.assertRaisesRegex(TypeError, "float64 logits"):
            build_packet_from_logits(logits_fn, [0.1], 1e-2)

    def test_logit_score_path_is_softmax_gauge_invariant(self):
        slopes = np.array([1.0, 0.2, -0.7, 0.4])
        offsets = np.array([0.3, -0.1, 0.2, -0.4])

        def base(z):
            return np.asarray(offsets + slopes * float(z[0]), dtype=np.float64)

        def gauged(z):
            value = float(z[0])
            scalar = 3.0 * value * value + math.sin(value)
            return np.asarray(base(z) + scalar, dtype=np.float64)

        original = build_packet_from_logits(base, [0.2], 1e-3)
        shifted = build_packet_from_logits(gauged, [0.2], 1e-3)
        np.testing.assert_allclose(shifted.q, original.q, atol=1e-14)
        np.testing.assert_allclose(shifted.metric, original.metric, atol=1e-9)
        np.testing.assert_allclose(shifted.first_kind, original.first_kind, atol=1e-7)
        np.testing.assert_allclose(shifted.cubic, original.cubic, atol=1e-10)

    def test_exact_logit_jvp_is_used_for_nonlinear_scores(self):
        slopes = np.array([1.0, 0.2, -0.7, 0.4])
        curvature = np.array([0.3, -0.2, 0.1, 0.5])

        def logits_fn(z):
            value = float(z[0])
            return np.asarray(slopes * value + curvature * value * value, dtype=np.float64)

        def jacobian_fn(z):
            value = float(z[0])
            return np.asarray([slopes + 2.0 * curvature * value], dtype=np.float64)

        point = 0.4
        packet = build_packet_from_logits(
            logits_fn, [point], 5e-2, logit_jacobian_fn=jacobian_fn
        )
        probabilities = softmax(logits_fn([point]))
        derivative = slopes + 2.0 * curvature * point
        centered = derivative - probabilities @ derivative
        exact_cubic = probabilities @ np.power(centered, 3)
        self.assertAlmostEqual(packet.cubic[0, 0, 0], exact_cubic, places=12)
        self.assertTrue(packet.accepted)
        self.assertIsNotNone(packet.jvp_consistency_tolerance_ratio)
        self.assertLessEqual(packet.jvp_consistency_tolerance_ratio, 1.0)

    def test_incorrect_exact_logit_jvp_is_rejected(self):
        probability = 0.3
        log_odds = math.log(probability / (1.0 - probability))

        def logits_fn(z):
            return np.array([log_odds + float(z[0]), 0.0], dtype=np.float64)

        def wrong_jacobian(_z):
            return np.zeros((1, 2), dtype=np.float64)

        packet = build_packet_from_logits(
            logits_fn,
            [0.0],
            1e-2,
            logit_jacobian_fn=wrong_jacobian,
        )
        self.assertFalse(packet.accepted)
        self.assertEqual(packet.rejection_reason, "jvp_consistency")
        self.assertAlmostEqual(packet.cubic[0, 0, 0], 0.0)
        self.assertAlmostEqual(packet.cubic_audit[0, 0, 0], 0.084, places=6)


class TestLossesAndControls(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.unembedding = rng.normal(size=(6, 2))
        self.packet = build_packet(
            affine_family(self.unembedding), [0.1, 0.2], 1e-3
        )

    def test_self_distillation_defects_vanish(self):
        self.assertEqual(metric_relative_loss(self.packet.metric, self.packet.metric), 0.0)
        self.assertEqual(levi_civita_loss(self.packet, self.packet), 0.0)
        self.assertEqual(raised_cubic_loss(self.packet, self.packet), 0.0)

    def test_jacobian_versus_gram_distinction(self):
        rng = np.random.default_rng(23)
        teacher_sqrt_jacobian = rng.normal(size=(8, 2))
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        student_sqrt_jacobian = rotation @ teacher_sqrt_jacobian
        teacher_metric = teacher_sqrt_jacobian.T @ teacher_sqrt_jacobian
        student_metric = student_sqrt_jacobian.T @ student_sqrt_jacobian
        self.assertLess(metric_relative_loss(student_metric, teacher_metric), 1e-20)
        self.assertGreater(
            sqrt_jacobian_loss(student_sqrt_jacobian, teacher_sqrt_jacobian), 0.1
        )
        self.assertGreater(
            centered_logit_jacobian_loss(
                student_sqrt_jacobian.T,
                teacher_sqrt_jacobian.T,
            ),
            0.0,
        )

    def test_centered_logit_jacobian_loss_removes_rowwise_gauge(self):
        rng = np.random.default_rng(24)
        teacher = rng.normal(size=(2, 7))
        pure_gauge = np.array([1.0, -0.2])[:, None]
        self.assertLess(
            centered_logit_jacobian_loss(teacher + pure_gauge, teacher),
            1e-28,
        )

    def test_sufficiency_decomposition_is_exactly_additive(self):
        rng = np.random.default_rng(29)
        fibers = np.array([0, 0, 1, 1, 2, 2])
        effects = rng.normal(size=(6, 2))
        transferred = np.empty_like(effects)
        for fiber in np.unique(fibers):
            transferred[fibers == fiber] = rng.normal(size=2)
        metric = np.array([[2.0, 0.3], [0.3, 1.0]])
        insufficiency, mismatch, total = sufficiency_decomposition(
            fibers, effects, transferred, metric, estimand="empirical"
        )
        self.assertGreaterEqual(insufficiency, 0.0)
        self.assertGreaterEqual(mismatch, 0.0)
        self.assertAlmostEqual(total, insufficiency + mismatch, places=12)

    def test_population_sufficiency_correction_reallocates_finite_fiber_bias(self):
        fibers = np.array([0, 0])
        effects = np.array([[-1.0], [1.0]])
        transferred = np.array([[-1.0], [-1.0]])
        insufficiency, mismatch, total = sufficiency_decomposition(
            fibers,
            effects,
            transferred,
            np.eye(1),
            estimand="population_unbiased",
        )
        self.assertAlmostEqual(insufficiency, 2.0)
        self.assertAlmostEqual(mismatch, 0.0)
        self.assertAlmostEqual(total, 2.0)
        self.assertAlmostEqual(total, insufficiency + mismatch)

    def test_context_shuffled_cubics_move_complete_context_blocks(self):
        contexts = ["a", "a", "b", "b", "c", "c", "d", "d"]
        keys = [0, 1] * 4
        strata = [0, 0, 0, 0, 1, 1, 1, 1]
        context_value = {"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0}
        cubics = [
            np.full((2, 2, 2), context_value[context] + key)
            for context, key in zip(contexts, keys)
        ]
        shuffled = context_shuffled_cubics(
            cubics, strata, contexts, keys, seed=31
        )
        for recipient in ("a", "b", "c", "d"):
            indices = [i for i, context in enumerate(contexts) if context == recipient]
            received = [float(shuffled[i][0, 0, 0]) for i in indices]
            possible_donors = (
                ("a", "b") if recipient in ("a", "b") else ("c", "d")
            )
            possible_donors = [donor for donor in possible_donors if donor != recipient]
            self.assertEqual(len(possible_donors), 1)
            donor = possible_donors[0]
            self.assertEqual(received, [context_value[donor], context_value[donor] + 1.0])

    def test_context_shuffle_rejects_singleton_stratum(self):
        cubics = [np.zeros((1, 1, 1))]
        with self.assertRaisesRegex(ValueError, "fewer than two contexts"):
            context_shuffled_cubics(cubics, [0], ["only"], [0])


class TestBounds(unittest.TestCase):
    def test_packet_connection_bound_dominates_direct_defect(self):
        rng = np.random.default_rng(37)
        m = 3
        base = rng.normal(size=(m, m))
        teacher_metric = base.T @ base + 0.5 * np.eye(m)
        student_metric = teacher_metric + 0.01 * np.eye(m)
        teacher_first = rng.normal(size=(m, m, m)) * 0.3
        teacher_first = 0.5 * (teacher_first + teacher_first.transpose(1, 0, 2))
        student_first = teacher_first + 0.005 * np.ones((m, m, m))
        teacher_cubic = rng.normal(size=(m, m, m)) * 0.2
        student_cubic = teacher_cubic - 0.004 * np.ones((m, m, m))
        floor = min(
            np.linalg.eigvalsh(teacher_metric).min(),
            np.linalg.eigvalsh(student_metric).min(),
        )
        for alpha in (-1.0, 0.0, 1.0):
            teacher_gamma = alpha_connection(
                teacher_metric, teacher_first, teacher_cubic, alpha
            )
            student_gamma = alpha_connection(
                student_metric, student_first, student_cubic, alpha
            )
            defect = tensor_operator_norm(student_gamma - teacher_gamma)
            bound = packet_connection_bound(
                floor,
                np.linalg.norm(student_metric - teacher_metric, 2),
                tensor_operator_norm(student_first - teacher_first, output_axis=2),
                tensor_operator_norm(student_cubic - teacher_cubic, output_axis=2),
                alpha,
                tensor_operator_norm(teacher_first, output_axis=2),
                tensor_operator_norm(teacher_cubic, output_axis=2),
            )
            self.assertLessEqual(defect, bound)

    def test_packet_bound_raises_the_actual_covariant_index(self):
        defect_lowered = np.array(
            [
                [[-0.52751, -0.41890, 0.73323], [-0.95448, 0.73694, -1.87282], [0.38810, -0.20251, 0.58199]],
                [[-0.95448, 0.73694, -1.87282], [0.55634, 0.36346, -0.33914], [0.35421, -0.26740, 0.61776]],
                [[0.38810, -0.20251, 0.58199], [0.35421, -0.26740, 0.61776], [1.96366, -0.21166, 1.03607]],
            ]
        )
        gamma_defect = np.einsum("lk,ijk->lij", np.eye(3), defect_lowered)
        direct = tensor_operator_norm(gamma_defect, output_axis=0)
        wrong_unfolding = tensor_operator_norm(defect_lowered, output_axis=0)
        compatible = tensor_operator_norm(defect_lowered, output_axis=2)
        self.assertGreater(direct, wrong_unfolding)
        bound = packet_connection_bound(1.0, 0.0, compatible, 0.0, 0.0, 0.0, 0.0)
        self.assertLessEqual(direct, bound + 1e-12)

    def test_transport_bound_dominates_exact_defect(self):
        offset_teacher = 0.8
        offset_student = 0.81
        length = 0.5
        alpha = 1.0

        def exact_transport(offset):
            return math.sin(offset + length) / math.sin(offset)

        true_defect = abs(
            exact_transport(offset_student) - exact_transport(offset_teacher)
        )

        grid = np.linspace(0.0, length, 11)
        fill_distance = 0.5 * (grid[1] - grid[0])
        teacher_fn = bernoulli_family()
        packet_defect = 0.0
        for point in grid:
            teacher_packet = build_packet(
                teacher_fn, [point + offset_teacher], 1e-4
            )
            student_packet = build_packet(
                teacher_fn, [point + offset_student], 1e-4
            )
            teacher_gamma = alpha_connection(
                teacher_packet.metric,
                teacher_packet.first_kind,
                teacher_packet.cubic,
                alpha,
            )
            student_gamma = alpha_connection(
                student_packet.metric,
                student_packet.first_kind,
                student_packet.cubic,
                alpha,
            )
            packet_defect = max(
                packet_defect, abs(student_gamma[0, 0, 0] - teacher_gamma[0, 0, 0])
            )

        sup_teacher = abs(1.0 / math.tan(offset_teacher))
        sup_student = abs(1.0 / math.tan(offset_student))
        holder = 1.0 / math.sin(offset_teacher) ** 2
        bound = packet_transport_bound(
            length,
            sup_teacher,
            sup_student,
            packet_defect,
            holder,
            holder,
            fill_distance,
            evidence_status="certified",
            evidence_source="analytic Bernoulli fixture",
        )
        self.assertEqual(bound.evidence_status, "certified")
        self.assertLessEqual(true_defect, bound.value)
        self.assertLess(bound.value, 1.0)

    def test_transport_bound_returns_infinity_instead_of_overflowing(self):
        bound = packet_transport_bound(
            1000.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            evidence_status="sampled",
            evidence_source="overflow regression grid",
        )
        self.assertTrue(math.isinf(bound.value))

    def test_transport_bound_requires_valid_evidence_label(self):
        with self.assertRaisesRegex(ValueError, "evidence_status"):
            packet_transport_bound(
                1.0,
                1.0,
                1.0,
                0.1,
                0.1,
                0.1,
                0.1,
                evidence_status="assumed",
                evidence_source="fixture",
            )


class TestAuditsAndGates(unittest.TestCase):
    def test_sobolev_grid_audit_matches_linear_closed_form(self):
        grid = np.linspace(0.0, 1.0, 201)
        offset = np.array([1.0, -0.5])
        slope = np.array([0.4, 0.2])
        values = offset[None, :] + grid[:, None] * slope[None, :]
        estimate = sobolev_grid_audit(values, grid[1] - grid[0], 2)
        zeroth = (
            float(offset @ offset)
            + float(offset @ slope)
            + float(slope @ slope) / 3.0
        )
        exact = zeroth + float(slope @ slope)
        self.assertAlmostEqual(estimate, exact, places=10)

    def test_sobolev_grid_audit_is_exact_for_coarse_linear_h1(self):
        values = np.array([[0.0], [0.5], [1.0]])
        self.assertAlmostEqual(sobolev_grid_audit(values, 0.5, 1), 4.0 / 3.0)

    def test_sobolev_grid_audit_rejects_underresolved_order(self):
        with self.assertRaisesRegex(ValueError, "requires at least"):
            sobolev_grid_audit(np.array([[17.0]]), 1.0, 4)

    def test_oscillatory_escape_small_kl_large_metric_gap(self):
        epsilon = 1e-3
        frequency = 2000.0

        def fast(z):
            wobble = epsilon * math.sin(frequency * float(np.atleast_1d(z)[0]))
            return np.array([0.5 + wobble, 0.5 - wobble])

        def slow(z):
            wobble = epsilon * math.sin(float(np.atleast_1d(z)[0]))
            return np.array([0.5 + wobble, 0.5 - wobble])

        for point in (0.0, 0.4, 0.9):
            fast_q = fast([point])
            slow_q = slow([point])
            kl = float(np.sum(fast_q * np.log(fast_q / slow_q)))
            self.assertLess(kl, 1e-4)
        fast_packet = build_packet(fast, [0.0], 1e-5)
        slow_packet = build_packet(slow, [0.0], 1e-5)
        self.assertGreater(
            abs(fast_packet.metric[0, 0] - slow_packet.metric[0, 0]), 10.0
        )

    def test_rank_gate_rejects_degenerate_chart(self):
        rng = np.random.default_rng(41)
        unembedding = np.column_stack([rng.normal(size=5), np.zeros(5)])
        packet = build_packet(affine_family(unembedding), [0.1, 0.2], 1e-3)
        self.assertFalse(packet.accepted)
        self.assertEqual(packet.rejection_reason, "rank")

    def test_refinement_gate_rejects_subgrid_noise(self):
        rng = np.random.default_rng(43)
        unembedding = rng.normal(size=(5, 1))
        direction = rng.normal(size=5)

        def noisy(z):
            value = float(np.atleast_1d(z)[0])
            logits = unembedding[:, 0] * value
            logits = logits + 0.01 * math.sin(1e7 * value) * direction
            return softmax(logits)

        packet = build_packet(noisy, [0.2], 1e-3)
        self.assertFalse(packet.accepted)
        self.assertEqual(packet.rejection_reason, "refinement")

    def test_refinement_gate_rejects_small_aliased_metric(self):
        step = 0.1
        frequency = 2.0 * math.pi / step
        amplitude = 1e-3 / frequency

        def aliased(z):
            wobble = amplitude * math.sin(frequency * float(z[0]))
            return np.array([0.5 + wobble, 0.5 - wobble])

        packet = build_packet(aliased, [0.0], step)
        self.assertFalse(packet.accepted)
        self.assertEqual(packet.rejection_reason, "refinement")
        self.assertGreater(packet.refinement_tolerance_ratio, 1.0)

    def test_incommensurate_audit_rejects_commensurate_logit_alias(self):
        step = 0.02
        slopes = np.array([0.0, 1.0, 3.0])
        direction = np.array([1.0, -0.2, -0.8])
        amplitude = 1e-3
        frequency = 8.0 * math.pi / step

        def logits_fn(z):
            value = float(z[0])
            return np.asarray(
                slopes * value
                + amplitude * math.sin(frequency * value) * direction,
                dtype=np.float64,
            )

        def aliased_jacobian(_z):
            return slopes[None, :].copy()

        exact_derivative = slopes + amplitude * frequency * direction
        exact_score = exact_derivative - exact_derivative.mean()
        exact_metric = float(np.mean(np.square(exact_score)))

        finite_difference_packet = build_packet_from_logits(
            logits_fn,
            [0.0],
            step,
        )
        self.assertGreater(
            abs(finite_difference_packet.metric.item() - exact_metric) / exact_metric,
            4.9,
        )
        self.assertFalse(finite_difference_packet.accepted)
        self.assertEqual(finite_difference_packet.rejection_reason, "refinement")

        false_jvp_packet = build_packet_from_logits(
            logits_fn,
            [0.0],
            step,
            logit_jacobian_fn=aliased_jacobian,
        )
        self.assertFalse(false_jvp_packet.accepted)
        self.assertEqual(false_jvp_packet.rejection_reason, "jvp_consistency")
        self.assertGreater(false_jvp_packet.jvp_consistency_tolerance_ratio, 1.0)

    def test_default_refinement_gate_rejects_small_chart_first_kind_error(self):
        step = 0.1
        phase = 0.4
        frequency = 5.0 / step
        chart_scale = 1e-4
        amplitude = chart_scale / frequency

        def oscillatory(z):
            angle = frequency * float(z[0]) + phase
            probability = 0.5 + amplitude * math.sin(angle)
            return np.array([probability, 1.0 - probability])

        probability = 0.5 + amplitude * math.sin(phase)
        first = amplitude * frequency * math.cos(phase)
        second = -(amplitude * frequency**2) * math.sin(phase)
        denominator = probability * (1.0 - probability)
        exact_first_kind = (
            first * second / denominator
            - 0.5
            * first**3
            * (1.0 - 2.0 * probability)
            / denominator**2
        )
        packet = build_packet(
            oscillatory,
            [0.0],
            step,
            metric_floor=1e-30,
        )
        relative_first_kind_error = abs(
            packet.first_kind[0, 0, 0] - exact_first_kind
        ) / abs(exact_first_kind)

        self.assertGreater(relative_first_kind_error, 0.1)
        # The dimensionless floor is far too small to rescue a genuine
        # order-ten-percent defect in the raised connection.
        self.assertEqual(packet.refinement_atol, 1e-6)
        self.assertFalse(packet.accepted)
        self.assertEqual(packet.rejection_reason, "refinement")
        self.assertGreater(packet.refinement_tolerance_ratio, 1.0)

    def test_structurally_zero_tensors_are_not_rejected_for_roundoff(self):
        """A vanishing L or C must not be compared against its own roundoff.

        The paper's boundary Bernoulli family has ``Gamma^LC = 0`` identically,
        and an antipodal head has ``L = C = 0`` at the symmetric point.  Both
        are exactly representable geometries, so both must be accepted.
        """

        for theta in (1.0, math.pi / 2.0):
            packet = build_packet(bernoulli_family(), [theta], 1e-4)
            self.assertTrue(packet.accepted, msg=f"theta={theta}")
            np.testing.assert_allclose(packet.metric, [[1.0]], atol=1e-8)

        rng = np.random.default_rng(3)
        block = rng.normal(size=(4, 2))
        packet = build_packet(
            affine_family(np.vstack([block, -block])), [0.0, 0.0], 1e-3
        )
        self.assertTrue(packet.accepted)
        self.assertLess(np.abs(packet.first_kind).max(), 1e-12)
        self.assertLess(np.abs(packet.cubic).max(), 1e-12)

    def test_refinement_verdict_is_invariant_to_chart_units(self):
        """``z -> z/s`` with ``h -> h/s`` is a relabelling, not an experiment.

        It rescales ``(G, L, C)`` by ``(s^2, s^3, s^3)``.  The refinement
        diagnostic compares invariants, so its values and verdict must not
        move with ``s``.  This deliberately inaccurate fixture would be
        rescued at small ``s`` by the former dimensionful raw-tensor floor.
        """

        base_step = 0.1
        phase = 0.4
        frequency = 5.0 / base_step
        amplitude = 1e-4 / frequency
        errors = []
        absolute_errors = []
        tolerance_ratios = []
        for scale in (1.0, 1e-2, 1e-4, 1e-6):
            def rescaled_oscillatory(z, *, _scale=scale):
                angle = frequency * _scale * float(z[0]) + phase
                probability = 0.5 + amplitude * math.sin(angle)
                return np.array([probability, 1.0 - probability])

            packet = build_packet(
                rescaled_oscillatory,
                [0.0],
                base_step / scale,
                metric_floor=1e-30,
            )
            self.assertFalse(packet.accepted, msg=f"scale={scale}")
            self.assertEqual(packet.rejection_reason, "refinement")
            errors.append(packet.refinement_error)
            absolute_errors.append(packet.refinement_absolute_error)
            tolerance_ratios.append(packet.refinement_tolerance_ratio)

        for diagnostics in (errors, absolute_errors, tolerance_ratios):
            np.testing.assert_allclose(
                diagnostics,
                np.full(len(diagnostics), diagnostics[0]),
                rtol=1e-12,
                atol=0.0,
            )

    def test_absolute_metric_floor_is_labeled_as_chart_scale(self):
        packet = build_packet(
            affine_family(np.array([[0.0], [1e-6], [-1e-6]])),
            [0.2],
            1e-3,
        )
        self.assertFalse(packet.accepted)
        self.assertEqual(packet.rejection_reason, "metric_scale")

    def test_packet_builder_rejects_invalid_simplex_maps(self):
        def negative(z):
            value = float(z[0])
            return np.array([0.5 + 0.1 * value, 0.6 - 0.05 * value, -0.1 - 0.05 * value])

        def unnormalized(z):
            value = float(z[0])
            return np.array([0.7 + 0.1 * value, 0.7 - 0.1 * value])

        for q_fn in (negative, unnormalized):
            with self.assertRaises(ValueError):
                build_packet(q_fn, [0.2], 1e-3)

    def test_packet_builder_rejects_invalid_numerical_contract(self):
        q_fn = affine_family(np.array([[0.0], [1.0], [-1.0]]))
        for step in (0.0, -1.0, math.inf):
            with self.assertRaises(ValueError):
                build_packet(q_fn, [0.2], step)

    def test_serialization_roundtrip_and_checksum(self):
        rng = np.random.default_rng(47)
        packet = with_test_provenance(
            build_packet(affine_family(rng.normal(size=(6, 2))), [0.1, -0.1], 1e-3)
        )
        payload = packet_to_dict(packet)
        restored = packet_from_dict(payload)
        self.assertIsInstance(restored, ConnectionPacket)
        np.testing.assert_allclose(restored.metric, packet.metric, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(restored.cubic, packet.cubic, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            restored.cubic_audit, packet.cubic_audit, rtol=1e-6, atol=1e-6
        )
        self.assertLess(quantization_error(packet), 1e-6)
        self.assertAlmostEqual(
            restored.serialization_quantization_error,
            payload["serialization_quantization_error"],
        )
        self.assertAlmostEqual(
            restored.serialization_metric_eigenvalue_error_bound,
            payload["serialization_metric_eigenvalue_error_bound"],
        )
        payload["metric"][0][0] += 1.0
        with self.assertRaises(ValueError):
            packet_from_dict(payload)

        metadata_payload = packet_to_dict(packet)
        metadata_payload["accepted"] = not metadata_payload["accepted"]
        with self.assertRaisesRegex(ValueError, "checksum"):
            packet_from_dict(metadata_payload)

    def test_packet_v6_checksum_has_cross_language_golden_vector(self):
        packet = with_test_provenance(ConnectionPacket(
            z=np.array([0.5]),
            step=0.125,
            q=np.array([0.25, 0.75]),
            metric=np.array([[2.0]]),
            first_kind=np.array([[[3.0]]]),
            cubic=np.array([[[4.0]]]),
            cubic_audit=np.array([[[5.0]]]),
            metric_eigenvalues=np.array([2.0]),
            refinement_error=0.0625,
            accepted=True,
            rejection_reason="",
        ))
        payload = packet_to_dict(packet, max_quantization_error=0.0)
        self.assertEqual(payload["checksum"], "2671145857178851b807996e69e8d9f0")

    def test_serialization_requires_reproducible_provenance_by_default(self):
        packet = build_packet(
            affine_family(np.array([[0.0], [1.0], [-1.0]])), [0.2], 1e-3
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            packet_to_dict(packet)

    def test_serialization_rejects_point_outside_provenance_bounds(self):
        packet = with_test_provenance(
            build_packet(
                affine_family(np.array([[0.0], [1.0], [-1.0]])), [0.2], 1e-3
            )
        )
        outside = replace(
            packet,
            provenance=replace(packet.provenance, chart_bounds=((0.3, 0.4),)),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            packet_to_dict(outside)

    def test_builder_rejects_out_of_bounds_stencil_before_teacher_call(self):
        evaluations = []
        provenance = PacketProvenance(
            teacher_model_id="test/model",
            teacher_revision="0123456789abcdef",
            tokenizer_hash="sha256:test-tokenizer",
            outcome_map_hash="sha256:test-outcomes",
            chart_id="bounded-chart",
            chart_bounds=((0.0, 1.0),),
            context_id="test-context",
            inference_dtype="float64",
            reproducible=True,
        )

        def q_fn(z):
            evaluations.append(float(z[0]))
            return softmax(np.array([0.0, float(z[0]), -float(z[0])]))

        with self.assertRaisesRegex(ValueError, "stencil.*outside"):
            build_packet(q_fn, [0.01], 0.02, provenance=provenance)
        self.assertEqual(evaluations, [])

    def test_serialization_rejects_forged_rejection_reason(self):
        packet = with_test_provenance(
            build_packet(
                affine_family(
                    np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
                ),
                [0.2, -0.1],
                1e-3,
            )
        )
        forged = replace(packet, accepted=False, rejection_reason="rank")
        with self.assertRaisesRegex(ValueError, "recomputed gate"):
            packet_to_dict(forged)

    def test_serialization_rejects_metric_made_singular_by_float32(self):
        packet = with_test_provenance(
            build_packet(
                affine_family(
                    np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
                ),
                [0.2, -0.1],
                1e-3,
            )
        )
        fragile_metric = np.array([[1.0, 1.0], [1.0, 1.0 + 4e-10]])
        fragile = replace(
            packet,
            metric=fragile_metric,
            metric_eigenvalues=np.linalg.eigvalsh(fragile_metric),
        )
        self.assertGreater(fragile.metric_eigenvalues.min(), fragile.metric_floor)
        self.assertEqual(
            np.linalg.eigvalsh(fragile_metric.astype(np.float32)).min(),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "serialization changes.*rank"):
            packet_to_dict(fragile)

    def test_serialization_rejects_nonsymmetric_accepted_metric(self):
        packet = with_test_provenance(
            build_packet(
                affine_family(
                    np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
                ),
                [0.2, -0.1],
                1e-3,
            )
        )
        bad_metric = packet.metric.copy()
        bad_metric[0, 1] += 0.1
        corrupted = replace(
            packet,
            metric=bad_metric,
            metric_eigenvalues=np.linalg.eigvalsh(bad_metric),
        )
        with self.assertRaisesRegex(ValueError, "symmetric"):
            packet_to_dict(corrupted)

    def test_serialization_enforces_quantization_tolerance(self):
        packet = with_test_provenance(
            build_packet(affine_family(np.array([[0.0], [1.0], [-1.0]])), [0.2], 1e-3)
        )
        with self.assertRaisesRegex(ValueError, "quantization"):
            packet_to_dict(packet, max_quantization_error=0.0)

    def test_quantization_error_is_invariant_to_uniform_tensor_scale(self):
        packet = build_packet(
            affine_family(np.array([[0.0], [1.0], [-1.0]])), [0.2], 1e-3
        )
        scaled = replace(
            packet,
            metric=packet.metric * 1e-12,
            first_kind=packet.first_kind * 1e-12,
            cubic=packet.cubic * 1e-12,
        )
        original_error = quantization_error(packet)
        scaled_error = quantization_error(scaled)
        # Binary32 rounding depends on the scaled mantissas, so exact equality
        # is not expected. Both errors must nevertheless remain on the same
        # relative scale; the old max(1, ||T||) rule suppressed the latter by
        # twelve orders of magnitude.
        self.assertGreater(scaled_error, 0.5 * original_error)
        self.assertLess(scaled_error, 2.0 * original_error)

    def test_serialization_rejects_wrong_schema(self):
        packet = with_test_provenance(
            build_packet(affine_family(np.array([[0.0], [1.0], [-1.0]])), [0.2], 1e-3)
        )
        payload = packet_to_dict(packet)
        payload["schema_version"] = "pcd-packet-1"
        with self.assertRaisesRegex(ValueError, "schema"):
            packet_from_dict(payload)

    def test_serialization_is_strict_json_and_rejects_nonfinite(self):
        packet = with_test_provenance(
            build_packet(affine_family(np.array([[0.0], [1.0], [-1.0]])), [0.2], 1e-3)
        )
        payload = packet_to_dict(packet)
        # The emitted body must survive a strict JSON round trip so that
        # other languages and strict parsers read identical shards.
        reparsed = json.loads(json.dumps(payload, allow_nan=False))
        np.testing.assert_allclose(
            packet_from_dict(reparsed).metric, packet.metric, rtol=1e-6, atol=1e-6
        )
        corrupted = replace(packet, refinement_error=math.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            packet_to_dict(corrupted)

    def test_large_metric_eigenvalues_use_measured_quantization_bound(self):
        packet = with_test_provenance(build_packet(
            affine_family(
                np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
            ),
            [0.2, -0.1],
            1e-3,
        ))
        angle = 0.37
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        metric = rotation @ np.diag([1.0, 1.3e8]) @ rotation.T
        large_packet = replace(
            packet,
            metric=metric,
            metric_eigenvalues=np.linalg.eigvalsh(metric),
        )
        payload = packet_to_dict(large_packet)
        restored = packet_from_dict(payload)
        self.assertGreater(
            restored.serialization_metric_eigenvalue_error_bound, 0.0
        )
        difference = np.max(
            np.abs(restored.metric_eigenvalues - np.linalg.eigvalsh(restored.metric))
        )
        self.assertLessEqual(
            difference,
            restored.serialization_metric_eigenvalue_error_bound + 1e-6,
        )

    def test_nonfinite_cubic_audit_serializes_as_null(self):
        # Squaring a tiny but strictly positive float64 probability underflows
        # to zero, so the probability-difference audit can be non-finite while
        # the score-moment primary estimator stays exact.
        offsets = np.array([0.0, -1.0, -400.0])
        slopes = np.array([1.0, 0.5, 2.0])

        def logits_fn(z):
            value = float(np.atleast_1d(z)[0])
            return np.asarray(offsets + slopes * value, dtype=np.float64)

        packet = with_test_provenance(
            build_packet_from_logits(logits_fn, [0.1], 1e-3)
        )
        self.assertTrue(packet.accepted)
        self.assertTrue(np.all(np.isfinite(packet.cubic)))
        self.assertFalse(np.all(np.isfinite(packet.cubic_audit)))

        payload = packet_to_dict(packet)
        self.assertFalse(payload["cubic_audit_finite"])
        self.assertIsNone(payload["cubic_audit"])
        restored = packet_from_dict(json.loads(json.dumps(payload, allow_nan=False)))
        self.assertTrue(np.all(np.isnan(restored.cubic_audit)))
        np.testing.assert_allclose(restored.cubic, packet.cubic, rtol=1e-6, atol=1e-6)

        inconsistent = packet_to_dict(packet)
        inconsistent["cubic_audit"] = np.zeros((1, 1, 1)).tolist()
        with self.assertRaises(ValueError):
            packet_from_dict(inconsistent)

    def test_serialization_rejects_accepted_negative_metric(self):
        packet = with_test_provenance(ConnectionPacket(
            z=np.array([0.0]),
            step=1e-3,
            q=np.array([0.5, 0.5]),
            metric=np.array([[-1.0]]),
            first_kind=np.zeros((1, 1, 1)),
            cubic=np.zeros((1, 1, 1)),
            cubic_audit=np.zeros((1, 1, 1)),
            metric_eigenvalues=np.array([-1.0]),
            refinement_error=0.0,
            accepted=True,
            rejection_reason="",
        ))
        with self.assertRaisesRegex(ValueError, "positive definite"):
            packet_to_dict(packet)


if __name__ == "__main__":
    unittest.main()
