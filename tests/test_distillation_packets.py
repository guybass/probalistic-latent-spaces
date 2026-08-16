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

import math
import unittest

import numpy as np

from predictive_geometry.distillation import (
    ConnectionPacket,
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
        self.assertTrue(packet.accepted)
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
        self.assertEqual(packet.excluded_outcomes, 0)

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
        teacher_jacobian = rng.normal(size=(8, 2))
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        student_jacobian = rotation @ teacher_jacobian
        teacher_metric = teacher_jacobian.T @ teacher_jacobian
        student_metric = student_jacobian.T @ student_jacobian
        self.assertLess(metric_relative_loss(student_metric, teacher_metric), 1e-20)
        self.assertGreater(sqrt_jacobian_loss(student_jacobian, teacher_jacobian), 0.1)
        self.assertGreater(
            centered_logit_jacobian_loss(student_jacobian, teacher_jacobian), 0.0
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
            fibers, effects, transferred, metric
        )
        self.assertGreaterEqual(insufficiency, 0.0)
        self.assertGreaterEqual(mismatch, 0.0)
        self.assertAlmostEqual(total, insufficiency + mismatch, places=12)

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
        )
        self.assertLessEqual(true_defect, bound)
        self.assertLess(bound, 1.0)


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
        self.assertAlmostEqual(estimate, exact, places=2)

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
        packet = build_packet(affine_family(rng.normal(size=(6, 2))), [0.1, -0.1], 1e-3)
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
        payload["metric"][0][0] += 1.0
        with self.assertRaises(ValueError):
            packet_from_dict(payload)

        metadata_payload = packet_to_dict(packet)
        metadata_payload["accepted"] = not metadata_payload["accepted"]
        with self.assertRaisesRegex(ValueError, "checksum"):
            packet_from_dict(metadata_payload)

    def test_serialization_enforces_quantization_tolerance(self):
        packet = build_packet(affine_family(np.array([[0.0], [1.0], [-1.0]])), [0.2], 1e-3)
        with self.assertRaisesRegex(ValueError, "quantization"):
            packet_to_dict(packet, max_quantization_error=0.0)

    def test_serialization_rejects_wrong_schema(self):
        packet = build_packet(affine_family(np.array([[0.0], [1.0], [-1.0]])), [0.2], 1e-3)
        payload = packet_to_dict(packet)
        payload["schema_version"] = "pcd-packet-obsolete"
        with self.assertRaisesRegex(ValueError, "schema"):
            packet_from_dict(payload)


if __name__ == "__main__":
    unittest.main()
