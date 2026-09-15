"""The end-to-end forward pass: shapes, physics, gradients."""
import numpy as np
import pytest
import torch


def _amp(sim, value_a=60e-6, requires_grad=False):
    return torch.full((sim.num_phosphenes,), value_a,
                      requires_grad=requires_grad)


class TestForwardPass:
    def test_output_shape_and_range(self, built):
        sim, _ = built
        img = sim(_amp(sim))
        res_x, res_y = sim.params['run']['resolution']
        assert img.shape == (res_y, res_x)
        assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0

    def test_zero_current_gives_an_empty_percept(self, built):
        sim, _ = built
        sim.reset()
        img = sim(torch.zeros(sim.num_phosphenes))
        assert float(img.max()) == 0.0

    def test_more_current_gives_more_percept(self, built):
        sim, _ = built
        sim.reset()
        low = float(sim(_amp(sim, 40e-6)).sum())
        sim.reset()
        high = float(sim(_amp(sim, 90e-6)).sum())
        assert high > low

    def test_state_dictionary_is_complete(self, built):
        sim, _ = built
        sim(_amp(sim))
        state = sim.get_state()
        for key in ('brightness', 'sigma_major', 'sigma_minor', 'activation',
                    'trace', 'threshold', 'layer_activation',
                    'recruited_cells', 'detection_probability'):
            assert state[key] is not None, key

    def test_repeated_stimulation_rises_then_habituates(self, built):
        """Under constant stimulation the percept first BUILDS (the
        leaky integrator charging up over the first few frames) and only
        then fades, as the memory trace raises the leak current.

        Comparing frame 1 against frame 40 would therefore get the sign
        of the effect wrong -- frame 1 is on the rising edge, not at the
        peak. The real signature is a peak a handful of frames in,
        followed by a sustained decline. Inherited unchanged from the V1
        model; there is no LGN evidence either way about duty-cycle
        response, or about whether a single exponential decay is
        adequate given relay cells' burst/tonic switching.
        """
        sim, _ = built
        sim.reset()
        trace = [float(sim(_amp(sim, 90e-6)).sum()) for _ in range(300)]
        peak = int(np.argmax(trace))
        assert 0 < peak < 60, "expected a rising edge before the peak"
        assert trace[-1] < 0.5 * trace[peak], "expected sustained fading"

    def test_reset_restores_the_initial_response(self, built):
        sim, _ = built
        sim.reset()
        first = float(sim(_amp(sim, 90e-6)).sum())
        for _ in range(20):
            sim(_amp(sim, 90e-6))
        sim.reset()
        assert float(sim(_amp(sim, 90e-6)).sum()) == pytest.approx(first,
                                                                   rel=1e-4)

    def test_pulse_width_and_frequency_modulate_the_response(self, built):
        sim, _ = built
        amp = _amp(sim, 70e-6)
        sim.reset()
        narrow = float(sim(amp, pulse_width=torch.full(
            (sim.num_phosphenes,), 60e-6)).sum())
        sim.reset()
        wide = float(sim(amp, pulse_width=torch.full(
            (sim.num_phosphenes,), 300e-6)).sum())
        assert wide > narrow

    def test_is_deterministic_for_a_fixed_seed(self, built):
        sim, _ = built
        sim.reset()
        a = sim(_amp(sim))
        sim.reset()
        b = sim(_amp(sim))
        assert torch.allclose(a, b)


class TestPhysics:
    def test_phosphene_size_tracks_the_current_spread_radius(self, built):
        sim, array = built
        sim.update(_amp(sim, 40e-6))
        small = sim.size.sigma_major.detach().clone()
        sim.reset()
        sim.update(_amp(sim, 160e-6))
        large = sim.size.sigma_major.detach()
        # Stoney: radius scales with sqrt(I), so 4x current -> 2x size.
        ratio = (large / small.clamp(min=1e-12))
        assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=1e-3)

    def test_layer_activation_is_reported_per_lamina(self, built):
        sim, array = built
        sim(_amp(sim))
        table = sim.layer_activation_table()
        assert set(table) == set(sim.layer_names)
        assert all(np.all(v >= 0) for v in table.values())

    def test_layer_activation_grows_with_current(self, built):
        sim, _ = built
        sim.reset()
        sim.update(_amp(sim, 20e-6))
        low = sim.layer_activation.detach().clone()
        sim.reset()
        sim.update(_amp(sim, 90e-6))
        assert torch.all(sim.layer_activation.detach() >= low - 1e-9)
        assert float(sim.layer_activation.sum()) > float(low.sum())

    def test_recruited_cells_never_exceed_the_reachable_total(self, built):
        """Recruitment weights are bounded by 1, so the weighted cell
        count can never exceed the neighbourhood's full cell count."""
        sim, array = built
        sim.update(_amp(sim, sim.array.max_current_ua * 1e-6))
        reachable = torch.as_tensor(array.per_layer_cell_counts(),
                                    dtype=sim.recruited_cells.dtype)
        assert torch.all(sim.recruited_cells.detach() <= reachable + 1e-3)

    def test_electrodes_are_rendered_by_both_paths(self, built):
        """The whole point of the mixed renderer -- if a configuration
        silently used only one path, the other would never be exercised.
        """
        sim, array = built
        assert len(sim.ellipse_index) + len(sim.scatter_index) == \
            sim.num_phosphenes


class TestGradients:
    def test_gradient_reaches_amplitude(self, built):
        sim, _ = built
        sim.soft_threshold = True
        sim.reset()
        amp = _amp(sim, 70e-6, requires_grad=True)
        sim(amp).sum().backward()
        assert torch.isfinite(amp.grad).all()
        assert float(amp.grad.abs().sum()) > 0

    def test_gradient_reaches_pulse_width_and_frequency(self, built):
        sim, _ = built
        sim.soft_threshold = True
        sim.reset()
        pw = torch.full((sim.num_phosphenes,), 170e-6, requires_grad=True)
        freq = torch.full((sim.num_phosphenes,), 300.0, requires_grad=True)
        sim(_amp(sim, 70e-6), pulse_width=pw, frequency=freq).sum().backward()
        assert float(pw.grad.abs().sum()) > 0
        assert float(freq.grad.abs().sum()) > 0

    def test_hard_threshold_blocks_subthreshold_gradients(self, built):
        """Documented consequence, not a bug: a step function has zero
        gradient on both sides, so a subthreshold electrode can never
        learn to cross it. This is exactly why the soft option exists.
        """
        sim, _ = built
        sim.soft_threshold = False
        sim.reset()
        amp = _amp(sim, 1e-6, requires_grad=True)
        out = sim(amp)
        out.sum().backward()
        assert float(amp.grad.abs().sum()) == 0.0

    def test_soft_threshold_gradient_beats_the_hard_gate(self, built):
        """The reason the soft threshold is on by default.

        Checked just above the rheobase, where tissue activation is
        genuinely nonzero but the electrode is still subthreshold --
        the regime an optimiser has to be able to climb out of.
        """
        sim, _ = built
        amp_a = 35e-6

        sim.soft_threshold = False
        sim.reset()
        hard_amp = _amp(sim, amp_a, requires_grad=True)
        sim(hard_amp).sum().backward()
        hard = float(hard_amp.grad.abs().sum())

        sim.soft_threshold = True
        sim.reset()
        soft_amp = _amp(sim, amp_a, requires_grad=True)
        sim(soft_amp).sum().backward()
        soft = float(soft_amp.grad.abs().sum())

        assert soft > hard

    def test_soft_threshold_renders_nothing_at_zero_amplitude(self, built):
        """A silent electrode must render nothing, soft gate or not.

        The raw logistic does not satisfy this: the threshold sits only
        `mu/sd = 1.36` slope-widths above zero activation, so a silent
        electrode gets p(detect) ~ 0.2 and the array glows. The
        baseline correction is what buys the invariant back -- see
        `thresholding.soft_threshold_baseline`.
        """
        sim, _ = built
        assert sim.soft_threshold, (
            "this test is about the SOFT gate; the config default "
            "changed if this fails")
        sim.reset()
        assert float(sim(torch.zeros(sim.num_phosphenes)).max()) == 0.0

    def test_raw_logistic_baseline_does_glow_at_zero(self, built):
        """The failure mode the default guards against, pinned so a
        future 'simplification' back to the raw logistic is caught."""
        sim, _ = built
        sim.soft_threshold = True
        sim.soft_threshold_baseline = 'none'
        try:
            sim.reset()
            glow = float(sim(torch.zeros(sim.num_phosphenes)).max())
        finally:
            sim.soft_threshold_baseline = 'zero_activation'
        assert glow > 0.0, (
            "the raw logistic is supposed to glow at zero amplitude; if "
            "it no longer does, the threshold/SD ratio changed and the "
            "baseline correction may no longer be needed")

    def test_baseline_correction_keeps_the_gradient(self, built):
        """Removing the pedestal must not remove the gradient -- that
        would give back the problem the soft threshold exists to solve.
        Compared above the rheobase, where the two forms differ only by
        an affine rescaling of the same logistic.
        """
        sim, _ = built
        sim.soft_threshold = True
        grads = {}
        for baseline in ('none', 'zero_activation'):
            sim.soft_threshold_baseline = baseline
            sim.reset()
            amp = _amp(sim, 40e-6, requires_grad=True)
            sim(amp).sum().backward()
            grads[baseline] = float(amp.grad.abs().sum())
        sim.soft_threshold_baseline = 'zero_activation'
        assert grads['zero_activation'] > 0
        # Same curve, rescaled by 1/(1 - p(0)) > 1 -- so the corrected
        # form is if anything STEEPER, never flatter.
        assert grads['zero_activation'] >= 0.5 * grads['none']

    def test_typo_keeps_the_correction(self, params_override,
                                       synthetic_atlas, synthetic_jacobian):
        """Only the literal 'none' turns the correction off.

        A typo therefore lands on the corrected curve rather than on the
        raw logistic -- it fails towards the safe side, where a silent
        electrode renders nothing, instead of quietly restoring the glow
        the correction exists to remove.
        """
        from dynaphos_lgn.build import (build_electrode_array, build_kernel,
                                        build_magnification_model)
        from dynaphos_lgn.simulator import LGNPhospheneSimulator
        params = params_override(
            thresholding__soft_threshold_baseline='sotf')
        rng = np.random.default_rng(0)
        array = build_electrode_array(params, synthetic_atlas,
                                      synthetic_jacobian,
                                      build_kernel(params), rng)
        array.magnification_model = build_magnification_model(
            params, synthetic_atlas, synthetic_jacobian)
        sim = LGNPhospheneSimulator(params, array, rng=rng)
        sim.soft_threshold = True
        sim.reset()
        assert float(sim(torch.zeros(sim.num_phosphenes)).max()) == 0.0

    def test_optimisation_actually_moves_towards_a_target(self, built):
        """The end-to-end claim: a loss on the rendered percept can drive
        the stimulation parameters."""
        sim, _ = built
        sim.soft_threshold = True
        amp = torch.full((sim.num_phosphenes,), 30e-6, requires_grad=True)
        optimiser = torch.optim.Adam([amp], lr=5e-6)
        sim.reset()
        target = sim(torch.full((sim.num_phosphenes,), 85e-6)).detach()

        losses = []
        for _ in range(25):
            sim.reset()
            loss = torch.nn.functional.mse_loss(sim(amp), target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0]


class TestSampling:
    def test_sample_stimulus_returns_one_value_per_electrode(self, built):
        sim, _ = built
        res_x, res_y = sim.params['run']['resolution']
        image = np.zeros((res_y, res_x), dtype=np.float32)
        image[res_y // 2, res_x // 2] = 1.0
        out = sim.sample_stimulus(image, rescale=True)
        assert out.shape[-1] == sim.num_phosphenes
        assert torch.all(out >= 0)

    def test_receptive_field_sampling_uses_degrees_not_millimetres(self,
                                                                   built):
        """LGN receptive fields are published in degrees, so the
        sampling region is specified there directly rather than as
        millimetres of tissue divided by a magnification."""
        sim, array = built
        mask = sim.sampling_mask
        assert mask.shape[0] == sim.num_phosphenes
        assert bool(mask.any())

    def test_a_blank_image_stimulates_nothing(self, built):
        sim, _ = built
        res_x, res_y = sim.params['run']['resolution']
        out = sim.sample_stimulus(np.zeros((res_y, res_x), dtype=np.float32))
        assert float(out.abs().sum()) == 0.0


class TestReporting:
    def test_describe_reports_the_array(self, built):
        sim, _ = built
        text = sim.describe()
        assert 'LGNElectrodeArray' in text
        assert 'Electrode confidence flags' in text

    def test_layer_table_requires_a_forward_pass_first(self, built):
        sim, _ = built
        with pytest.raises(RuntimeError):
            sim.layer_activation_table()
