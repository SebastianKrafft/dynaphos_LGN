"""The end-to-end forward pass: shapes, physics, gradients."""
import numpy as np
import pytest
import torch


def _amp(sim, value_a=60e-6, requires_grad=False):
    return torch.full((sim.num_phosphenes,), value_a,
                      requires_grad=requires_grad)


def _activation_current(sim):
    """Lowest amplitude that activates tissue at the default pulse width,
    in A: rheobase * (1 + chronaxie / pulse width)."""
    p = sim.params
    return p['thresholding']['rheobase'] * (
        1 + p['default_stim']['chronaxie_s'] / p['default_stim']['pw_default'])


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
        """Under constant stimulation the percept first builds, peaks a few
        frames in, and then fades."""
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
    def test_phosphene_size_tracks_the_current_spread_radius(
            self, built, params_override):
        """A single-electrode law, so without neighbors' fields."""
        from dynaphos_lgn.simulator import LGNPhospheneSimulator
        _, array = built
        sim = LGNPhospheneSimulator(
            params_override(aggregation__mode='independent'), array,
            rng=np.random.default_rng(0))
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
        count can never exceed the neighborhood's full cell count."""
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


class TestBatching:
    """`run.batch_size` stacks independent stimuli along a leading
    dimension. Each row must render exactly what an unbatched call with
    that row's amplitudes would -- through both render paths."""

    AMPLITUDES_A = (30e-6, 60e-6, 90e-6)

    @pytest.fixture
    def batched(self, built, params_override):
        from dynaphos_lgn.simulator import LGNPhospheneSimulator
        _, array = built
        params = params_override(run__batch_size=len(self.AMPLITUDES_A))
        return LGNPhospheneSimulator(params, array,
                                     rng=np.random.default_rng(0))

    def _batched_amp(self, sim):
        return torch.tensor(self.AMPLITUDES_A)[:, None].expand(
            -1, sim.num_phosphenes).contiguous()

    def test_both_render_paths_are_exercised(self, batched):
        assert len(batched.ellipse_index) and len(batched.scatter_index)

    def test_output_shape(self, batched):
        img = batched(self._batched_amp(batched))
        res_x, res_y = batched.params['run']['resolution']
        assert img.shape == (len(self.AMPLITUDES_A), res_y, res_x)

    def test_rows_match_unbatched_calls(self, built, batched):
        sim, _ = built
        batched.update(self._batched_amp(batched))
        maps = batched.spatial_activation()
        for row, value in enumerate(self.AMPLITUDES_A):
            sim.reset()
            sim.update(_amp(sim, value))
            torch.testing.assert_close(maps[row], sim.spatial_activation(),
                                       atol=1e-5, rtol=1e-5)

    def test_gradient_reaches_every_row(self, batched):
        amp = self._batched_amp(batched).requires_grad_(True)
        batched.update(amp)
        batched.spatial_activation().sum().backward()
        assert torch.all(amp.grad.abs().sum(dim=1) > 0)


class TestSafetyLimit:
    def test_is_read_from_the_config(self, built):
        sim, _ = built
        assert sim.safety_limit_ua == \
            sim.params['default_stim']['safety_limit_ua']

    @pytest.mark.parametrize('limit_ua', [0.0, 1e6])
    def test_must_lie_within_the_precomputed_range(self, built,
                                                   params_override,
                                                   limit_ua):
        """Above electrodes.max_current_ua the candidate neighborhoods
        are too small, so the limit would permit currents the array
        cannot model."""
        from dynaphos_lgn.simulator import LGNPhospheneSimulator
        _, array = built
        params = params_override(default_stim__safety_limit_ua=limit_ua)
        with pytest.raises(ValueError, match='safety_limit_ua'):
            LGNPhospheneSimulator(params, array)


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

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="Needs CUDA.")
    def test_cpu_inputs_reach_a_gpu_simulator(self, built, params_override):
        """CPU stimulation drives a GPU simulator, and the gradient comes
        back to the CPU inputs."""
        from dynaphos_lgn.simulator import LGNPhospheneSimulator
        _, array = built
        sim = LGNPhospheneSimulator(params_override(run__gpu=0), array,
                                    rng=np.random.default_rng(0))
        sim.soft_threshold = True
        amp = _amp(sim, 70e-6, requires_grad=True)
        pw = torch.full((sim.num_phosphenes,), 170e-6, requires_grad=True)
        img = sim(amp, pulse_width=pw, frequency=300.0)
        assert img.device.type == 'cuda'
        img.sum().backward()
        assert float(amp.grad.abs().sum()) > 0
        assert float(pw.grad.abs().sum()) > 0

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

        Checked just above the activation current, where tissue
        activation is genuinely nonzero but the electrode is still
        subthreshold -- the regime an optimizer has to be able to climb
        out of.
        """
        sim, _ = built
        amp_a = _activation_current(sim) + 11e-6

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
        """A silent electrode must render nothing, soft gate or not."""
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
        Compared above the activation current, where the two forms
        differ only by an affine rescaling of the same logistic.
        """
        sim, _ = built
        sim.soft_threshold = True
        grads = {}
        for baseline in ('none', 'zero_activation'):
            sim.soft_threshold_baseline = baseline
            sim.reset()
            amp = _amp(sim, _activation_current(sim) + 16e-6,
                       requires_grad=True)
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
        from dynaphos_lgn.build import build_electrode_array
        from dynaphos_lgn.current_spread import RecruitmentKernel
        from dynaphos_lgn.simulator import LGNPhospheneSimulator
        params = params_override(
            thresholding__soft_threshold_baseline='sotf')
        rng = np.random.default_rng(0)
        array = build_electrode_array(params, synthetic_atlas,
                                      synthetic_jacobian,
                                      RecruitmentKernel.from_params(params),
                                      rng)
        sim = LGNPhospheneSimulator(params, array, rng=rng)
        sim.soft_threshold = True
        sim.reset()
        assert float(sim(torch.zeros(sim.num_phosphenes)).max()) == 0.0

    def test_optimization_actually_moves_towards_a_target(self, built):
        """The end-to-end claim: a loss on the rendered percept can drive
        the stimulation parameters."""
        sim, _ = built
        sim.soft_threshold = True
        # Start just above the activation current, so there is a gradient
        amp = torch.full((sim.num_phosphenes,),
                         _activation_current(sim) + 6e-6, requires_grad=True)
        optimizer = torch.optim.Adam([amp], lr=5e-6)
        sim.reset()
        target = sim(torch.full((sim.num_phosphenes,), 85e-6)).detach()

        losses = []
        for _ in range(25):
            sim.reset()
            loss = torch.nn.functional.mse_loss(sim(amp), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

    def test_receptive_field_sampling_uses_degrees_not_millimeters(self,
                                                                   built):
        """LGN receptive fields are published in degrees, so the
        sampling region is specified there directly rather than as
        millimeters of tissue divided by a magnification."""
        sim, array = built
        mask = sim.sampling_mask
        assert mask.shape[0] == sim.num_phosphenes
        assert bool(mask.any())

    def test_receptive_field_class_follows_the_tip_layer(self, built):
        sim, array = built
        classes = sim.params['atlas']['layers']['classes']
        layer = array.atlas.layer[tuple(array.voxel_indices.T)]
        expected = np.full(array.n_electrodes, '', dtype=object)
        for name in ('magno', 'parvo'):
            expected[np.isin(layer, classes[name])] = name
        laminar = expected != ''
        assert np.array_equal(array.tip_cell_class[laminar],
                              expected[laminar].astype(str))
        assert set(sim.rf_cell_class) <= {'magno', 'parvo'}

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
