use std::borrow::Cow;

use crate::{Complex, DenoiseParams, RnnModel, FRAME_SIZE, FREQ_SIZE, NB_BANDS};

/// This is the low-level entry-point into `nnnoiseless`: by using the `DenoiseState` directly,
/// you can denoise your audio while keeping copying to a minimum. For a higher-level
/// denoising experience, try [`DenoiseSignal`](crate::DenoiseSignal); for several channels at
/// once, see [`MultiDenoiser`](crate::MultiDenoiser).
///
/// This struct directly contains various memory buffers that are used while denoising. As such,
/// this is quite a large struct, and should probably be kept behind some kind of pointer.
///
/// # Example
///
/// ```rust
/// # use nnnoiseless::DenoiseState;
/// // One second of 440Hz sine wave at 48kHz sample rate. Note that the input data consists of
/// // `f32`s, but the values should be in the range of an `i16`.
/// let sine: Vec<_> = (0..48_000)
///     .map(|x| (x as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48_000.0).sin() * i16::MAX as f32)
///     .collect();
/// let mut output = Vec::new();
/// let mut out_buf = [0.0; DenoiseState::FRAME_SIZE];
/// let mut denoise = DenoiseState::new();
/// let mut first = true;
/// for chunk in sine.chunks_exact(DenoiseState::FRAME_SIZE) {
///     denoise.process_frame(&mut out_buf[..], chunk);
///
///     // We throw away the first output, as discussed in the documentation for
///     //`DenoiseState::process_frame`.
///     if !first {
///         output.extend_from_slice(&out_buf[..]);
///     }
///     first = false;
/// }
/// ```
#[derive(Clone)]
pub struct DenoiseState<'model> {
    params: DenoiseParams,
    /// Most recent gains that we applied.
    lastg: [f32; NB_BANDS],
    /// Gains for the frame currently being analysed.
    gains: [f32; NB_BANDS],
    rnn: crate::rnn::RnnState<'model>,
    feat: crate::features::DenoiseFeatures,
    /// Voice-activity probability for the frame `analyze` most recently looked at.
    vad_scratch: f32,

    /// Delayed spectra, gains, voice-activity probabilities and silence flags. All empty
    /// unless [`DenoiseParams::lookahead`] asked for them.
    spec_ring: Vec<Vec<Complex>>,
    gain_ring: Vec<[f32; NB_BANDS]>,
    vad_ring: Vec<f32>,
    silence_ring: Vec<bool>,
    ring_pos: usize,
}

impl DenoiseState<'static> {
    /// A `DenoiseState` processes this many samples at a time.
    pub const FRAME_SIZE: usize = FRAME_SIZE;

    pub(crate) fn default() -> Self {
        DenoiseState::from_model_owned(Cow::Owned(RnnModel::default()), DenoiseParams::default())
    }

    /// Creates a new `DenoiseState`.
    pub fn new() -> Box<DenoiseState<'static>> {
        Box::new(Self::default())
    }

    /// Creates a new `DenoiseState` with custom denoising parameters.
    pub fn with_params(params: DenoiseParams) -> Box<DenoiseState<'static>> {
        Box::new(DenoiseState::from_model_owned(
            Cow::Owned(RnnModel::default()),
            params,
        ))
    }

    /// Creates a new `DenoiseState` owning a custom model.
    ///
    /// The main difference between this method and `DenoiseState::with_model` is that here
    /// `DenoiseState` will own the model; this might be more convenient.
    pub fn from_model(model: RnnModel) -> Box<DenoiseState<'static>> {
        Box::new(DenoiseState::from_model_owned(
            Cow::Owned(model),
            DenoiseParams::default(),
        ))
    }
}

impl<'model> DenoiseState<'model> {
    /// Creates a new `DenoiseState` using a custom model.
    ///
    /// The main difference between this method and `DenoiseState::from_model` is that here
    /// `DenoiseState` will borrow the model; this might create some lifetime-related pain, but
    /// it means that the same model can be shared between multiple `DenoiseState`s.
    pub fn with_model(model: &'model RnnModel) -> Box<DenoiseState<'model>> {
        Box::new(DenoiseState::from_model_owned(
            Cow::Borrowed(model),
            DenoiseParams::default(),
        ))
    }

    /// Creates a new `DenoiseState` using a custom model and custom parameters.
    pub fn with_model_and_params(
        model: &'model RnnModel,
        params: DenoiseParams,
    ) -> Box<DenoiseState<'model>> {
        Box::new(DenoiseState::from_model_owned(Cow::Borrowed(model), params))
    }

    pub(crate) fn from_model_owned(
        model: Cow<'model, RnnModel>,
        params: DenoiseParams,
    ) -> DenoiseState<'model> {
        let look = params.lookahead_value();
        DenoiseState {
            params,
            lastg: [0.0; NB_BANDS],
            gains: [0.0; NB_BANDS],
            rnn: crate::rnn::RnnState::new(model),
            feat: crate::features::DenoiseFeatures::new(),
            vad_scratch: 0.0,
            spec_ring: vec![vec![Complex::default(); FREQ_SIZE]; look],
            gain_ring: vec![[0.0; NB_BANDS]; if look > 0 { look + 1 } else { 0 }],
            vad_ring: vec![0.0; if look > 0 { look + 1 } else { 0 }],
            silence_ring: vec![true; look],
            ring_pos: 0,
        }
    }

    /// The parameters this state was built with.
    pub fn params(&self) -> &DenoiseParams {
        &self.params
    }

    /// How many frames of output lag the input.
    ///
    /// This is one frame for the algorithm's own overlap-add delay, plus any frames requested
    /// by [`DenoiseParams::lookahead`].
    ///
    /// The one-frame floor is intrinsic and cannot be removed: reconstructing input frame `k`
    /// needs the analysis window spanning frames `k` and `k+1`, so no causal implementation
    /// can emit frame `k` before it has been given frame `k+1`. [`denoise_offline`] takes care
    /// of the bookkeeping if you have the whole signal up front.
    pub fn latency_frames(&self) -> usize {
        1 + self.params.lookahead_value()
    }

    /// Forgets all history, as though this state had just been created.
    pub fn reset(&mut self) {
        self.lastg = [0.0; NB_BANDS];
        self.gains = [0.0; NB_BANDS];
        self.rnn.reset();
        self.feat.reset();
        self.vad_scratch = 0.0;
        for s in self.spec_ring.iter_mut() {
            for v in s.iter_mut() {
                *v = Complex::default();
            }
        }
        for g in self.gain_ring.iter_mut() {
            *g = [0.0; NB_BANDS];
        }
        for v in self.vad_ring.iter_mut() {
            *v = 0.0;
        }
        for s in self.silence_ring.iter_mut() {
            *s = true;
        }
        self.ring_pos = 0;
    }

    /// Processes a chunk of samples.
    ///
    /// Both `output` and `input` should be slices of length `DenoiseState::FRAME_SIZE`, and they
    /// are assumed to be in 16-bit, 48kHz signed PCM format. Note that although the input and
    /// output are `f32`s, they are supposed to come from 16-bit integers. In particular, they
    /// should be in the range `[-32768.0, 32767.0]` instead of the range `[-1.0, 1.0]` which
    /// is more common for floating-point PCM.
    ///
    /// The current output of `process_frame` depends on the current input, but also on the
    /// preceding inputs. Because of this, you might prefer to discard the very first output; it
    /// will contain some fade-in artifacts. See [`DenoiseState::latency_frames`], and
    /// [`denoise_offline`] if you have the whole signal and would rather not think about it.
    ///
    /// Returns the probability that the emitted frame contained speech.
    pub fn process_frame(&mut self, output: &mut [f32], input: &[f32]) -> f32 {
        assert_eq!(output.len(), FRAME_SIZE);
        assert_eq!(input.len(), FRAME_SIZE);

        let silence = self.analyze(input);
        self.synthesize(output, silence)
    }

    /// Runs feature extraction and inference for one frame, leaving the resulting gains in
    /// `self.gains`. Returns whether the frame was silent.
    pub(crate) fn analyze(&mut self, input: &[f32]) -> bool {
        self.feat.shift_and_filter_input(input);
        let silence = self
            .feat
            .compute_frame_features_with(self.params.pitch_interval_value());

        if silence {
            // No inference on silent frames, so nothing new to smooth. Optionally let the
            // remembered gains fade instead of holding them until audio returns.
            let decay = self.params.silence_gain_decay_value();
            if decay != 1.0 {
                for g in self.lastg.iter_mut() {
                    *g *= decay;
                }
            }
            self.gains = [0.0; NB_BANDS];
            self.vad_scratch = 0.0;
        } else {
            let mut vad_prob = [0.0];
            self.rnn
                .compute(&mut self.gains[..], &mut vad_prob[..], self.feat.features());
            self.smooth_gains(vad_prob[0]);
            self.vad_scratch = vad_prob[0];
        }
        silence
    }

    /// Applies the smoothing, floor and gating rules from [`DenoiseParams`].
    fn smooth_gains(&mut self, vad: f32) {
        let p = &self.params;
        let decay = p.gain_decay_value();
        let rise = p.gain_rise_value();
        let min_gain = p.min_gain();
        let gated = vad < p.vad_threshold_value();

        for (i, gain) in self.gains.iter_mut().enumerate() {
            let last = self.lastg[i];
            // Limit how fast suppression can engage...
            let mut g = gain.max(decay * last);
            // ...and how fast it can let go again.
            g = g.min(last + rise);
            if gated {
                g = g.min(min_gain);
            }
            // Never attenuate past the configured floor.
            g = g.max(min_gain);
            *gain = g;
            self.lastg[i] = g;
        }
    }

    /// The gains computed by the last call to [`DenoiseState::analyze`].
    pub(crate) fn gains(&self) -> &[f32; NB_BANDS] {
        &self.gains
    }

    /// Overrides the gains that [`DenoiseState::synthesize`] will apply. Used to link the
    /// channels of a multi-channel stream together.
    pub(crate) fn set_gains(&mut self, gains: &[f32; NB_BANDS]) {
        self.gains = *gains;
        self.lastg = *gains;
    }

    /// Applies the gains and produces one frame of output.
    pub(crate) fn synthesize(&mut self, output: &mut [f32], silence: bool) -> f32 {
        if self.params.lookahead_value() == 0 {
            if !silence {
                if self.params.pitch_filter_enabled() {
                    self.feat.pitch_filter(&self.gains);
                }
                let mut gf = [1.0; FREQ_SIZE];
                crate::interp_band_gain(&mut gf[..], &self.gains[..]);
                self.feat.apply_gain(&gf);
            }
            self.feat.frame_synthesis(output);
            return self.vad_scratch;
        }

        self.synthesize_delayed(output, silence)
    }

    /// The lookahead path: hold each spectrum back for a few frames and apply the largest gain
    /// seen over that window, so that a speech onset is not attenuated before we know about it.
    fn synthesize_delayed(&mut self, output: &mut [f32], silence: bool) -> f32 {
        let look = self.params.lookahead_value();

        // The comb filter is tied to the pitch of *this* frame, so it has to run now.
        if !silence && self.params.pitch_filter_enabled() {
            self.feat.pitch_filter(&self.gains);
        }

        // Rotate the current spectrum into the delay line and take out the oldest one.
        self.feat
            .swap_spectrum(&mut self.spec_ring[self.ring_pos % look]);
        let emitted_silence =
            std::mem::replace(&mut self.silence_ring[self.ring_pos % look], silence);

        // Silent frames never had gains computed, so they hold zeros, which are neutral for a
        // maximum and keep them from propagating passthrough into their neighbours.
        self.gain_ring[self.ring_pos % (look + 1)] = self.gains;
        self.vad_ring[self.ring_pos % (look + 1)] = self.vad_scratch;

        let mut combined = [0.0f32; NB_BANDS];
        for g in &self.gain_ring {
            for (c, &v) in combined.iter_mut().zip(g.iter()) {
                *c = c.max(v);
            }
        }

        if !emitted_silence {
            let mut gf = [1.0; FREQ_SIZE];
            crate::interp_band_gain(&mut gf[..], &combined[..]);
            self.feat.apply_gain(&gf);
        }
        self.feat.frame_synthesis(output);

        // The frame we just emitted entered the delay line `look` frames ago.
        let emitted_vad = self.vad_ring[(self.ring_pos + 1) % (look + 1)];
        self.ring_pos = self.ring_pos.wrapping_add(1);
        emitted_vad
    }
}

/// Denoises a complete signal, returning output that lines up with the input sample for
/// sample.
///
/// [`DenoiseState::process_frame`] is a streaming interface, so its output lags its input and
/// the caller is expected to account for that. When the whole signal is already in memory
/// there is no reason to make anyone think about it: this runs the stream out past the end,
/// drops the leading latency, and hands back a buffer the same length as the input. Inputs
/// that are not a whole number of frames are zero-padded internally.
///
/// # Example
///
/// ```rust
/// use nnnoiseless::{denoise_offline, DenoiseParams};
///
/// let noisy: Vec<f32> = (0..12_345).map(|i| (i as f32 * 0.3).sin() * 3000.0).collect();
/// // Two frames of lookahead is a good default when latency does not matter.
/// let clean = denoise_offline(DenoiseParams::default().lookahead(2), &noisy);
/// assert_eq!(clean.len(), noisy.len());
/// ```
pub fn denoise_offline(params: DenoiseParams, input: &[f32]) -> Vec<f32> {
    let mut state = DenoiseState::with_params(params);
    let latency = state.latency_frames();

    let mut frame = [0.0; FRAME_SIZE];
    let mut out = Vec::with_capacity(input.len() + (latency + 1) * FRAME_SIZE);

    let mut chunks = input.chunks_exact(FRAME_SIZE);
    for chunk in chunks.by_ref() {
        state.process_frame(&mut frame, chunk);
        out.extend_from_slice(&frame);
    }

    // Zero-pad a ragged final frame rather than dropping it.
    let tail = chunks.remainder();
    if !tail.is_empty() {
        let mut padded = [0.0; FRAME_SIZE];
        padded[..tail.len()].copy_from_slice(tail);
        state.process_frame(&mut frame, &padded);
        out.extend_from_slice(&frame);
    }

    // Push the delay line out with silence so the real tail emerges.
    let silence = [0.0; FRAME_SIZE];
    for _ in 0..latency {
        state.process_frame(&mut frame, &silence);
        out.extend_from_slice(&frame);
    }

    out.drain(..(latency * FRAME_SIZE).min(out.len()));
    out.resize(input.len(), 0.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    fn speech_like(n: usize) -> Vec<f32> {
        let mut seed = 0x2545f491u32;
        let mut lp = 0.0f32;
        (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let w = ((seed >> 16) as i32 - 32768) as f32 / 32768.0;
                lp = 0.85 * lp + 0.15 * w;
                let t = i as f32 / 48000.0;
                let env = (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin()).powf(1.5);
                let mut s = 0.0;
                for h in 1..=12 {
                    s += (2.0 * std::f32::consts::PI * 150.0 * h as f32 * t).sin()
                        / (h as f32).powf(1.2);
                }
                s * env * 6000.0 + (w * 0.5 + lp * 2.0) * 1500.0
            })
            .collect()
    }

    fn run(params: DenoiseParams, input: &[f32]) -> Vec<f32> {
        let mut st = DenoiseState::with_params(params);
        let mut o = vec![0.0; FRAME_SIZE];
        let mut out = Vec::with_capacity(input.len());
        for f in input.chunks_exact(FRAME_SIZE) {
            st.process_frame(&mut o, f);
            out.extend_from_slice(&o);
        }
        out
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    #[test]
    fn state_is_send_and_sync() {
        assert_send_sync::<DenoiseState<'static>>();
    }

    #[test]
    fn default_params_match_the_plain_constructor() {
        let input = speech_like(48000 / 2);
        let a = run(DenoiseParams::default(), &input);
        let b = {
            let mut st = DenoiseState::new();
            let mut o = vec![0.0; FRAME_SIZE];
            let mut out = Vec::new();
            for f in input.chunks_exact(FRAME_SIZE) {
                st.process_frame(&mut o, f);
                out.extend_from_slice(&o);
            }
            out
        };
        assert_eq!(a, b);
    }

    /// A gain floor must actually stop the denoiser from suppressing all the way down.
    #[test]
    fn attenuation_limit_leaves_a_noise_floor() {
        // Pure noise: the denoiser wants to remove essentially all of it.
        let mut seed = 999u32;
        let input: Vec<f32> = (0..48000)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as i32 - 32768) as f32 / 8.0
            })
            .collect();

        let unlimited = run(DenoiseParams::default(), &input);
        let limited = run(DenoiseParams::default().max_attenuation_db(6.0), &input);

        // Skip the warm-up frame in both.
        let (u, l) = (rms(&unlimited[FRAME_SIZE..]), rms(&limited[FRAME_SIZE..]));
        assert!(l > u, "floor should pass more through: {l} vs {u}");

        let input_rms = rms(&input[FRAME_SIZE..]);
        // 6dB of attenuation is a factor of two; allow generous slack for the band structure.
        assert!(
            l > input_rms * 0.2,
            "6dB floor kept too little: {l} vs input {input_rms}"
        );
    }

    /// Gating on voice activity must silence a noise-only signal.
    #[test]
    fn vad_gating_suppresses_non_speech() {
        let mut seed = 4242u32;
        let input: Vec<f32> = (0..48000)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as i32 - 32768) as f32 / 8.0
            })
            .collect();

        let ungated = run(DenoiseParams::default(), &input);
        let gated = run(DenoiseParams::default().vad_threshold(0.95), &input);
        assert!(
            rms(&gated[FRAME_SIZE..]) < rms(&ungated[FRAME_SIZE..]) + 1e-6,
            "gating should not increase the level"
        );
    }

    /// The offline helper must hand back output that lines up with the input, whatever
    /// latency the configuration introduces.
    #[test]
    fn offline_helper_aligns_output_with_input() {
        let input = speech_like(FRAME_SIZE * 20 + 137);

        for params in [
            DenoiseParams::default(),
            DenoiseParams::default().lookahead(3),
        ] {
            let out = denoise_offline(params, &input);
            assert_eq!(out.len(), input.len(), "length should be preserved");

            // Correlate against the input to confirm there is no residual frame offset. The
            // right alignment has to beat every neighbouring shift.
            let corr = |shift: usize| -> f32 {
                let n = input.len() - FRAME_SIZE * 6;
                input[FRAME_SIZE * 2..(FRAME_SIZE * 2 + n)]
                    .iter()
                    .zip(&out[(FRAME_SIZE * 2 + shift)..])
                    .map(|(a, b)| a * b)
                    .sum()
            };
            let aligned = corr(0);
            for shift in [FRAME_SIZE, FRAME_SIZE * 2] {
                assert!(
                    aligned > corr(shift),
                    "{params:?}: shifted by {shift} correlated better than aligned"
                );
            }
        }
    }

    /// Lookahead delays the output by the requested number of frames and otherwise keeps the
    /// signal intact.
    #[test]
    fn lookahead_delays_output_and_preserves_energy() {
        let input = speech_like(FRAME_SIZE * 40);
        let plain = run(DenoiseParams::default(), &input);
        let look = run(DenoiseParams::default().lookahead(2), &input);

        let st = DenoiseState::with_params(DenoiseParams::default().lookahead(2));
        assert_eq!(st.latency_frames(), 3);

        // The extra two frames of latency come from empty spectra, so they are exactly zero;
        // the third frame is the usual fade-in.
        assert_eq!(rms(&look[..FRAME_SIZE * 2]), 0.0);

        // Line the two up and check they carry comparable energy.
        let a = &plain[FRAME_SIZE..(plain.len() - FRAME_SIZE * 2)];
        let b = &look[(FRAME_SIZE * 3)..];
        let n = a.len().min(b.len());
        let (ra, rb) = (rms(&a[..n]), rms(&b[..n]));
        assert!(
            rb > ra * 0.8 && rb < ra * 1.3,
            "lookahead changed the level too much: {rb} vs {ra}"
        );
        // Taking the maximum gain over the window can only let more through.
        assert!(rb >= ra * 0.95, "lookahead lost energy: {rb} vs {ra}");
    }

    /// Skipping the pitch search must stay close to searching every frame.
    #[test]
    fn pitch_interval_tracks_the_full_search() {
        let input = speech_like(FRAME_SIZE * 40);
        let full = run(DenoiseParams::default(), &input);
        let every_third = run(DenoiseParams::default().pitch_interval(3), &input);

        let (a, b) = (rms(&full[FRAME_SIZE..]), rms(&every_third[FRAME_SIZE..]));
        assert!(
            (a - b).abs() < 0.25 * a,
            "decimated pitch search drifted too far: {b} vs {a}"
        );
    }

    #[test]
    fn reset_restores_a_reproducible_state() {
        let input = speech_like(FRAME_SIZE * 10);
        let mut st = DenoiseState::new();
        let mut o = vec![0.0; FRAME_SIZE];

        let mut first = Vec::new();
        for f in input.chunks_exact(FRAME_SIZE) {
            st.process_frame(&mut o, f);
            first.extend_from_slice(&o);
        }

        st.reset();
        let mut second = Vec::new();
        for f in input.chunks_exact(FRAME_SIZE) {
            st.process_frame(&mut o, f);
            second.extend_from_slice(&o);
        }
        assert_eq!(first, second);
    }

    #[test]
    fn disabling_the_pitch_filter_still_produces_sane_audio() {
        let input = speech_like(FRAME_SIZE * 20);
        let out = run(DenoiseParams::default().pitch_filter(false), &input);
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(rms(&out[FRAME_SIZE..]) > 0.0);
    }
}
