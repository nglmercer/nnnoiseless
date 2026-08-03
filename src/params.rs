//! Tunable knobs for the denoiser.

/// Controls how the per-band gains produced by the neural network are turned into the gains
/// that actually get applied.
///
/// Every default here reproduces the original RNNoise behaviour exactly, so
/// `DenoiseParams::default()` changes nothing. The knobs exist because the fixed constants
/// buried in the signal path are the ones people most often need to trade off: how much noise
/// is allowed through, how quickly suppression engages, and how aggressively non-speech is
/// gated.
///
/// # Example
///
/// Leaving a 12dB noise floor instead of suppressing all the way down is the usual remedy for
/// the "musical noise" and pumping that full suppression can produce:
///
/// ```rust
/// use nnnoiseless::{DenoiseParams, DenoiseState};
///
/// let params = DenoiseParams::default().max_attenuation_db(12.0);
/// let mut state = DenoiseState::with_params(params);
/// # let _ = &mut state;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseParams {
    gain_decay: f32,
    gain_rise: f32,
    min_gain: f32,
    vad_threshold: f32,
    silence_gain_decay: f32,
    pitch_filter: bool,
    pitch_interval: usize,
    lookahead: usize,
}

impl Default for DenoiseParams {
    fn default() -> DenoiseParams {
        DenoiseParams {
            gain_decay: 0.6,
            // Gains live in `0..=1`, so a limit of 1.0 is the same as no limit at all.
            gain_rise: 1.0,
            min_gain: 0.0,
            vad_threshold: 0.0,
            silence_gain_decay: 1.0,
            pitch_filter: true,
            pitch_interval: 1,
            lookahead: 0,
        }
    }
}

impl DenoiseParams {
    /// The default parameters, which reproduce the original RNNoise signal path.
    pub fn new() -> DenoiseParams {
        DenoiseParams::default()
    }

    /// Limits how much the denoiser is allowed to attenuate a band, in decibels.
    ///
    /// With no limit (the default) a band can be suppressed to silence. That maximizes noise
    /// removal but is also what produces "musical noise": isolated bands switching between
    /// full and zero gain between frames. Capping the attenuation at, say, 12dB leaves a low
    /// noise floor which masks those artifacts and usually sounds considerably more natural.
    ///
    /// `db` is interpreted as a magnitude, so both `12.0` and `-12.0` mean 12dB of
    /// attenuation. Values are clamped to be non-negative.
    pub fn max_attenuation_db(mut self, db: f32) -> DenoiseParams {
        let db = db.abs();
        self.min_gain = if db.is_finite() {
            10f32.powf(-db / 20.0)
        } else {
            0.0
        };
        self
    }

    /// Removes any limit on attenuation, so that bands may be suppressed to silence.
    pub fn no_attenuation_limit(mut self) -> DenoiseParams {
        self.min_gain = 0.0;
        self
    }

    /// Lower bound on how fast a band's gain may fall, as a fraction of its previous value.
    ///
    /// The default `0.6` means a gain can drop to at most 60% of last frame's value in one
    /// 10ms frame, which stops suppression from slamming shut on a transient. `0.0` removes
    /// the limit; values near `1.0` make suppression engage very slowly.
    pub fn gain_decay(mut self, decay: f32) -> DenoiseParams {
        self.gain_decay = decay.clamp(0.0, 1.0);
        self
    }

    /// Upper bound on how much a band's gain may rise in a single frame, additively.
    ///
    /// The default `1.0` spans the whole gain range and so imposes no limit, which is what
    /// the original algorithm does: suppression is allowed to disengage instantly. Smaller
    /// values ease back to passthrough over several frames, which softens the click that an
    /// instant release can produce at the start of a word.
    pub fn gain_rise(mut self, rise: f32) -> DenoiseParams {
        self.gain_rise = rise.clamp(0.0, 1.0);
        self
    }

    /// Gates output when the voice-activity probability falls below `threshold`.
    ///
    /// Frames below the threshold are attenuated to the floor set by
    /// [`DenoiseParams::max_attenuation_db`] (silence, if no floor is set). The default `0.0`
    /// disables gating. Values above about `0.9` will clip the quiet edges of words.
    pub fn vad_threshold(mut self, threshold: f32) -> DenoiseParams {
        self.vad_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// How much to decay the remembered gains on each frame the input is silent.
    ///
    /// When the input drops below the silence threshold the denoiser skips inference entirely
    /// and keeps its previous gains. Those stale gains are then reused when audio returns,
    /// which can produce a click after a long pause. The default `1.0` keeps the original
    /// behaviour; a value like `0.9` fades the memory out instead.
    pub fn silence_gain_decay(mut self, decay: f32) -> DenoiseParams {
        self.silence_gain_decay = decay.clamp(0.0, 1.0);
        self
    }

    /// Enables or disables the comb filter that reinforces the detected pitch. On by default.
    pub fn pitch_filter(mut self, enabled: bool) -> DenoiseParams {
        self.pitch_filter = enabled;
        self
    }

    /// Runs the pitch search only every `interval` frames, reusing the last result in between.
    ///
    /// The pitch search is roughly a third of the total cost, and the pitch period changes
    /// slowly compared to the 10ms frame rate, so an interval of 2 or 3 buys real speed. This
    /// is the one knob here that trades quality for speed, so it defaults to `1` (search every
    /// frame). Values below 1 are treated as 1.
    pub fn pitch_interval(mut self, interval: usize) -> DenoiseParams {
        self.pitch_interval = interval.max(1);
        self
    }

    /// Computes gains this many frames ahead, and applies the largest gain seen in that window.
    ///
    /// The denoiser is otherwise strictly causal, so it only finds out that a word started
    /// after the onset has already been attenuated. Looking ahead a frame or two lets the
    /// onset through intact. The cost is `lookahead` extra frames of latency, so this is meant
    /// for offline processing; it defaults to `0`.
    ///
    /// See [`crate::DenoiseState::latency_frames`] for the resulting delay.
    pub fn lookahead(mut self, frames: usize) -> DenoiseParams {
        self.lookahead = frames;
        self
    }

    pub(crate) fn gain_decay_value(&self) -> f32 {
        self.gain_decay
    }
    pub(crate) fn gain_rise_value(&self) -> f32 {
        self.gain_rise
    }
    pub(crate) fn min_gain(&self) -> f32 {
        self.min_gain
    }
    pub(crate) fn vad_threshold_value(&self) -> f32 {
        self.vad_threshold
    }
    pub(crate) fn silence_gain_decay_value(&self) -> f32 {
        self.silence_gain_decay
    }
    pub(crate) fn pitch_filter_enabled(&self) -> bool {
        self.pitch_filter
    }
    pub(crate) fn pitch_interval_value(&self) -> usize {
        self.pitch_interval
    }
    pub(crate) fn lookahead_value(&self) -> usize {
        self.lookahead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_original_behaviour() {
        let p = DenoiseParams::default();
        assert_eq!(p.gain_decay_value(), 0.6);
        // A rise limit of 1.0 spans the whole gain range, so it never binds.
        assert_eq!(p.gain_rise_value(), 1.0);
        assert_eq!(p.min_gain(), 0.0);
        assert_eq!(p.vad_threshold_value(), 0.0);
        assert_eq!(p.silence_gain_decay_value(), 1.0);
        assert!(p.pitch_filter_enabled());
        assert_eq!(p.pitch_interval_value(), 1);
        assert_eq!(p.lookahead_value(), 0);
    }

    #[test]
    fn attenuation_limit_converts_decibels_to_a_linear_gain() {
        let p = DenoiseParams::default().max_attenuation_db(20.0);
        assert!((p.min_gain() - 0.1).abs() < 1e-6, "{}", p.min_gain());

        let p = DenoiseParams::default().max_attenuation_db(6.0206);
        assert!((p.min_gain() - 0.5).abs() < 1e-4, "{}", p.min_gain());

        // The sign is not meaningful: attenuation is always downward.
        assert_eq!(
            DenoiseParams::default()
                .max_attenuation_db(-12.0)
                .min_gain(),
            DenoiseParams::default().max_attenuation_db(12.0).min_gain()
        );

        assert_eq!(
            DenoiseParams::default()
                .max_attenuation_db(f32::INFINITY)
                .min_gain(),
            0.0
        );
        assert_eq!(
            DenoiseParams::default()
                .max_attenuation_db(20.0)
                .no_attenuation_limit()
                .min_gain(),
            0.0
        );
    }

    #[test]
    fn out_of_range_values_are_clamped() {
        assert_eq!(
            DenoiseParams::default().gain_decay(-1.0).gain_decay_value(),
            0.0
        );
        assert_eq!(
            DenoiseParams::default().gain_decay(5.0).gain_decay_value(),
            1.0
        );
        assert_eq!(
            DenoiseParams::default().gain_rise(-3.0).gain_rise_value(),
            0.0
        );
        assert_eq!(
            DenoiseParams::default()
                .vad_threshold(7.0)
                .vad_threshold_value(),
            1.0
        );
        assert_eq!(
            DenoiseParams::default()
                .pitch_interval(0)
                .pitch_interval_value(),
            1
        );
    }
}
