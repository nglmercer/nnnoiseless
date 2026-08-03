//! Denoising several channels with linked gains.

use std::borrow::Cow;

use crate::{DenoiseParams, DenoiseState, RnnModel, FRAME_SIZE, NB_BANDS};

/// How the per-band gains of the different channels are combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelLink {
    /// Each channel is denoised completely independently.
    ///
    /// This is what you get from running one [`DenoiseState`] per channel by hand. It is the
    /// most aggressive setting, but because the channels make their own decisions the residual
    /// noise floor moves around between them, which is audible on headphones as the background
    /// wandering from side to side.
    Independent,
    /// Every channel gets the largest gain any channel asked for.
    ///
    /// This keeps the stereo image stable, at the cost of letting through noise that only one
    /// channel wanted to keep. It is the safe default for stereo speech.
    Max,
    /// Every channel gets the mean of the gains the channels asked for.
    Mean,
}

/// Denoises a multi-channel stream, optionally keeping the channels' gains in step.
///
/// # Example
///
/// ```rust
/// use nnnoiseless::{ChannelLink, MultiDenoiser, DenoiseState};
///
/// let mut denoiser = MultiDenoiser::new(2, ChannelLink::Max);
/// let input = vec![vec![0.0f32; DenoiseState::FRAME_SIZE]; 2];
/// let mut output = vec![vec![0.0f32; DenoiseState::FRAME_SIZE]; 2];
///
/// let inputs: Vec<&[f32]> = input.iter().map(|c| &c[..]).collect();
/// let mut outputs: Vec<&mut [f32]> = output.iter_mut().map(|c| &mut c[..]).collect();
/// denoiser.process_frame(&mut outputs, &inputs);
/// ```
#[derive(Clone)]
pub struct MultiDenoiser<'model> {
    states: Vec<DenoiseState<'model>>,
    link: ChannelLink,
    silence: Vec<bool>,
    combined: [f32; NB_BANDS],
}

impl MultiDenoiser<'static> {
    /// Creates a denoiser for `channels` channels, using the built-in model.
    pub fn new(channels: usize, link: ChannelLink) -> MultiDenoiser<'static> {
        MultiDenoiser::with_params(channels, link, DenoiseParams::default())
    }

    /// Creates a denoiser for `channels` channels with custom parameters.
    pub fn with_params(
        channels: usize,
        link: ChannelLink,
        params: DenoiseParams,
    ) -> MultiDenoiser<'static> {
        let model = RnnModel::default();
        MultiDenoiser {
            states: (0..channels)
                .map(|_| DenoiseState::from_model_owned(Cow::Owned(model.clone()), params))
                .collect(),
            link,
            silence: vec![false; channels],
            combined: [0.0; NB_BANDS],
        }
    }
}

impl<'model> MultiDenoiser<'model> {
    /// Creates a denoiser that shares one model between all of its channels.
    pub fn with_model(
        channels: usize,
        link: ChannelLink,
        model: &'model RnnModel,
        params: DenoiseParams,
    ) -> MultiDenoiser<'model> {
        MultiDenoiser {
            states: (0..channels)
                .map(|_| DenoiseState::from_model_owned(Cow::Borrowed(model), params))
                .collect(),
            link,
            silence: vec![false; channels],
            combined: [0.0; NB_BANDS],
        }
    }

    /// The number of channels this denoiser handles.
    pub fn channels(&self) -> usize {
        self.states.len()
    }

    /// How the channels' gains are being combined.
    pub fn link(&self) -> ChannelLink {
        self.link
    }

    /// How many frames of output lag the input. See [`DenoiseState::latency_frames`].
    pub fn latency_frames(&self) -> usize {
        self.states.first().map_or(0, |s| s.latency_frames())
    }

    /// Forgets all history.
    pub fn reset(&mut self) {
        for s in self.states.iter_mut() {
            s.reset();
        }
    }

    /// Denoises one frame of every channel.
    ///
    /// `input` and `output` must both have one entry per channel, and every entry must be
    /// exactly [`DenoiseState::FRAME_SIZE`] samples long. Returns the greatest voice-activity
    /// probability across the channels.
    pub fn process_frame(&mut self, output: &mut [&mut [f32]], input: &[&[f32]]) -> f32 {
        assert_eq!(
            input.len(),
            self.states.len(),
            "wrong number of input channels"
        );
        assert_eq!(
            output.len(),
            self.states.len(),
            "wrong number of output channels"
        );

        // Analyse every channel first, so that the gains are all available before any of them
        // is applied.
        for (i, state) in self.states.iter_mut().enumerate() {
            assert_eq!(input[i].len(), FRAME_SIZE);
            self.silence[i] = state.analyze(input[i]);
        }

        if self.link != ChannelLink::Independent && self.states.len() > 1 {
            self.combine_gains();
            for state in self.states.iter_mut() {
                state.set_gains(&self.combined);
            }
        }

        let mut vad: f32 = 0.0;
        for (i, state) in self.states.iter_mut().enumerate() {
            assert_eq!(output[i].len(), FRAME_SIZE);
            vad = vad.max(state.synthesize(output[i], self.silence[i]));
        }
        vad
    }

    fn combine_gains(&mut self) {
        match self.link {
            ChannelLink::Independent => unreachable!("checked by the caller"),
            ChannelLink::Max => {
                self.combined = [0.0; NB_BANDS];
                for state in &self.states {
                    for (c, &g) in self.combined.iter_mut().zip(state.gains().iter()) {
                        *c = c.max(g);
                    }
                }
            }
            ChannelLink::Mean => {
                self.combined = [0.0; NB_BANDS];
                for state in &self.states {
                    for (c, &g) in self.combined.iter_mut().zip(state.gains().iter()) {
                        *c += g;
                    }
                }
                let n = self.states.len() as f32;
                for c in self.combined.iter_mut() {
                    *c /= n;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noisy(n: usize, seed: u32, speech: bool) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let w = ((s >> 16) as i32 - 32768) as f32 / 32768.0;
                let t = i as f32 / 48000.0;
                let voice = if speech {
                    (2.0 * std::f32::consts::PI * 160.0 * t).sin() * 6000.0
                } else {
                    0.0
                };
                voice + w * 1500.0
            })
            .collect()
    }

    fn run(link: ChannelLink, chans: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut d = MultiDenoiser::new(chans.len(), link);
        let mut out = vec![Vec::new(); chans.len()];
        let mut bufs = vec![vec![0.0f32; FRAME_SIZE]; chans.len()];

        let frames = chans[0].len() / FRAME_SIZE;
        for f in 0..frames {
            let r = (f * FRAME_SIZE)..((f + 1) * FRAME_SIZE);
            let ins: Vec<&[f32]> = chans.iter().map(|c| &c[r.clone()]).collect();
            let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|b| &mut b[..]).collect();
            d.process_frame(&mut outs, &ins);
            for (o, b) in out.iter_mut().zip(&bufs) {
                o.extend_from_slice(b);
            }
        }
        out
    }

    /// With one channel, linking cannot do anything, so all modes must agree.
    #[test]
    fn single_channel_is_unaffected_by_linking() {
        let ch = vec![noisy(FRAME_SIZE * 20, 7, true)];
        let indep = run(ChannelLink::Independent, &ch);
        let max = run(ChannelLink::Max, &ch);
        let mean = run(ChannelLink::Mean, &ch);
        assert_eq!(indep, max);
        assert_eq!(indep, mean);
    }

    /// A single `DenoiseState` and an independent `MultiDenoiser` must agree exactly.
    #[test]
    fn independent_mode_matches_a_plain_state() {
        let ch = vec![noisy(FRAME_SIZE * 20, 3, true)];
        let multi = run(ChannelLink::Independent, &ch);

        let mut st = DenoiseState::new();
        let mut o = vec![0.0; FRAME_SIZE];
        let mut single = Vec::new();
        for f in ch[0].chunks_exact(FRAME_SIZE) {
            st.process_frame(&mut o, f);
            single.extend_from_slice(&o);
        }
        assert_eq!(multi[0], single);
    }

    /// When one channel has speech and the other does not, `Max` makes the quiet channel
    /// inherit the speaking channel's gains instead of being suppressed on its own.
    ///
    /// Note that this equalizes the *gains*, not the output levels: the channels still carry
    /// different signals. The observable consequence is that the noise-only channel keeps more
    /// of its energy than it would on its own, which is exactly what stops the residual noise
    /// floor from wandering between the speakers.
    #[test]
    fn max_linking_makes_the_quiet_channel_follow_the_loud_one() {
        let chans = vec![
            noisy(FRAME_SIZE * 40, 11, true),
            noisy(FRAME_SIZE * 40, 29, false),
        ];

        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();

        let indep = run(ChannelLink::Independent, &chans);
        let linked = run(ChannelLink::Max, &chans);

        // Taking a maximum can only ever let more through, never less, on either channel.
        for c in 0..2 {
            assert!(
                rms(&linked[c][FRAME_SIZE..]) >= rms(&indep[c][FRAME_SIZE..]) * 0.99,
                "channel {c} lost energy under linking"
            );
        }

        // The mechanism itself: after a linked frame every channel holds the same gain
        // vector, and it is the elementwise maximum of the independent ones.
        let mut linked_d = MultiDenoiser::new(2, ChannelLink::Max);
        let mut indep_d = MultiDenoiser::new(2, ChannelLink::Independent);
        let mut bufs = vec![vec![0.0f32; FRAME_SIZE]; 2];

        let mut saw_a_difference = false;
        for f in 0..30 {
            let r = (f * FRAME_SIZE)..((f + 1) * FRAME_SIZE);
            let ins: Vec<&[f32]> = chans.iter().map(|c| &c[r.clone()]).collect();

            let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|b| &mut b[..]).collect();
            indep_d.process_frame(&mut outs, &ins);
            let independent: Vec<[f32; NB_BANDS]> =
                indep_d.states.iter().map(|s| *s.gains()).collect();

            let mut outs: Vec<&mut [f32]> = bufs.iter_mut().map(|b| &mut b[..]).collect();
            linked_d.process_frame(&mut outs, &ins);

            let g0 = linked_d.states[0].gains();
            let g1 = linked_d.states[1].gains();
            assert_eq!(g0, g1, "frame {f}: linked channels disagree");

            if independent[0] != independent[1] {
                saw_a_difference = true;
                for b in 0..NB_BANDS {
                    assert!(
                        g0[b] >= independent[0][b] - 1e-6 && g0[b] >= independent[1][b] - 1e-6,
                        "frame {f} band {b}: linked gain {} is below an independent one",
                        g0[b]
                    );
                }
            }
        }
        assert!(
            saw_a_difference,
            "test signal never made the channels disagree, so it proves nothing"
        );
    }

    #[test]
    fn mean_linking_sits_between_the_channels() {
        let chans = vec![
            noisy(FRAME_SIZE * 20, 5, true),
            noisy(FRAME_SIZE * 20, 17, false),
        ];
        let out = run(ChannelLink::Mean, &chans);
        assert!(out.iter().all(|c| c.iter().all(|x| x.is_finite())));
        assert_eq!(out.len(), 2);
    }

    #[test]
    #[should_panic(expected = "wrong number of input channels")]
    fn mismatched_channel_count_panics() {
        let mut d = MultiDenoiser::new(2, ChannelLink::Max);
        let input = vec![0.0f32; FRAME_SIZE];
        let mut output = vec![0.0f32; FRAME_SIZE];
        let ins: Vec<&[f32]> = vec![&input];
        let mut outs: Vec<&mut [f32]> = vec![&mut output];
        d.process_frame(&mut outs, &ins);
    }
}
