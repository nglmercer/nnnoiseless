//! Structures for computing audio features.
//!
//! This module contains utilities for computing features of an audio signal. These features are
//! fed into the trained neural net for noise removal and speech detection.

use std::sync::Arc;

use realfft::{ComplexToReal, RealToComplex};

use crate::{
    common, Complex, CEPS_MEM, FRAME_SIZE, FREQ_SIZE, NB_BANDS, NB_DELTA_CEPS, NB_FEATURES,
    PITCH_BUF_SIZE, WINDOW_SIZE,
};

/// How much room past the required history we keep, in frames.
///
/// Every frame needs the last `PITCH_BUF_SIZE` samples as one contiguous slice. Shifting the
/// whole buffer down by a frame each time meant moving 1248 floats per frame. Instead we
/// append into slack space and compact only once it runs out, which amortizes the move to
/// `PITCH_BUF_SIZE / HISTORY_SLACK_FRAMES` floats per frame.
const HISTORY_SLACK_FRAMES: usize = 8;
const HISTORY_CAP: usize = PITCH_BUF_SIZE + HISTORY_SLACK_FRAMES * FRAME_SIZE;

/// Contains the necessary state to compute the features of audio input and synthesize the output.
///
/// This is quite a large struct and should probably be kept behind some kind of pointer.
#[derive(Clone)]
pub struct DenoiseFeatures {
    /// Recent input samples. The live window is always `history[end - PITCH_BUF_SIZE..end]`.
    history: Vec<f32>,
    /// One past the newest sample in `history`.
    end: usize,
    /// This is some sort of ring buffer, storing the last bunch of cepstra.
    cepstral_mem: [[f32; NB_BANDS]; CEPS_MEM],
    /// Cached squared distances between every pair of entries in `cepstral_mem`.
    ///
    /// Only one row of `cepstral_mem` changes per frame, so recomputing the whole matrix each
    /// time (64 x 22 operations) was wasted work; updating one row and column costs 8 x 22.
    ceps_dist: [[f32; CEPS_MEM]; CEPS_MEM],
    /// The index pointing to the most recent cepstrum in `cepstral_mem`. The previous cepstra are
    /// at indices mem_id - 1, mem_id - 2, etc (wrapped appropriately).
    mem_id: usize,
    mem_hp_x: [f32; 2],
    synthesis_mem: [f32; FRAME_SIZE],
    window_buf: [f32; WINDOW_SIZE],

    /// The Fourier transform of the most recent frame of input.
    x: Vec<Complex>,
    /// The Fourier transform of a pitch-period-shifted window of input.
    p: Vec<Complex>,
    /// The band energies of `x` (the signal).
    ex: [f32; NB_BANDS],
    /// The band energies of `p` (the signal, lagged by one pitch period).
    ep: [f32; NB_BANDS],
    /// The band correlations between `x` (the signal) and `p` (the pitch-period-lagged signal).
    exp: [f32; NB_BANDS],
    /// The computed features.
    features: [f32; NB_FEATURES],

    /// Number of frames processed, used to decide when to redo the pitch search.
    frame_count: u64,

    fft_scratch: Vec<Complex>,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,

    pitch_finder: crate::pitch::PitchFinder,
}

impl DenoiseFeatures {
    /// Creates a new, empty, `DenoiseFeatures`.
    pub fn new() -> DenoiseFeatures {
        let c = common();
        let fwd = Arc::clone(&c.fft_fwd);
        let inv = Arc::clone(&c.fft_inv);
        let scratch_len = fwd.get_scratch_len().max(inv.get_scratch_len());

        DenoiseFeatures {
            history: vec![0.0; HISTORY_CAP],
            end: PITCH_BUF_SIZE,
            cepstral_mem: [[0.0; NB_BANDS]; CEPS_MEM],
            ceps_dist: [[0.0; CEPS_MEM]; CEPS_MEM],
            mem_id: 0,
            mem_hp_x: [0.0; 2],
            synthesis_mem: [0.0; FRAME_SIZE],
            window_buf: [0.0; WINDOW_SIZE],
            x: vec![Complex::default(); FREQ_SIZE],
            p: vec![Complex::default(); FREQ_SIZE],
            ex: [0.0; NB_BANDS],
            ep: [0.0; NB_BANDS],
            exp: [0.0; NB_BANDS],
            features: [0.0; NB_FEATURES],
            frame_count: 0,
            fft_scratch: vec![Complex::default(); scratch_len],
            fwd,
            inv,
            pitch_finder: crate::pitch::PitchFinder::new(),
        }
    }

    /// Returns the computed features.
    pub fn features(&self) -> &[f32] {
        &self.features[..]
    }

    /// The per-band energies of the current input frame.
    pub fn band_energies(&self) -> &[f32] {
        &self.ex[..]
    }

    /// The per-band energies of the input lagged by one pitch period.
    pub fn pitch_band_energies(&self) -> &[f32] {
        &self.ep[..]
    }

    /// The per-band correlation between the input and its pitch-lagged copy.
    ///
    /// This is the quantity the training scripts compare against the ideal gain.
    pub fn band_correlations(&self) -> &[f32] {
        &self.exp[..]
    }

    /// Forgets all history, as though no audio had been processed.
    pub fn reset(&mut self) {
        for x in self.history.iter_mut() {
            *x = 0.0;
        }
        self.end = PITCH_BUF_SIZE;
        self.cepstral_mem = [[0.0; NB_BANDS]; CEPS_MEM];
        self.ceps_dist = [[0.0; CEPS_MEM]; CEPS_MEM];
        self.mem_id = 0;
        self.mem_hp_x = [0.0; 2];
        self.synthesis_mem = [0.0; FRAME_SIZE];
        self.features = [0.0; NB_FEATURES];
        self.frame_count = 0;
        self.pitch_finder = crate::pitch::PitchFinder::new();
    }

    /// Makes room for one more frame and returns the range it should be written to.
    fn advance(&mut self) -> std::ops::Range<usize> {
        if self.end + FRAME_SIZE > HISTORY_CAP {
            self.history
                .copy_within((self.end - PITCH_BUF_SIZE)..self.end, 0);
            self.end = PITCH_BUF_SIZE;
        }
        let range = self.end..(self.end + FRAME_SIZE);
        self.end += FRAME_SIZE;
        range
    }

    /// The last `PITCH_BUF_SIZE` input samples, oldest first.
    ///
    /// The hot paths inline this so that they can borrow `history` and `window_buf`
    /// separately, so it only survives as a test helper.
    #[cfg(test)]
    fn history(&self) -> &[f32] {
        &self.history[(self.end - PITCH_BUF_SIZE)..self.end]
    }

    /// Shifts our input buffer and adds the new input to it. This is mainly used when generating
    /// training data: when running the noise reduction we use [`DenoiseFeatures::shift_and_filter_input`]
    /// instead.
    pub fn shift_input(&mut self, input: &[f32]) {
        assert!(input.len() == FRAME_SIZE);
        let range = self.advance();
        self.history[range].copy_from_slice(input);
    }

    /// Shifts our input buffer and adds the new input to it, while running the input through a
    /// high-pass filter.
    pub fn shift_and_filter_input(&mut self, input: &[f32]) {
        assert!(input.len() == FRAME_SIZE);
        let range = self.advance();
        crate::util::BIQUAD_HP.filter(&mut self.history[range], &mut self.mem_hp_x, input);
    }

    fn find_pitch(&mut self, interval: usize) -> usize {
        // The pitch period moves slowly compared to the 10ms frame rate, so the search can be
        // run less often. This is off by default because it does change the output.
        if interval > 1 && !self.frame_count.is_multiple_of(interval as u64) {
            let (period, _) = self.pitch_finder.last();
            if period != 0 {
                return period;
            }
        }
        // Borrowing `history` and `pitch_finder` as separate fields keeps this allocation-free.
        let start = self.end - PITCH_BUF_SIZE;
        let (pitch, _gain) = self.pitch_finder.process(&self.history[start..self.end]);
        pitch
    }

    /// Computes the features of the current frame.
    ///
    /// The return value is `true` if the input was pretty much silent.
    pub fn compute_frame_features(&mut self) -> bool {
        self.compute_frame_features_with(1)
    }

    /// As [`DenoiseFeatures::compute_frame_features`], but only redoing the pitch search every
    /// `pitch_interval` frames.
    pub(crate) fn compute_frame_features_with(&mut self, pitch_interval: usize) -> bool {
        let mut ly = [0.0; NB_BANDS];
        let mut tmp = [0.0; NB_BANDS];

        {
            let hist_start = self.end - PITCH_BUF_SIZE;
            let (history, window_buf) = (&self.history[hist_start..self.end], &mut self.window_buf);
            transform_input(
                history,
                0,
                window_buf,
                &mut self.x,
                &mut self.ex,
                &self.fwd,
                &mut self.fft_scratch,
            );
        }
        let pitch_idx = self.find_pitch(pitch_interval);
        self.frame_count = self.frame_count.wrapping_add(1);

        {
            let hist_start = self.end - PITCH_BUF_SIZE;
            let (history, window_buf) = (&self.history[hist_start..self.end], &mut self.window_buf);
            transform_input(
                history,
                pitch_idx,
                window_buf,
                &mut self.p,
                &mut self.ep,
                &self.fwd,
                &mut self.fft_scratch,
            );
        }
        crate::compute_band_corr(&mut self.exp[..], &self.x, &self.p);
        for i in 0..NB_BANDS {
            self.exp[i] /= (0.001 + self.ex[i] * self.ep[i]).sqrt();
        }
        crate::dct(&mut tmp[..], &self.exp[..]);
        for (i, &value) in tmp.iter().take(NB_DELTA_CEPS).enumerate() {
            self.features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = value;
        }

        self.features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3;
        self.features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9;
        self.features[NB_BANDS + 3 * NB_DELTA_CEPS] = 0.01 * (pitch_idx as f32 - 300.0);
        let mut log_max = -2.0;
        let mut follow = -2.0;
        let mut e = 0.0;
        for (i, value) in ly.iter_mut().enumerate() {
            *value = (1e-2 + self.ex[i])
                .log10()
                .max(log_max - 7.0)
                .max(follow - 1.5);
            log_max = log_max.max(*value);
            follow = (follow - 1.5).max(*value);
            e += self.ex[i];
        }

        if e < 0.04 {
            /* If there's no audio, avoid messing up the state. */
            for i in 0..NB_FEATURES {
                self.features[i] = 0.0;
            }
            return true;
        }
        crate::dct(&mut self.features, &ly[..]);
        self.features[0] -= 12.0;
        self.features[1] -= 4.0;
        let ceps_0_idx = self.mem_id;
        let ceps_1_idx = if self.mem_id < 1 {
            CEPS_MEM + self.mem_id - 1
        } else {
            self.mem_id - 1
        };
        let ceps_2_idx = if self.mem_id < 2 {
            CEPS_MEM + self.mem_id - 2
        } else {
            self.mem_id - 2
        };

        for i in 0..NB_BANDS {
            self.cepstral_mem[ceps_0_idx][i] = self.features[i];
        }
        // Only the row we just wrote can have changed, so refresh that row and column of the
        // cached distance matrix rather than all of it.
        for j in 0..CEPS_MEM {
            let mut dist = 0.0;
            for k in 0..NB_BANDS {
                let tmp = self.cepstral_mem[ceps_0_idx][k] - self.cepstral_mem[j][k];
                dist += tmp * tmp;
            }
            self.ceps_dist[ceps_0_idx][j] = dist;
            self.ceps_dist[j][ceps_0_idx] = dist;
        }
        self.mem_id += 1;

        let ceps_0 = &self.cepstral_mem[ceps_0_idx];
        let ceps_1 = &self.cepstral_mem[ceps_1_idx];
        let ceps_2 = &self.cepstral_mem[ceps_2_idx];
        for i in 0..NB_DELTA_CEPS {
            self.features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
            self.features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
            self.features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2.0 * ceps_1[i] + ceps_2[i];
        }

        /* Spectral variability features. */
        let mut spec_variability = 0.0;
        if self.mem_id == CEPS_MEM {
            self.mem_id = 0;
        }
        for i in 0..CEPS_MEM {
            let mut min_dist = 1e15f32;
            for j in 0..CEPS_MEM {
                if j != i {
                    min_dist = min_dist.min(self.ceps_dist[i][j]);
                }
            }
            spec_variability += min_dist;
        }

        self.features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM as f32 - 2.1;

        false
    }

    /// Applies a filter to the audio, attenuating pitches that have poor correlation with the
    /// pitch-lagged signal.
    pub fn pitch_filter(&mut self, gain: &[f32; NB_BANDS]) {
        let mut r = [0.0; NB_BANDS];
        let mut rf = [0.0; FREQ_SIZE];
        for i in 0..NB_BANDS {
            r[i] = if self.exp[i] > gain[i] {
                1.0
            } else {
                let exp_sq = self.exp[i] * self.exp[i];
                let g_sq = gain[i] * gain[i];
                exp_sq * (1.0 - g_sq) / (0.001 + g_sq * (1.0 - exp_sq))
            };
            r[i] = r[i].clamp(0.0, 1.0).sqrt();
            r[i] *= (self.ex[i] / (1e-8 + self.ep[i])).sqrt();
        }
        crate::interp_band_gain(&mut rf[..], &r[..]);
        for ((x, p), &rf) in self.x.iter_mut().zip(&self.p).zip(rf.iter()) {
            *x += p * rf;
        }

        let mut new_e = [0.0; NB_BANDS];
        crate::compute_band_corr(&mut new_e[..], &self.x, &self.x);
        for i in 0..NB_BANDS {
            r[i] = (self.ex[i] / (1e-8 + new_e[i])).sqrt();
        }
        crate::interp_band_gain(&mut rf[..], &r[..]);
        for (x, &rf) in self.x.iter_mut().zip(rf.iter()) {
            *x *= rf;
        }
    }

    pub(crate) fn apply_gain(&mut self, gain: &[f32; FREQ_SIZE]) {
        for (x, &g) in self.x.iter_mut().zip(gain.iter()) {
            *x *= g;
        }
    }

    /// Hands out the current spectrum so that a caller can hold on to it while it computes
    /// gains from later frames. Used by the lookahead path.
    pub(crate) fn swap_spectrum(&mut self, other: &mut Vec<Complex>) {
        std::mem::swap(&mut self.x, other);
    }

    pub(crate) fn frame_synthesis(&mut self, out: &mut [f32]) {
        assert_eq!(out.len(), FRAME_SIZE);

        // The inverse transform requires a purely real DC and Nyquist bin. Everything we do to
        // the spectrum scales it by real gains, so this only guards against rounding.
        self.x[0].im = 0.0;
        let last = self.x.len() - 1;
        self.x[last].im = 0.0;

        self.inv
            .process_with_scratch(&mut self.x, &mut self.window_buf, &mut self.fft_scratch)
            .expect("inverse FFT buffers are sized at construction");

        // The synthesis window carries the inverse transform's normalization, so no separate
        // scaling pass is needed here.
        crate::apply_synthesis_window_in_place(&mut self.window_buf[..]);
        for (i, sample) in out.iter_mut().enumerate() {
            *sample = self.window_buf[i] + self.synthesis_mem[i];
            self.synthesis_mem[i] = self.window_buf[FRAME_SIZE + i];
        }
    }
}

impl Default for DenoiseFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// Fourier transforms the input.
///
/// The Fourier transform goes in `x` and the band energies go in `ex`.
fn transform_input(
    input: &[f32],
    lag: usize,
    window_buf: &mut [f32; WINDOW_SIZE],
    x: &mut [Complex],
    ex: &mut [f32],
    fwd: &Arc<dyn RealToComplex<f32>>,
    scratch: &mut [Complex],
) {
    let input = &input[input.len().checked_sub(WINDOW_SIZE + lag).unwrap()..];
    crate::apply_window(&mut window_buf[..], input);
    fwd.process_with_scratch(&mut window_buf[..], x, scratch)
        .expect("forward FFT buffers are sized at construction");

    // In the original RNNoise code, the forward transform is normalized and the inverse
    // tranform isn't. `realfft` doesn't normalize either one, so we do it ourselves.
    let norm = common().wnorm();
    for v in x.iter_mut() {
        *v *= norm;
    }

    crate::compute_band_corr(ex, x, x);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(offset: f32) -> Vec<f32> {
        (0..FRAME_SIZE)
            .map(|i| (i as f32 * 0.05 + offset).sin() * 5000.0)
            .collect()
    }

    /// The amortized history buffer must present the same window as a naive shift-down would.
    #[test]
    fn history_window_matches_a_naive_shift() {
        let mut feat = DenoiseFeatures::new();
        let mut naive = vec![0.0f32; PITCH_BUF_SIZE];

        // Run past the point where compaction has to happen at least twice.
        for k in 0..(HISTORY_SLACK_FRAMES * 3 + 1) {
            let frame = ramp(k as f32);
            feat.shift_input(&frame);

            naive.copy_within(FRAME_SIZE.., 0);
            naive[(PITCH_BUF_SIZE - FRAME_SIZE)..].copy_from_slice(&frame);

            assert_eq!(feat.history(), &naive[..], "mismatch after frame {k}");
        }
    }

    /// A round trip with unit gains must reconstruct the input, which is what proves the
    /// normalization folded into the synthesis window is right.
    #[test]
    fn unit_gain_round_trip_reconstructs_the_input() {
        let mut feat = DenoiseFeatures::new();
        let gain = [1.0; FREQ_SIZE];
        let mut out = vec![0.0; FRAME_SIZE];

        let frames: Vec<Vec<f32>> = (0..4).map(|k| ramp(k as f32 * 7.0)).collect();
        for (k, frame) in frames.iter().enumerate() {
            feat.shift_input(frame);
            feat.compute_frame_features();
            feat.apply_gain(&gain);
            feat.frame_synthesis(&mut out);

            // Output lags the input by exactly one frame.
            if k >= 1 {
                let expected = &frames[k - 1];
                for (i, (&got, &want)) in out.iter().zip(expected).enumerate() {
                    assert!(
                        (got - want).abs() < 1.0,
                        "frame {k} sample {i}: {got} vs {want}"
                    );
                }
            }
        }
    }

    /// The cached distance matrix has to agree with recomputing it from scratch.
    #[test]
    fn cached_cepstral_distances_match_a_full_recomputation() {
        let mut feat = DenoiseFeatures::new();
        for k in 0..20 {
            feat.shift_input(&ramp(k as f32 * 3.0));
            feat.compute_frame_features();

            for i in 0..CEPS_MEM {
                for j in 0..CEPS_MEM {
                    let mut want = 0.0;
                    for b in 0..NB_BANDS {
                        let d = feat.cepstral_mem[i][b] - feat.cepstral_mem[j][b];
                        want += d * d;
                    }
                    let got = feat.ceps_dist[i][j];
                    assert!(
                        (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                        "frame {k}, dist[{i}][{j}]: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn reset_returns_to_the_initial_state() {
        let mut feat = DenoiseFeatures::new();
        for k in 0..12 {
            feat.shift_and_filter_input(&ramp(k as f32));
            feat.compute_frame_features();
        }
        feat.reset();

        let fresh = DenoiseFeatures::new();
        assert_eq!(feat.history(), fresh.history());
        assert_eq!(feat.features(), fresh.features());
        assert_eq!(feat.mem_id, fresh.mem_id);
    }
}
