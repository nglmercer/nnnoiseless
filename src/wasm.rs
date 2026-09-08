//! JavaScript/WebAssembly bindings.
//!
//! Build with:
//!
//! ```text
//! wasm-pack build --target web --no-default-features --features wasm
//! ```
//!
//! Two entry points are exposed, matching the two ways audio arrives in a browser:
//!
//! * [`Denoiser`] is a streaming interface for live audio. It buffers internally, so it can be
//!   fed the 128-sample blocks an `AudioWorkletProcessor` receives even though the algorithm
//!   works in 480-sample frames.
//! * [`denoise_buffer`] processes a whole decoded `AudioBuffer` at once, resampling to and from
//!   48kHz as needed.
//!
//! Both take and return samples in the browser's usual `-1.0..=1.0` range and do the
//! conversion to the 16-bit scale the denoiser expects internally.

use wasm_bindgen::prelude::*;

use crate::{denoise_offline, DenoiseParams, DenoiseState, Resampler, FRAME_SIZE};

/// The browser's float sample range is `-1..1`; the denoiser wants the range of an `i16`.
const SCALE: f32 = 32_768.0;

/// Streaming denoiser for live audio.
///
/// Push whatever block size the audio graph hands you; pull back whatever is ready. Output
/// lags input by one 10ms frame plus any lookahead, so the first call or two return fewer
/// samples than they were given.
///
/// ```js
/// const denoiser = new Denoiser();
/// denoiser.setAttenuationLimitDb(12);
/// const out = denoiser.push(inputBlock); // Float32Array, may be shorter than the input
/// ```
#[wasm_bindgen]
pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    /// Input samples not yet forming a complete frame.
    pending: Vec<f32>,
    /// Output waiting to be handed back to JavaScript.
    ready: Vec<f32>,
    frame_in: Vec<f32>,
    frame_out: Vec<f32>,
    vad: f32,
    /// Frames still to be discarded to cover the algorithm's latency.
    warmup: usize,
}

#[wasm_bindgen]
impl Denoiser {
    /// Creates a denoiser for 48kHz mono audio.
    ///
    /// Ask for a 48kHz `AudioContext` (`new AudioContext({ sampleRate: 48000 })`) so that no
    /// resampling is needed in the live path.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Denoiser {
        Denoiser::with_settings(0.0, 0.0, 0)
    }

    /// Creates a denoiser with the tuning knobs set up front.
    ///
    /// `attenuation_limit_db` of `0` means unlimited suppression; `vad_threshold` of `0`
    /// disables gating.
    #[wasm_bindgen(js_name = withSettings)]
    pub fn with_settings(
        attenuation_limit_db: f32,
        vad_threshold: f32,
        lookahead: usize,
    ) -> Denoiser {
        let params = build_params(attenuation_limit_db, vad_threshold, lookahead);
        let state = DenoiseState::with_params(params);
        let warmup = state.latency_frames();
        Denoiser {
            state,
            pending: Vec::with_capacity(FRAME_SIZE * 2),
            ready: Vec::with_capacity(FRAME_SIZE * 4),
            frame_in: vec![0.0; FRAME_SIZE],
            frame_out: vec![0.0; FRAME_SIZE],
            vad: 0.0,
            warmup,
        }
    }

    /// The number of samples in one processing frame (480, i.e. 10ms at 48kHz).
    #[wasm_bindgen(getter, js_name = frameSize)]
    pub fn frame_size(&self) -> usize {
        FRAME_SIZE
    }

    /// Probability that the most recently emitted frame contained speech, in `0..=1`.
    #[wasm_bindgen(getter)]
    pub fn vad(&self) -> f32 {
        self.vad
    }

    /// How many samples of delay this configuration introduces.
    #[wasm_bindgen(getter, js_name = latencySamples)]
    pub fn latency_samples(&self) -> usize {
        self.state.latency_frames() * FRAME_SIZE
    }

    /// The instruction set the kernels were compiled for, as a string.
    #[wasm_bindgen(getter, js_name = activeIsa)]
    pub fn active_isa(&self) -> String {
        crate::active_isa().to_string()
    }

    /// Caps how far any band may be attenuated. `0` removes the cap.
    #[wasm_bindgen(js_name = setAttenuationLimitDb)]
    pub fn set_attenuation_limit_db(&mut self, db: f32) {
        let mut p = *self.state.params();
        p = if db > 0.0 {
            p.max_attenuation_db(db)
        } else {
            p.no_attenuation_limit()
        };
        self.rebuild(p);
    }

    /// Gates frames whose speech probability is below `threshold`. `0` disables gating.
    #[wasm_bindgen(js_name = setVadThreshold)]
    pub fn set_vad_threshold(&mut self, threshold: f32) {
        let p = self.state.params().vad_threshold(threshold);
        self.rebuild(p);
    }

    /// Rebuilds the underlying state, preserving nothing but the parameters.
    ///
    /// Changing parameters mid-stream would otherwise leave the delay line sized for the old
    /// configuration, so this resets instead of trying to migrate. It costs one frame of
    /// history, which is inaudible against a control being moved.
    fn rebuild(&mut self, params: DenoiseParams) {
        self.state = DenoiseState::with_params(params);
        self.warmup = self.state.latency_frames();
        self.pending.clear();
        self.ready.clear();
    }

    /// Feeds samples in and returns whatever output is ready.
    ///
    /// The returned array is usually the same length as the input, but is shorter while the
    /// denoiser is filling its delay line.
    pub fn push(&mut self, input: &[f32]) -> Vec<f32> {
        self.pending.extend_from_slice(input);

        while self.pending.len() >= FRAME_SIZE {
            for (dst, src) in self.frame_in.iter_mut().zip(&self.pending[..FRAME_SIZE]) {
                *dst = src * SCALE;
            }
            self.pending.drain(..FRAME_SIZE);

            self.vad = self
                .state
                .process_frame(&mut self.frame_out, &self.frame_in);

            if self.warmup > 0 {
                self.warmup -= 1;
                continue;
            }
            self.ready.extend(self.frame_out.iter().map(|s| s / SCALE));
        }

        // Hand back at most as much as was asked for, so the caller's block size is preserved
        // once the stream is running.
        let take = self.ready.len().min(input.len());
        self.ready.drain(..take).collect()
    }

    /// Forgets all history.
    pub fn reset(&mut self) {
        self.state.reset();
        self.warmup = self.state.latency_frames();
        self.pending.clear();
        self.ready.clear();
        self.vad = 0.0;
    }
}

impl Default for Denoiser {
    fn default() -> Denoiser {
        Denoiser::new()
    }
}

fn build_params(attenuation_limit_db: f32, vad_threshold: f32, lookahead: usize) -> DenoiseParams {
    let mut p = DenoiseParams::default().lookahead(lookahead);
    if attenuation_limit_db > 0.0 {
        p = p.max_attenuation_db(attenuation_limit_db);
    }
    if vad_threshold > 0.0 {
        p = p.vad_threshold(vad_threshold);
    }
    p
}

/// Denoises a complete buffer, resampling to 48kHz and back if necessary.
///
/// This is the one to use for a decoded `AudioBuffer`: it returns a buffer the same length as
/// the input, at the same sample rate, with the algorithm's latency already compensated for.
///
/// `attenuation_limit_db` of `0` means unlimited suppression; `vad_threshold` of `0` disables
/// gating.
#[wasm_bindgen(js_name = denoiseBuffer)]
pub fn denoise_buffer(
    samples: &[f32],
    sample_rate: f32,
    attenuation_limit_db: f32,
    vad_threshold: f32,
    lookahead: usize,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let params = build_params(attenuation_limit_db, vad_threshold, lookahead);
    let rate = sample_rate as f64;
    let needs_resampling = (rate - 48_000.0).abs() > f64::EPSILON;

    // Up to 48kHz, denoise, and back down.
    let at_48k: Vec<f32> = if needs_resampling {
        let mut r = Resampler::to_denoiser_rate(rate, 1);
        let mut out = Vec::with_capacity(samples.len() * 2);
        let scaled: Vec<f32> = samples.iter().map(|s| s * SCALE).collect();
        r.process(&scaled, &mut out);
        r.flush(&mut out);
        out
    } else {
        samples.iter().map(|s| s * SCALE).collect()
    };

    let denoised = denoise_offline(params, &at_48k);

    let mut result: Vec<f32> = if needs_resampling {
        let mut r = Resampler::new(48_000.0, rate, 1);
        let mut out = Vec::with_capacity(samples.len() + 64);
        r.process(&denoised, &mut out);
        r.flush(&mut out);
        out
    } else {
        denoised
    };

    for s in result.iter_mut() {
        *s /= SCALE;
    }
    // Round-tripping through the resampler can be a sample or two off; give the caller
    // exactly what it asked for.
    result.resize(samples.len(), 0.0);
    result
}

/// The instruction set the kernels were compiled for.
#[wasm_bindgen(js_name = activeIsa)]
pub fn active_isa_js() -> String {
    crate::active_isa().to_string()
}

/// The crate version, so a page can show what it is running.
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(feature = "hush")]
/// Streaming Hush denoiser for normalized 16 kHz mono audio.
///
/// Construct it with [`HushDenoiser::from_model_bytes`]. The model bundle is
/// the `advanced_dfnet16k_model_best_onnx.tar.gz` artifact from Hush and must
/// be supplied by the caller so it does not get silently embedded in every
/// WebAssembly build.
#[wasm_bindgen]
pub struct HushDenoiser {
    state: crate::HushDenoiser,
    pending: Vec<f32>,
    ready: Vec<f32>,
    frame_in: Vec<f32>,
    frame_out: Vec<f32>,
    lsnr_db: f32,
    warmup: usize,
    received_samples: usize,
    input_energy: f32,
}

#[cfg(feature = "hush")]
#[wasm_bindgen]
impl HushDenoiser {
    /// Builds a Hush denoiser from an ONNX model bundle held in memory.
    #[wasm_bindgen(js_name = fromModelBytes)]
    pub fn from_model_bytes(
        model_bytes: &[u8],
        attenuation_limit_db: f32,
    ) -> Result<HushDenoiser, JsValue> {
        let model = crate::HushModel::from_bytes(model_bytes).map_err(js_error)?;
        let attenuation = if attenuation_limit_db > 0.0 {
            attenuation_limit_db
        } else {
            100.0
        };
        let state = model
            .denoiser_with_attenuation_db(attenuation)
            .map_err(js_error)?;
        let frame_size = state.frame_size();
        let warmup = crate::HUSH_SYNTHESIS_DELAY_SAMPLES.div_ceil(frame_size);
        Ok(HushDenoiser {
            state,
            pending: Vec::with_capacity(frame_size * 2),
            ready: Vec::with_capacity(frame_size * 4),
            frame_in: vec![0.0; frame_size],
            frame_out: vec![0.0; frame_size],
            lsnr_db: -15.0,
            warmup,
            received_samples: 0,
            input_energy: 0.0,
        })
    }

    /// Hush's sample rate, 16 kHz.
    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> usize {
        self.state.sample_rate()
    }

    /// Hush's streaming frame size, 160 samples.
    #[wasm_bindgen(getter, js_name = frameSize)]
    pub fn frame_size(&self) -> usize {
        self.state.frame_size()
    }

    /// Hush's documented algorithmic delay in samples.
    ///
    /// The streaming adapter aligns frames using Hush's shorter synthesis
    /// delay; this getter intentionally reports the model-level latency.
    #[wasm_bindgen(getter, js_name = latencySamples)]
    pub fn latency_samples(&self) -> usize {
        self.state.latency_samples()
    }

    /// Local-SNR estimate in decibels from the most recently emitted frame.
    #[wasm_bindgen(getter, js_name = lsnrDb)]
    pub fn lsnr_db(&self) -> f32 {
        self.lsnr_db
    }

    fn process_frame(&mut self) -> Result<(), JsValue> {
        self.lsnr_db = self
            .state
            .process_frame(&mut self.frame_out, &self.frame_in)
            .map_err(js_error)?;
        if self.warmup > 0 {
            self.warmup -= 1;
        } else {
            self.ready.extend_from_slice(&self.frame_out);
        }
        Ok(())
    }

    /// Feeds normalized samples in and returns whatever output is ready.
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.received_samples = self.received_samples.saturating_add(input.len());
        let chunk_energy: f32 = input.iter().map(|sample| sample * sample).sum();
        if chunk_energy.is_finite() {
            self.input_energy = (self.input_energy + chunk_energy).min(f32::MAX / 2.0);
        }
        self.pending.extend_from_slice(input);
        let frame_size = self.frame_size();
        while self.pending.len() >= frame_size {
            self.frame_in.copy_from_slice(&self.pending[..frame_size]);
            self.pending.drain(..frame_size);
            self.process_frame()?;
        }

        let take = self.ready.len().min(input.len());
        Ok(self.ready.drain(..take).collect())
    }

    /// Finishes the current finite stream and returns all remaining output.
    ///
    /// A partial input frame is padded with zeroes, then one synthesis-delay
    /// frame is processed to release the final real audio. Samples belonging
    /// only to the padding are omitted. The denoiser is reset before this
    /// method returns, so the next call to [`Self::push`] starts a new stream.
    /// Calling `finish()` with no input returns an empty array after resetting
    /// the session.
    pub fn finish(&mut self) -> Result<Vec<f32>, JsValue> {
        let frame_size = self.frame_size();
        let pending_len = self.pending.len();
        if self.received_samples == 0 {
            self.reset()?;
            return Ok(Vec::new());
        }

        if pending_len > 0 {
            self.frame_in[..pending_len].copy_from_slice(&self.pending);
            self.frame_in[pending_len..].fill(0.0);
            self.pending.clear();
            self.process_frame()?;
        }

        let flush_level = if self.input_energy >= 1e-7 * self.received_samples as f32 {
            0.0004
        } else {
            0.0
        };
        self.frame_in.fill(flush_level);
        for _ in 0..crate::HUSH_SYNTHESIS_DELAY_SAMPLES.div_ceil(frame_size) {
            self.process_frame()?;
        }

        let final_frame_len = if pending_len > 0 {
            pending_len
        } else {
            frame_size
        };
        let padding_len = frame_size - final_frame_len;
        if padding_len > 0 {
            let ready_len = self.ready.len();
            if ready_len < padding_len {
                return Err(JsValue::from_str(
                    "Hush finish produced fewer samples than the padded frame",
                ));
            }
            self.ready.truncate(ready_len - padding_len);
        }

        let output = self.ready.drain(..).collect();
        self.reset()?;
        Ok(output)
    }

    /// Changes the maximum attenuation and resets the stream's processing history.
    #[wasm_bindgen(js_name = setAttenuationLimitDb)]
    pub fn set_attenuation_limit_db(&mut self, db: f32) -> Result<(), JsValue> {
        let attenuation = if db > 0.0 { db } else { 100.0 };
        self.state
            .set_attenuation_limit_db(attenuation)
            .map_err(js_error)?;
        self.reset()
    }

    /// Forgets all Hush recurrent, spectral, and overlap-add history.
    pub fn reset(&mut self) -> Result<(), JsValue> {
        self.state.reset().map_err(js_error)?;
        self.pending.clear();
        self.ready.clear();
        self.lsnr_db = -15.0;
        self.warmup = crate::HUSH_SYNTHESIS_DELAY_SAMPLES.div_ceil(self.frame_size());
        self.received_samples = 0;
        self.input_energy = 0.0;
        Ok(())
    }
}

#[cfg(feature = "hush")]
/// Denoises a complete normalized audio buffer with Hush.
///
/// The model always runs at 16 kHz. Other input rates are resampled to 16 kHz
/// and the result is resampled back to the original rate. The returned buffer
/// has the same length as `samples`.
#[wasm_bindgen(js_name = denoiseHushBuffer)]
pub fn denoise_hush_buffer(
    samples: &[f32],
    sample_rate: f32,
    attenuation_limit_db: f32,
    model_bytes: &[u8],
) -> Result<Vec<f32>, JsValue> {
    crate::denoise_hush_buffer(samples, sample_rate, attenuation_limit_db, model_bytes)
        .map_err(js_error)
}

#[cfg(feature = "hush")]
fn js_error(error: crate::HushError) -> JsValue {
    JsValue::from_str(&error.to_string())
}
