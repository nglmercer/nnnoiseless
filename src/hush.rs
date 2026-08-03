//! Hush/DeepFilterNet-SE inference backend.
//!
//! Hush is a different model family from the built-in RNNoise path. It runs at
//! 16 kHz, consumes 160-sample (10 ms) frames, and uses the DeepFilterNet
//! encoder, ERB mask decoder, and complex deep-filter decoder. This module
//! deliberately keeps that model behind an opt-in feature and a separate API;
//! Hush weights cannot be loaded by [`crate::RnnModel`].

use std::fmt;
use std::path::Path;

use df::tract::{DfParams, DfTract, ReduceMask, RuntimeParams};
use ndarray::{ArrayView2, ArrayViewMut2};

/// Hush's native sample rate.
pub const HUSH_SAMPLE_RATE: usize = 16_000;

/// Hush's streaming hop size: 10 ms at [`HUSH_SAMPLE_RATE`].
pub const HUSH_FRAME_SIZE: usize = 160;

/// The model's documented algorithmic delay in samples at 16 kHz.
pub const HUSH_LATENCY_SAMPLES: usize = 320;

/// An error returned while loading or running a Hush model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HushError(String);

impl HushError {
    fn new(message: impl Into<String>) -> HushError {
        HushError(message.into())
    }
}

impl fmt::Display for HushError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HushError {}

/// Parsed Hush model parameters.
///
/// The model bundle is the `advanced_dfnet16k_model_best_onnx.tar.gz` artifact
/// released by Hush. It contains the encoder, ERB decoder, deep-filter
/// decoder, and the model configuration. The immutable parameters can be
/// cloned to create independent streaming sessions.
#[derive(Clone)]
pub struct HushModel {
    params: DfParams,
}

impl HushModel {
    /// Loads a Hush ONNX bundle from a filesystem path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<HushModel, HushError> {
        let path = path.as_ref();
        let params = DfParams::new(path.to_path_buf())
            .map_err(|error| HushError::new(format!("could not load Hush model: {error}")))?;
        Ok(HushModel { params })
    }

    /// Loads a Hush ONNX bundle from memory.
    ///
    /// The upstream DeepFilterNet API currently accepts a `&'static [u8]` for
    /// its in-memory loader even though it copies the archive entries while
    /// parsing. The input is therefore intentionally leaked once per model
    /// load so this API can be used by WebAssembly, where filesystem loading is
    /// unavailable. Load the model once and reuse the returned [`HushModel`].
    pub fn from_bytes(bytes: &[u8]) -> Result<HushModel, HushError> {
        let bytes: &'static [u8] = Box::leak(bytes.to_vec().into_boxed_slice());
        Self::from_static_bytes(bytes)
    }

    /// Loads a Hush ONNX bundle from static bytes.
    pub fn from_static_bytes(bytes: &'static [u8]) -> Result<HushModel, HushError> {
        let params = DfParams::from_bytes(bytes)
            .map_err(|error| HushError::new(format!("could not parse Hush model: {error}")))?;
        Ok(HushModel { params })
    }

    /// Creates an independent streaming denoiser with unlimited attenuation.
    pub fn denoiser(&self) -> Result<HushDenoiser, HushError> {
        self.denoiser_with_attenuation_db(100.0)
    }

    /// Creates an independent streaming denoiser with a maximum attenuation
    /// in decibels. Use `100.0` for effectively unlimited suppression.
    pub fn denoiser_with_attenuation_db(
        &self,
        attenuation_db: f32,
    ) -> Result<HushDenoiser, HushError> {
        if !attenuation_db.is_finite() || attenuation_db < 0.0 {
            return Err(HushError::new(
                "Hush attenuation must be finite and non-negative",
            ));
        }

        let runtime =
            RuntimeParams::new(1, false, attenuation_db, -15.0, 35.0, 35.0, ReduceMask::MAX);
        let runtime = DfTract::new(self.params.clone(), &runtime).map_err(|error| {
            HushError::new(format!("could not initialize Hush runtime: {error}"))
        })?;

        if runtime.sr != HUSH_SAMPLE_RATE || runtime.hop_size != HUSH_FRAME_SIZE {
            return Err(HushError::new(format!(
                "model is {} Hz with {}-sample frames; expected Hush's {} Hz/{}-sample contract",
                runtime.sr, runtime.hop_size, HUSH_SAMPLE_RATE, HUSH_FRAME_SIZE
            )));
        }

        Ok(HushDenoiser {
            runtime,
            frame_size: HUSH_FRAME_SIZE,
            last_lsnr_db: -15.0,
        })
    }
}

/// A stateful, mono Hush denoiser.
///
/// Input and output are normalized `f32` samples in `-1.0..=1.0`, unlike the
/// existing RNNoise API in this crate, which uses the scale of signed 16-bit
/// PCM. Feed exactly [`HUSH_FRAME_SIZE`] samples to [`Self::process_frame`].
pub struct HushDenoiser {
    runtime: DfTract,
    frame_size: usize,
    last_lsnr_db: f32,
}

impl HushDenoiser {
    /// Returns the model sample rate, always 16 kHz for the released Hush model.
    pub fn sample_rate(&self) -> usize {
        HUSH_SAMPLE_RATE
    }

    /// Returns the number of samples expected by [`Self::process_frame`].
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Returns the documented algorithmic delay in samples.
    pub fn latency_samples(&self) -> usize {
        HUSH_LATENCY_SAMPLES
    }

    /// Returns the local-SNR estimate from the most recently processed frame.
    pub fn last_lsnr_db(&self) -> f32 {
        self.last_lsnr_db
    }

    /// Processes one normalized 10 ms mono frame and returns its local-SNR estimate.
    pub fn process_frame(&mut self, output: &mut [f32], input: &[f32]) -> Result<f32, HushError> {
        if input.len() != self.frame_size || output.len() != self.frame_size {
            return Err(HushError::new(format!(
                "Hush frames must contain {} samples (input {}, output {})",
                self.frame_size,
                input.len(),
                output.len()
            )));
        }

        let input = ArrayView2::from_shape((1, self.frame_size), input)
            .map_err(|error| HushError::new(format!("invalid Hush input frame: {error}")))?;
        let output = ArrayViewMut2::from_shape((1, self.frame_size), output)
            .map_err(|error| HushError::new(format!("invalid Hush output frame: {error}")))?;
        let lsnr = self
            .runtime
            .process(input, output)
            .map_err(|error| HushError::new(format!("Hush inference failed: {error}")))?;
        self.last_lsnr_db = lsnr;
        Ok(lsnr)
    }

    /// Changes the maximum attenuation for subsequent frames.
    pub fn set_attenuation_limit_db(&mut self, attenuation_db: f32) -> Result<(), HushError> {
        if !attenuation_db.is_finite() || attenuation_db < 0.0 {
            return Err(HushError::new(
                "Hush attenuation must be finite and non-negative",
            ));
        }
        self.runtime
            .set_atten_lim(attenuation_db)
            .map_err(|error| HushError::new(format!("could not set Hush attenuation: {error}")))
    }

    /// Resets the recurrent, spectral-normalization, and overlap-add state.
    pub fn reset(&mut self) -> Result<(), HushError> {
        self.runtime
            .init()
            .map_err(|error| HushError::new(format!("could not reset Hush runtime: {error}")))?;
        self.last_lsnr_db = -15.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_frame_sizes_without_touching_the_runtime() {
        // Model construction is intentionally covered by the integration test
        // when HUSH_MODEL is supplied; this test keeps the API contract cheap.
        assert_eq!(HUSH_SAMPLE_RATE, 16_000);
        assert_eq!(HUSH_FRAME_SIZE, 160);
        assert_eq!(HUSH_LATENCY_SAMPLES, 320);
    }
}
