use std::borrow::Cow;

use crate::util::{relu, sigmoid_approx, tansig_approx, zip3};

/// Sanity cap on layer sizes, so that a corrupt model file cannot ask us to allocate
/// something absurd. It is not a limit on what the algorithm supports.
const MAX_NEURONS: usize = 4096;

/// The scale the training scripts use when quantizing weights to `i8`.
const WEIGHTS_SCALE: f32 = 1.0 / 256.0;

// It's annoying to expose a public API with `i8`s, because `include_bytes` works with `u8`s only.
// So we do conversions from `&[i8]` to `&[u8]` internally. Hopefully at some point rust will have
// a safe API for this...
fn to_i8(x: &[u8]) -> &[i8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const i8, x.len()) }
}

/// The pointwise activation used by a neural-network layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// Hyperbolic tangent.
    Tanh = 0,
    /// Logistic sigmoid.
    Sigmoid = 1,
    /// Rectified linear unit.
    Relu = 2,
}

impl Activation {
    fn from_code(x: i32) -> Option<Activation> {
        match x {
            0 => Some(Activation::Tanh),
            1 => Some(Activation::Sigmoid),
            2 => Some(Activation::Relu),
            _ => None,
        }
    }

    #[inline(always)]
    fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => sigmoid_approx(x),
            Activation::Tanh => tansig_approx(x),
            Activation::Relu => relu(x),
        }
    }
}

/// Model weights, stored in whichever form the build asked for.
///
/// By default the `i8` weights from the model file are widened to `f32` once, at load time,
/// and the quantization scale is folded in. That removes a per-multiply-accumulate widening
/// from the inner loop, which measured ~1.8x on the inference stage without SIMD. The
/// `low-memory` feature keeps the original `i8` data instead, at ~4x less memory.
#[derive(Clone)]
struct Weights {
    #[cfg(not(feature = "low-memory"))]
    data: Vec<f32>,
    #[cfg(feature = "low-memory")]
    data: Cow<'static, [i8]>,
}

impl Weights {
    /// The scale that still has to be applied after accumulation. It is `1.0` when the scale
    /// was already folded into the stored weights.
    #[cfg(not(feature = "low-memory"))]
    const POST_SCALE: f32 = 1.0;
    #[cfg(feature = "low-memory")]
    const POST_SCALE: f32 = WEIGHTS_SCALE;

    #[cfg(not(feature = "low-memory"))]
    fn new(src: Cow<'static, [i8]>) -> Weights {
        Weights {
            data: src.iter().map(|&x| x as f32 * WEIGHTS_SCALE).collect(),
        }
    }

    #[cfg(feature = "low-memory")]
    fn new(src: Cow<'static, [i8]>) -> Weights {
        Weights { data: src }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.data.len()
    }

    /// `out[j] += sum_i data[i * stride + offset + j] * input[i]`
    #[inline]
    fn matvec(&self, stride: usize, offset: usize, out: &mut [f32], input: &[f32]) {
        let k = &crate::common().kernels;
        #[cfg(not(feature = "low-memory"))]
        (k.matvec)(&self.data, stride, offset, out, input);
        #[cfg(feature = "low-memory")]
        (k.matvec_i8)(&self.data, stride, offset, out, input);
    }

    /// Copies `out.len()` values starting at `start` into `out`.
    #[inline]
    fn load(&self, out: &mut [f32], start: usize) {
        let src = &self.data[start..(start + out.len())];
        #[cfg(not(feature = "low-memory"))]
        out.copy_from_slice(src);
        #[cfg(feature = "low-memory")]
        for (o, &s) in out.iter_mut().zip(src) {
            *o = s as f32;
        }
    }

    /// Re-quantizes back to the on-disk representation.
    fn to_i8(&self) -> Vec<i8> {
        #[cfg(not(feature = "low-memory"))]
        {
            self.data
                .iter()
                .map(|&x| (x / WEIGHTS_SCALE).round().clamp(-128.0, 127.0) as i8)
                .collect()
        }
        #[cfg(feature = "low-memory")]
        {
            self.data.to_vec()
        }
    }
}

/// A fully connected neural-network layer loaded from the compact model format.
#[derive(Clone)]
pub struct DenseLayer {
    bias: Weights,
    input_weights: Weights,
    nb_inputs: usize,
    nb_neurons: usize,
    activation: Activation,
}

/// A gated recurrent unit layer loaded from the compact model format.
#[derive(Clone)]
pub struct GruLayer {
    bias: Weights,
    input_weights: Weights,
    recurrent_weights: Weights,
    nb_inputs: usize,
    nb_neurons: usize,
    activation: Activation,
}

impl DenseLayer {
    /// Number of input values this layer consumes.
    pub fn nb_inputs(&self) -> usize {
        self.nb_inputs
    }
    /// Number of output neurons.
    pub fn nb_neurons(&self) -> usize {
        self.nb_neurons
    }
    /// The activation applied to the output.
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

impl GruLayer {
    /// Number of input values this layer consumes.
    pub fn nb_inputs(&self) -> usize {
        self.nb_inputs
    }
    /// Number of recurrent neurons.
    pub fn nb_neurons(&self) -> usize {
        self.nb_neurons
    }
    /// The activation applied to the candidate state.
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

/// An `RnnModel` contains all the model parameters for the denoising algorithm.
/// `nnnoiseless` has a built-in model that should work for most purposes, but if you have
/// specific needs then you might benefit from training a custom model. Scripts for model
/// training are available as part of [`RNNoise`]; once the model is trained, you can load it
/// here.
///
/// Two on-disk formats are understood, and [`RnnModel::from_bytes`] detects which one it was
/// handed:
///
/// * **v1**, the original RNNoise layout, which stores each layer dimension in a single
///   signed byte and so cannot describe a layer wider than 127 neurons;
/// * **v2**, which is the same weight data behind a short header with 32-bit dimensions, and
///   therefore has no practical width limit. Use [`RnnModel::to_bytes`] to convert.
///
/// [`RNNoise`]: https://github.com/xiph/rnnoise
#[derive(Clone)]
pub struct RnnModel {
    pub(crate) input_dense: DenseLayer,
    pub(crate) vad_gru: GruLayer,
    pub(crate) noise_gru: GruLayer,
    pub(crate) denoise_gru: GruLayer,
    pub(crate) denoise_output: DenseLayer,
    pub(crate) vad_output: DenseLayer,
}

/// Magic bytes introducing the v2 model format.
const MAGIC_V2: &[u8; 4] = b"NNNM";
const FORMAT_VERSION: u16 = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    Dense,
    Gru,
}

/// A cursor over the model file that reads whichever dimension encoding the format uses.
struct Reader<'a> {
    bytes: &'a [i8],
    pos: usize,
    v2: bool,
}

impl<'a> Reader<'a> {
    fn dim(&mut self) -> Option<usize> {
        if self.v2 {
            let b = self.bytes.get(self.pos..(self.pos + 4))?;
            self.pos += 4;
            let v = u32::from_le_bytes([b[0] as u8, b[1] as u8, b[2] as u8, b[3] as u8]) as usize;
            if v == 0 || v > MAX_NEURONS {
                return None;
            }
            Some(v)
        } else {
            let b = *self.bytes.get(self.pos)?;
            self.pos += 1;
            if b > 0 {
                Some(b as usize)
            } else {
                None
            }
        }
    }

    fn byte(&mut self) -> Option<i8> {
        let b = *self.bytes.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    fn array(&mut self, len: usize, moo: fn(&'a [i8]) -> Cow<'static, [i8]>) -> Option<Weights> {
        let slice = self.bytes.get(self.pos..(self.pos + len))?;
        self.pos += len;
        Some(Weights::new(moo(slice)))
    }

    fn kind(&mut self, expected: LayerKind) -> Option<()> {
        if !self.v2 {
            return Some(());
        }
        let k = self.byte()?;
        let got = match k {
            0 => LayerKind::Dense,
            1 => LayerKind::Gru,
            _ => return None,
        };
        (got == expected).then_some(())
    }
}

impl RnnModel {
    /// Reads an `RnnModel` from an array of bytes, in either supported format.
    pub fn from_bytes(bytes: &[u8]) -> Option<RnnModel> {
        RnnModel::from_bytes_impl(bytes, |xs| Cow::Owned(xs.to_owned()))
    }

    /// Reads an `RnnModel` from a static array of bytes.
    ///
    /// This differs from [`RnnModel::from_bytes`] in that the returned model can borrow the
    /// provided `bytes` array instead of copying it. Note that borrowing only actually
    /// happens under the `low-memory` feature: by default the weights are widened to `f32` at
    /// load time, which needs its own storage regardless.
    ///
    /// ```ignore
    /// let weight_data: &'static [u8] = include_bytes!("/path/to/model/weights.rnn");
    /// let model = RnnModel::from_static_bytes(weight_data).expect("Corrupted model file");
    /// ```
    pub fn from_static_bytes(bytes: &'static [u8]) -> Option<RnnModel> {
        RnnModel::from_bytes_impl(bytes, Cow::Borrowed)
    }

    /// Serializes this model in the v2 format, which can describe layers of any width.
    ///
    /// Weights are stored quantized, exactly as the training scripts emit them, so a
    /// v1 -> v2 -> v1 round trip is lossless.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC_V2);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&6u16.to_le_bytes());

        fn push_dense(out: &mut Vec<u8>, l: &DenseLayer) {
            out.push(0);
            out.push(l.activation as u8);
            out.extend_from_slice(&(l.nb_inputs as u32).to_le_bytes());
            out.extend_from_slice(&(l.nb_neurons as u32).to_le_bytes());
            out.extend(l.input_weights.to_i8().iter().map(|&x| x as u8));
            out.extend(l.bias.to_i8().iter().map(|&x| x as u8));
        }
        fn push_gru(out: &mut Vec<u8>, l: &GruLayer) {
            out.push(1);
            out.push(l.activation as u8);
            out.extend_from_slice(&(l.nb_inputs as u32).to_le_bytes());
            out.extend_from_slice(&(l.nb_neurons as u32).to_le_bytes());
            out.extend(l.input_weights.to_i8().iter().map(|&x| x as u8));
            out.extend(l.recurrent_weights.to_i8().iter().map(|&x| x as u8));
            out.extend(l.bias.to_i8().iter().map(|&x| x as u8));
        }

        push_dense(&mut out, &self.input_dense);
        push_gru(&mut out, &self.vad_gru);
        push_gru(&mut out, &self.noise_gru);
        push_gru(&mut out, &self.denoise_gru);
        push_dense(&mut out, &self.denoise_output);
        push_dense(&mut out, &self.vad_output);
        out
    }

    /// Reads an `RnnModel` from an array of bytes.
    ///
    /// The v1 format is simple: each NN layer is represented by an array of signed `i8`s,
    /// and these layers are simply concatenated.
    ///
    /// The format for a dense layer is
    /// <nb_inputs> <nb_neurons> <activation>
    /// <weights...>
    /// <bias...>
    /// where each of the <?> terms represents a single integer, and each of the <?...> terms
    /// represents an array of integers of the appropriate length (`weights` has length
    /// `nb_neurons * nb_inputs` and `bias` has length `nb_neurons`).
    ///
    /// The format for a GRU layer is
    /// <nb_inputs> <nb_neurons> <activation>
    /// <input_weights...>
    /// <recurrent_weights...>
    /// <bias...>
    /// where `input_weights` and `recurrent_weights` have length `3 * nb_inputs * nb_neurons` each,
    /// and `bias` has length `3 * nb_neurons`.
    ///
    /// The v2 format prefixes the whole file with `"NNNM"`, a `u16` version and a `u16` layer
    /// count, and gives each layer a one-byte kind tag, a one-byte activation and two
    /// little-endian `u32` dimensions before the same weight arrays.
    fn from_bytes_impl<'a>(
        bytes: &'a [u8],
        moo: fn(&'a [i8]) -> Cow<'static, [i8]>,
    ) -> Option<RnnModel> {
        let v2 = bytes.len() >= 8 && &bytes[..4] == MAGIC_V2;
        let mut r = if v2 {
            if u16::from_le_bytes([bytes[4], bytes[5]]) != FORMAT_VERSION
                || u16::from_le_bytes([bytes[6], bytes[7]]) != 6
            {
                return None;
            }
            Reader {
                bytes: to_i8(bytes),
                pos: 8,
                v2: true,
            }
        } else {
            Reader {
                bytes: to_i8(bytes),
                pos: 0,
                v2: false,
            }
        };

        fn read_dense<'a>(
            r: &mut Reader<'a>,
            moo: fn(&'a [i8]) -> Cow<'static, [i8]>,
        ) -> Option<DenseLayer> {
            r.kind(LayerKind::Dense)?;
            // v1 orders the header as inputs, neurons, activation; v2 puts the activation
            // right after the kind tag so that the dimensions stay 4-byte aligned.
            let (nb_inputs, nb_neurons, activation) = if r.v2 {
                let a = Activation::from_code(r.byte()? as i32)?;
                (r.dim()?, r.dim()?, a)
            } else {
                let i = r.dim()?;
                let n = r.dim()?;
                (i, n, Activation::from_code(r.byte()? as i32)?)
            };
            let input_weights = r.array(nb_neurons.checked_mul(nb_inputs)?, moo)?;
            let bias = r.array(nb_neurons, moo)?;
            Some(DenseLayer {
                nb_inputs,
                nb_neurons,
                input_weights,
                bias,
                activation,
            })
        }

        fn read_gru<'a>(
            r: &mut Reader<'a>,
            moo: fn(&'a [i8]) -> Cow<'static, [i8]>,
        ) -> Option<GruLayer> {
            r.kind(LayerKind::Gru)?;
            let (nb_inputs, nb_neurons, activation) = if r.v2 {
                let a = Activation::from_code(r.byte()? as i32)?;
                (r.dim()?, r.dim()?, a)
            } else {
                let i = r.dim()?;
                let n = r.dim()?;
                (i, n, Activation::from_code(r.byte()? as i32)?)
            };
            let input_weights =
                r.array(3usize.checked_mul(nb_neurons)?.checked_mul(nb_inputs)?, moo)?;
            let recurrent_weights = r.array(
                3usize.checked_mul(nb_neurons)?.checked_mul(nb_neurons)?,
                moo,
            )?;
            let bias = r.array(3 * nb_neurons, moo)?;
            Some(GruLayer {
                nb_inputs,
                nb_neurons,
                input_weights,
                recurrent_weights,
                bias,
                activation,
            })
        }

        let input_dense = read_dense(&mut r, moo)?;
        let vad_gru = read_gru(&mut r, moo)?;
        let noise_gru = read_gru(&mut r, moo)?;
        let denoise_gru = read_gru(&mut r, moo)?;
        let denoise_output = read_dense(&mut r, moo)?;
        let vad_output = read_dense(&mut r, moo)?;

        if r.pos != r.bytes.len() {
            return None;
        }

        // The input to the first layer must match the number of features we compute, the
        // denoise output must produce one gain per band, and the vad output is a single
        // probability. Everything else only has to be internally consistent, so that wider
        // models than the built-in one can be loaded.
        if input_dense.nb_inputs != crate::NB_FEATURES
            || denoise_output.nb_neurons != crate::NB_BANDS
            || vad_output.nb_neurons != 1
        {
            return None;
        }
        if input_dense.nb_neurons != vad_gru.nb_inputs || vad_gru.nb_neurons != vad_output.nb_inputs
        {
            return None;
        }
        if crate::NB_FEATURES + input_dense.nb_neurons + vad_gru.nb_neurons != noise_gru.nb_inputs {
            return None;
        }
        if crate::NB_FEATURES + vad_gru.nb_neurons + noise_gru.nb_neurons != denoise_gru.nb_inputs {
            return None;
        }
        if denoise_gru.nb_neurons != denoise_output.nb_inputs {
            return None;
        }

        Some(RnnModel {
            input_dense,
            vad_gru,
            noise_gru,
            denoise_gru,
            denoise_output,
            vad_output,
        })
    }

    /// The widest layer in this model, which sets the size of the inference scratch buffers.
    fn max_neurons(&self) -> usize {
        [
            self.input_dense.nb_neurons,
            self.vad_gru.nb_neurons,
            self.noise_gru.nb_neurons,
            self.denoise_gru.nb_neurons,
            self.denoise_output.nb_neurons,
        ]
        .into_iter()
        .max()
        .unwrap()
    }
}

impl Default for RnnModel {
    fn default() -> RnnModel {
        let bytes: &'static [u8] = include_bytes!("weights.rnn");
        RnnModel::from_static_bytes(bytes).unwrap()
    }
}

impl DenseLayer {
    fn compute(&self, output: &mut [f32], input: &[f32]) {
        debug_assert_eq!(output.len(), self.nb_neurons);
        debug_assert_eq!(input.len(), self.nb_inputs);
        self.bias.load(output, 0);
        self.input_weights.matvec(self.nb_neurons, 0, output, input);

        let scale = Weights::POST_SCALE;
        let act = self.activation;
        for out in output.iter_mut() {
            *out = act.apply(*out * scale);
        }
    }
}

/// Per-`RnnState` scratch for GRU evaluation. Sized from the model, so that model width is
/// not capped by a fixed-size stack buffer.
#[derive(Clone)]
struct GruScratch {
    z: Vec<f32>,
    r: Vec<f32>,
    h: Vec<f32>,
}

impl GruScratch {
    fn new(n: usize) -> GruScratch {
        GruScratch {
            z: vec![0.0; n],
            r: vec![0.0; n],
            h: vec![0.0; n],
        }
    }
}

impl GruLayer {
    fn compute(&self, state: &mut [f32], input: &[f32], scratch: &mut GruScratch) {
        let n = self.nb_neurons;
        let stride = 3 * n;
        let scale = Weights::POST_SCALE;
        debug_assert_eq!(state.len(), n);
        debug_assert_eq!(input.len(), self.nb_inputs);

        let z = &mut scratch.z[0..n];
        let r = &mut scratch.r[0..n];
        let h = &mut scratch.h[0..n];

        // Compute update gate.
        self.bias.load(z, 0);
        self.input_weights.matvec(stride, 0, z, input);
        self.recurrent_weights.matvec(stride, 0, z, state);
        for z in z.iter_mut() {
            *z = sigmoid_approx(scale * *z);
        }

        // Compute reset gate.
        self.bias.load(r, n);
        self.input_weights.matvec(stride, n, r, input);
        self.recurrent_weights.matvec(stride, n, r, state);
        for (out, &s) in r.iter_mut().zip(&state[..]) {
            *out = s * sigmoid_approx(scale * *out);
        }

        // Compute output.
        self.bias.load(h, 2 * n);
        self.input_weights.matvec(stride, 2 * n, h, input);
        self.recurrent_weights.matvec(stride, 2 * n, h, r);

        let act = self.activation;
        for (s, &z, &h) in zip3(state, &z[..], &h[..]) {
            let h = act.apply(scale * h);
            *s = z * *s + (1.0 - z) * h;
        }
    }
}

/// The recurrent state of one denoising stream.
#[derive(Clone)]
pub(crate) struct RnnState<'model> {
    model: Cow<'model, RnnModel>,
    vad_gru_state: Vec<f32>,
    noise_gru_state: Vec<f32>,
    denoise_gru_state: Vec<f32>,
    scratch: GruScratch,
    dense_out: Vec<f32>,
    buf: Vec<f32>,
    denoise_buf: Vec<f32>,
}

impl<'model> RnnState<'model> {
    pub(crate) fn new(model: Cow<'model, RnnModel>) -> RnnState<'model> {
        let vad_gru_state = vec![0.0f32; model.vad_gru.nb_neurons];
        let noise_gru_state = vec![0.0f32; model.noise_gru.nb_neurons];
        let denoise_gru_state = vec![0.0f32; model.denoise_gru.nb_neurons];
        let scratch = GruScratch::new(model.max_neurons());
        let dense_out = vec![0.0f32; model.input_dense.nb_neurons];
        let buf = vec![0.0f32; model.noise_gru.nb_inputs];
        let denoise_buf = vec![0.0f32; model.denoise_gru.nb_inputs];
        RnnState {
            model,
            vad_gru_state,
            noise_gru_state,
            denoise_gru_state,
            scratch,
            dense_out,
            buf,
            denoise_buf,
        }
    }

    /// Resets the recurrent state, as though no audio had been seen yet.
    pub(crate) fn reset(&mut self) {
        for s in self
            .vad_gru_state
            .iter_mut()
            .chain(&mut self.noise_gru_state)
            .chain(&mut self.denoise_gru_state)
        {
            *s = 0.0;
        }
    }

    pub(crate) fn compute(&mut self, gains: &mut [f32], vad: &mut [f32], input: &[f32]) {
        assert_eq!(input.len(), crate::NB_FEATURES);
        assert_eq!(gains.len(), crate::NB_BANDS);
        assert_eq!(vad.len(), 1);

        let model = &self.model;
        let nd = model.input_dense.nb_neurons;
        let nv = model.vad_gru.nb_neurons;
        let nn = model.noise_gru.nb_neurons;

        model.input_dense.compute(&mut self.dense_out, input);
        model
            .vad_gru
            .compute(&mut self.vad_gru_state, &self.dense_out, &mut self.scratch);
        model.vad_output.compute(vad, &self.vad_gru_state);

        self.buf[..nd].copy_from_slice(&self.dense_out);
        self.buf[nd..(nd + nv)].copy_from_slice(&self.vad_gru_state);
        self.buf[(nd + nv)..].copy_from_slice(input);
        model
            .noise_gru
            .compute(&mut self.noise_gru_state, &self.buf, &mut self.scratch);

        self.denoise_buf[..nv].copy_from_slice(&self.vad_gru_state);
        self.denoise_buf[nv..(nv + nn)].copy_from_slice(&self.noise_gru_state);
        self.denoise_buf[(nv + nn)..].copy_from_slice(input);
        model.denoise_gru.compute(
            &mut self.denoise_gru_state,
            &self.denoise_buf,
            &mut self.scratch,
        );
        model.denoise_output.compute(gains, &self.denoise_gru_state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_model_has_the_expected_shape() {
        let m = RnnModel::default();
        assert_eq!(m.input_dense.nb_inputs(), 42);
        assert_eq!(m.input_dense.nb_neurons(), 24);
        assert_eq!(m.vad_gru.nb_neurons(), 24);
        assert_eq!(m.noise_gru.nb_neurons(), 48);
        assert_eq!(m.denoise_gru.nb_neurons(), 96);
        assert_eq!(m.denoise_output.nb_neurons(), 22);
        assert_eq!(m.vad_output.nb_neurons(), 1);
        assert_eq!(m.max_neurons(), 96);
    }

    /// A v1 model must survive a round trip through the v2 encoder unchanged.
    #[test]
    fn v2_round_trip_preserves_weights() {
        let original = RnnModel::default();
        let encoded = original.to_bytes();
        assert_eq!(&encoded[..4], MAGIC_V2);
        let decoded = RnnModel::from_bytes(&encoded).expect("v2 model should parse");

        assert_eq!(
            decoded.denoise_gru.nb_neurons,
            original.denoise_gru.nb_neurons
        );
        assert_eq!(
            decoded.denoise_gru.input_weights.to_i8(),
            original.denoise_gru.input_weights.to_i8()
        );
        assert_eq!(
            decoded.vad_output.bias.to_i8(),
            original.vad_output.bias.to_i8()
        );
        assert_eq!(
            decoded.input_dense.input_weights.len(),
            original.input_dense.input_weights.len()
        );
    }

    /// Both encodings must drive inference to the same answer.
    #[test]
    fn v1_and_v2_models_infer_identically() {
        let v1 = RnnModel::default();
        let v2 = RnnModel::from_bytes(&v1.to_bytes()).unwrap();
        let features: Vec<f32> = (0..crate::NB_FEATURES)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();

        let mut s1 = RnnState::new(Cow::Owned(v1));
        let mut s2 = RnnState::new(Cow::Owned(v2));
        let (mut g1, mut g2) = ([0.0; crate::NB_BANDS], [0.0; crate::NB_BANDS]);
        let (mut v1o, mut v2o) = ([0.0], [0.0]);
        for _ in 0..5 {
            s1.compute(&mut g1, &mut v1o, &features);
            s2.compute(&mut g2, &mut v2o, &features);
        }
        assert_eq!(g1, g2);
        assert_eq!(v1o, v2o);
    }

    #[test]
    fn truncated_and_oversized_models_are_rejected() {
        let good = RnnModel::default().to_bytes();
        assert!(RnnModel::from_bytes(&good[..good.len() - 1]).is_none());

        let mut extra = good.clone();
        extra.push(0);
        assert!(RnnModel::from_bytes(&extra).is_none());

        let mut bad_magic = good.clone();
        bad_magic[3] = b'X';
        assert!(RnnModel::from_bytes(&bad_magic).is_none());

        let mut bad_version = good;
        bad_version[4] = 9;
        assert!(RnnModel::from_bytes(&bad_version).is_none());
    }

    /// The v1 format cannot express a layer wider than 127, which is the reason v2 exists.
    #[test]
    fn v2_header_can_describe_wide_layers() {
        let mut r = Reader {
            bytes: to_i8(&[0x00, 0x02, 0x00, 0x00]),
            pos: 0,
            v2: true,
        };
        assert_eq!(r.dim(), Some(512));

        let mut too_big = Reader {
            bytes: to_i8(&[0xff, 0xff, 0xff, 0xff]),
            pos: 0,
            v2: true,
        };
        assert_eq!(too_big.dim(), None);
    }
}
