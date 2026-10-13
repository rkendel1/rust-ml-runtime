//! Deterministic synthetic GGUF fixtures (a tiny llama-architecture model with
//! seeded **random weights**).
//!
//! These files exercise the real llama.cpp engine end to end (tokenizer,
//! chat template, forward pass, sampling, grammar) without downloading
//! anything, so default test runs need no network and no model licence. They
//! generate gibberish by construction and say nothing about the quality of any
//! real model. Not for production use.

use std::{fs, io, path::Path};

/// Shape and behaviour of a synthetic model.
#[derive(Clone, Debug)]
pub struct SyntheticSpec {
    pub name: String,
    pub layers: u32,
    pub embd: u32,
    pub ff: u32,
    pub heads: u32,
    pub context: u32,
    pub seed: u64,
    /// When false the vocabulary has no EOS token, so generation only ends at a
    /// length or stop-sequence limit.
    pub with_eos: bool,
    /// Store matrices as F16 (realistic memory traffic) instead of F32.
    pub f16: bool,
}

impl SyntheticSpec {
    /// A fast fixture: ~100k parameters.
    pub fn tiny() -> Self {
        Self {
            name: "synthetic-tiny".into(),
            layers: 2,
            embd: 64,
            ff: 128,
            heads: 4,
            context: 512,
            seed: 7,
            with_eos: true,
            f16: false,
        }
    }

    /// A benchmark fixture: ~100M parameters, F16, so decode speed is
    /// meaningful. Weights are random.
    pub fn bench_100m() -> Self {
        Self {
            name: "synthetic-100m".into(),
            layers: 12,
            embd: 768,
            ff: 2048,
            heads: 12,
            context: 2048,
            seed: 11,
            with_eos: false,
            f16: true,
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift64*, mapped to roughly N(0, 0.02) via a 4-sum approximation.
        let mut sum = 0.0;
        for _ in 0..4 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
            sum += (v >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        }
        sum * 0.035
    }
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if exp >= 31 {
        sign | 0x7c00
    } else if exp <= 0 {
        if exp < -10 {
            sign
        } else {
            let m = (mant | 0x0080_0000) >> (1 - exp);
            sign | ((m + 0x1000) >> 13) as u16
        }
    } else {
        sign | ((exp as u16) << 10) | ((mant + 0x1000) >> 13) as u16
    }
}

const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;

struct Tensor {
    name: String,
    dims: Vec<u64>,
    ty: u32,
    data: Vec<u8>,
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

enum Value {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strings(Vec<String>),
    F32s(Vec<f32>),
    I32s(Vec<i32>),
}

fn put_kv(out: &mut Vec<u8>, key: &str, value: &Value) {
    put_string(out, key);
    match value {
        Value::U32(v) => {
            out.extend(4u32.to_le_bytes());
            out.extend(v.to_le_bytes());
        }
        Value::F32(v) => {
            out.extend(6u32.to_le_bytes());
            out.extend(v.to_le_bytes());
        }
        Value::Bool(v) => {
            out.extend(7u32.to_le_bytes());
            out.push(u8::from(*v));
        }
        Value::Str(v) => {
            out.extend(8u32.to_le_bytes());
            put_string(out, v);
        }
        Value::Strings(v) => {
            out.extend(9u32.to_le_bytes());
            out.extend(8u32.to_le_bytes());
            out.extend((v.len() as u64).to_le_bytes());
            v.iter().for_each(|s| put_string(out, s));
        }
        Value::F32s(v) => {
            out.extend(9u32.to_le_bytes());
            out.extend(6u32.to_le_bytes());
            out.extend((v.len() as u64).to_le_bytes());
            v.iter().for_each(|f| out.extend(f.to_le_bytes()));
        }
        Value::I32s(v) => {
            out.extend(9u32.to_le_bytes());
            out.extend(5u32.to_le_bytes());
            out.extend((v.len() as u64).to_le_bytes());
            v.iter().for_each(|i| out.extend(i.to_le_bytes()));
        }
    }
}

const CHATML: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

/// Writes the synthetic model to `path` (GGUF v3).
pub fn write_synthetic_gguf(path: &Path, spec: &SyntheticSpec) -> io::Result<()> {
    // Vocabulary: <unk>, <s>, </s> (EOS only when requested), 256 byte tokens,
    // and one piece per printable ASCII character (space is U+2581).
    let mut tokens: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    let mut scores: Vec<f32> = vec![0.0; 3];
    let mut types: Vec<i32> = vec![2, 3, 3];
    for byte in 0..=255u32 {
        tokens.push(format!("<0x{byte:02X}>"));
        scores.push(-1000.0);
        types.push(6);
    }
    for ch in 32u8..=126 {
        tokens.push(if ch == b' ' {
            "\u{2581}".into()
        } else {
            (ch as char).to_string()
        });
        scores.push(0.0);
        types.push(1);
    }
    let vocab = tokens.len() as u64;
    let (embd, ff, heads) = (spec.embd as u64, spec.ff as u64, spec.heads);
    assert!(
        spec.embd.is_multiple_of(heads),
        "embd must divide evenly by heads"
    );

    let mut rng = Rng(spec.seed.max(1) | 1);
    let mut tensors: Vec<Tensor> = Vec::new();
    let matrix = |name: String, rows: u64, cols: u64, rng: &mut Rng| {
        let n = (rows * cols) as usize;
        let mut data = Vec::with_capacity(n * if spec.f16 { 2 } else { 4 });
        for _ in 0..n {
            let v = rng.next();
            if spec.f16 {
                data.extend(f32_to_f16(v).to_le_bytes());
            } else {
                data.extend(v.to_le_bytes());
            }
        }
        Tensor {
            name,
            dims: vec![cols, rows],
            ty: if spec.f16 { GGML_F16 } else { GGML_F32 },
            data,
        }
    };
    let norm = |name: String, n: u64| Tensor {
        name,
        dims: vec![n],
        ty: GGML_F32,
        data: (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
    };
    tensors.push(matrix("token_embd.weight".into(), vocab, embd, &mut rng));
    tensors.push(norm("output_norm.weight".into(), embd));
    for layer in 0..spec.layers {
        let p = |s: &str| format!("blk.{layer}.{s}");
        tensors.push(norm(p("attn_norm.weight"), embd));
        for t in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            tensors.push(matrix(p(&format!("{t}.weight")), embd, embd, &mut rng));
        }
        tensors.push(norm(p("ffn_norm.weight"), embd));
        tensors.push(matrix(p("ffn_gate.weight"), ff, embd, &mut rng));
        tensors.push(matrix(p("ffn_up.weight"), ff, embd, &mut rng));
        tensors.push(matrix(p("ffn_down.weight"), embd, ff, &mut rng));
    }

    let mut kv: Vec<(&str, Value)> = vec![
        ("general.architecture", Value::Str("llama".into())),
        ("general.name", Value::Str(spec.name.clone())),
        ("llama.context_length", Value::U32(spec.context)),
        ("llama.embedding_length", Value::U32(spec.embd)),
        ("llama.block_count", Value::U32(spec.layers)),
        ("llama.feed_forward_length", Value::U32(spec.ff)),
        ("llama.attention.head_count", Value::U32(heads)),
        ("llama.attention.head_count_kv", Value::U32(heads)),
        ("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
        ("llama.rope.dimension_count", Value::U32(spec.embd / heads)),
        ("tokenizer.ggml.model", Value::Str("llama".into())),
        ("tokenizer.ggml.tokens", Value::Strings(tokens)),
        ("tokenizer.ggml.scores", Value::F32s(scores)),
        ("tokenizer.ggml.token_type", Value::I32s(types)),
        ("tokenizer.ggml.unknown_token_id", Value::U32(0)),
        ("tokenizer.ggml.bos_token_id", Value::U32(1)),
        ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
        ("tokenizer.chat_template", Value::Str(CHATML.into())),
    ];
    if spec.with_eos {
        kv.push(("tokenizer.ggml.eos_token_id", Value::U32(2)));
    }

    let mut out = Vec::new();
    out.extend(b"GGUF");
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kv.len() as u64).to_le_bytes());
    for (key, value) in &kv {
        put_kv(&mut out, key, value);
    }
    let mut offset = 0u64;
    for t in &tensors {
        put_string(&mut out, &t.name);
        out.extend((t.dims.len() as u32).to_le_bytes());
        t.dims.iter().for_each(|d| out.extend(d.to_le_bytes()));
        out.extend(t.ty.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += (t.data.len() as u64).div_ceil(32) * 32;
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    for t in &tensors {
        out.extend(&t.data);
        out.resize(out.len().div_ceil(32) * 32, 0);
    }
    fs::write(path, out)
}
