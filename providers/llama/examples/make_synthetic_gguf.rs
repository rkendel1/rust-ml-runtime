//! Writes a synthetic (random-weight) GGUF fixture:
//!   cargo run --example make_synthetic_gguf -- <out.gguf> [tiny|100m]
use ml_runtime_llama_provider::synthetic::{write_synthetic_gguf, SyntheticSpec};

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let out = args
        .next()
        .expect("usage: make_synthetic_gguf <out.gguf> [tiny|100m]");
    let spec = match args.next().as_deref() {
        None | Some("tiny") => SyntheticSpec::tiny(),
        Some("100m") => SyntheticSpec::bench_100m(),
        Some(other) => panic!("unknown preset {other}"),
    };
    write_synthetic_gguf(std::path::Path::new(&out), &spec)?;
    println!("wrote {out} ({})", spec.name);
    Ok(())
}
