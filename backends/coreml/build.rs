fn main() {
    // `objc` 0.2 macros still inspect this legacy Clippy feature in expansions.
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"cargo-clippy\"))");

    if std::env::var_os("CARGO_CFG_TARGET_OS").is_some_and(|value| value == "macos") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
