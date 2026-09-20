fn main() {
    if std::env::var_os("CARGO_CFG_TARGET_OS").is_some_and(|value| value == "macos") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
