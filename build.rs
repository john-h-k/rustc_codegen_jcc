use std::env;
use std::path::PathBuf;

fn main() {
    // println!("cargo:rustc-link-arg=-fsanitize=addsress,undefined");

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=../jcc/build");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo::rerun-if-changed=../jcc/build/libjcc.a");
    println!("cargo:rustc-link-lib=jcc");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("jcc.h")
        .allowlist_file(".*/alloc.h")
        .allowlist_file(".*/aarch64.h")
        .allowlist_file(".*/compiler.h")
        .allowlist_file(".*/fs.h")
        .allowlist_file(".*/ir/ir.h")
        .clang_arg("-I../jcc/src")
        .derive_default(true)
        .derive_debug(true)
        .prepend_enum_name(false)
        .anon_fields_prefix("_")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("jcc.rs"))
        .expect("Couldn't write bindings!");
}
