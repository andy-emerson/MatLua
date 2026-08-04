//! Build PUC Lua 5.4 (vendored) when the `lua` feature is enabled.

fn main() {
    if std::env::var("CARGO_FEATURE_LUA").is_err() {
        return;
    }

    let lua_src = "vendor/lua/src";
    println!("cargo:rerun-if-changed={lua_src}");
    println!("cargo:rerun-if-changed=build.rs");
    // Without this, toggling APICHECK is silently ignored on incremental builds.
    println!("cargo:rerun-if-env-changed=MATLUA_LUA_APICHECK");

    let sources = [
        "lapi.c",
        "lauxlib.c",
        "lbaselib.c",
        "lcode.c",
        "lcorolib.c",
        "lctype.c",
        "ldblib.c",
        "ldebug.c",
        "ldo.c",
        "ldump.c",
        "lfunc.c",
        "lgc.c",
        "linit.c",
        "liolib.c",
        "llex.c",
        "lmathlib.c",
        "lmem.c",
        "loadlib.c",
        "lobject.c",
        "lopcodes.c",
        "loslib.c",
        "lparser.c",
        "lstate.c",
        "lstring.c",
        "lstrlib.c",
        "ltable.c",
        "ltablib.c",
        "ltm.c",
        "lundump.c",
        "lutf8lib.c",
        "lvm.c",
        "lzio.c",
    ];

    let mut build = cc::Build::new();
    build.include(lua_src);
    build.warnings(false);
    // Host-style embed: portable defaults. APICHECK can be forced by env for CI.
    if std::env::var("MATLUA_LUA_APICHECK").is_ok() {
        build.define("LUA_USE_APICHECK", None);
    }
    // Gate on the TARGET OS: #[cfg(target_os)] in a build script tests the
    // HOST, which links the wrong configuration when cross-compiling.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        build.define("LUA_USE_POSIX", None);
        build.define("LUA_USE_DLOPEN", None);
    }
    for file in sources {
        build.file(format!("{lua_src}/{file}"));
    }
    build.compile("lua54");

    if target_os != "windows" {
        println!("cargo:rustc-link-lib=m");
    }
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=dl");
    }
}
