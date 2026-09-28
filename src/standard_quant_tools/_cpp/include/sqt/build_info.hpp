#pragma once

// Which sources and which build produced this extension.
//
// Exposed to Python as `_sqt_core.__build_info__`. The package compares
// `source_digest` against the same digest computed over the C++ sources
// beside it at import, and refuses an extension built from different ones:
// an extension that merely loads can be weeks older than the Python calling
// it, and every kernel it carries then answers with the old code while
// reporting itself available.
//
// Defined in src/build_info.cpp, the one translation unit that includes the
// build-time stamp, so a source edit recompiles that small file and relinks
// rather than recompiling the bindings.

namespace sqt {

struct BuildInfo {
    const char* source_digest;  // SHA-256 over the native tree, hex
    int         source_files;   // how many files that digest covers
    const char* build_type;     // the configuration compiled, e.g. "Release"
    bool        native_arch;    // host-CPU codegen was requested
    const char* compiler;       // compiler id and version
    const char* openmp;         // OpenMP version linked, "" when none
    const char* openmp_runtime; // the runtime library itself ("vcomp",
                                // "libomp", "libgomp", ...), "" when none;
                                // MSVC reports version 2.0 for vcomp and
                                // libomp alike, so only this tells them apart
    const char* pgo;            // "off", "generate" or "use"
};

BuildInfo build_info() noexcept;

}  // namespace sqt
