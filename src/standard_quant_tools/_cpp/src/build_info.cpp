#include "sqt/build_info.hpp"

// Generated into the build tree at build time by cmake/source_digest.cmake;
// it changes whenever any file under the native tree does.
#include "sqt_build_stamp.hpp"

// The configuration comes from a compile definition rather than the stamp:
// a multi-config generator (Visual Studio, Ninja Multi-Config) picks it when
// building, not when configuring, so one generated header cannot carry it
// for every configuration.
#ifndef SQT_BUILD_CONFIG
#define SQT_BUILD_CONFIG ""
#endif

namespace sqt {

BuildInfo build_info() noexcept {
    return BuildInfo{
        SQT_SOURCE_DIGEST,
        SQT_SOURCE_FILES,
        SQT_BUILD_CONFIG,
        SQT_BUILD_NATIVE_ARCH != 0,
        SQT_BUILD_COMPILER,
        SQT_BUILD_OPENMP,
        SQT_BUILD_PGO,
    };
}

}  // namespace sqt
