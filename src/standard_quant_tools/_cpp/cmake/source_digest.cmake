# ── Source digest of the native tree, stamped into _sqt_core at BUILD time ────
#
# Run as `cmake -P` by the custom command in ../CMakeLists.txt, never
# included. It hashes every C++ source and CMake file under the native tree
# and writes a header that build_info.cpp compiles into the extension, where
# Python reads it back as `_sqt_core.__build_info__`.
#
# WHY IT EXISTS. An extension that loads is not an extension built from the
# code beside it. An editable install keeps its own compiled copy in
# site-packages while its Python resolves to the source tree, so every native
# fix after that copy was built is silently absent while the Python calling
# it is current -- and nothing that answers "is the extension there" can
# notice, because it is. The package recomputes this same digest over the
# sources beside it at import and refuses an extension whose stamp differs.
#
# WHY BUILD TIME AND NOT CONFIGURE TIME. A configure-time hash is only as
# fresh as the last `cmake -S/-B`, and an ordinary edit-then-`cmake --build`
# never reconfigures. As a custom command that DEPENDS on every file it
# hashes, the stamp moves with every rebuild that recompiles anything.
#
# THE RECIPE, which standard_quant_tools/_native_build.py repeats byte for
# byte (a test runs both on one tree and compares):
#   - every regular file under the native tree whose name ends in one of the
#     suffixes below, or is named CMakeLists.txt -- matched case-sensitively,
#     so both languages select the same files on every filesystem;
#   - paths relative to the native tree, '/'-separated, sorted bytewise;
#   - one line per file, "<sha256 of its bytes>  <relative path>\n" (the
#     `sha256sum` format, so a reader can reproduce it by hand);
#   - the digest is the SHA-256 of those lines concatenated.
# No timestamps and no absolute paths go in, so two checkouts of the same
# commit stamp the same digest wherever they live.
#
# Inputs (-D):
#   SQT_NATIVE_DIR  the native tree to hash (required)
#   SQT_OUTPUT      the header to write (required)
#   SQT_FACTS_FILE  a CMake file of configure-time build facts (optional)

cmake_minimum_required(VERSION 3.19)

if(NOT SQT_NATIVE_DIR OR NOT SQT_OUTPUT)
    message(FATAL_ERROR "source_digest.cmake needs -DSQT_NATIVE_DIR and -DSQT_OUTPUT")
endif()

# Defaults for a run without a facts file, e.g. a check of the recipe alone.
set(SQT_FACT_COMPILER "unknown")
set(SQT_FACT_NATIVE_ARCH 0)
set(SQT_FACT_OPENMP "")
set(SQT_FACT_PGO "off")
if(SQT_FACTS_FILE AND EXISTS "${SQT_FACTS_FILE}")
    include("${SQT_FACTS_FILE}")
endif()

file(GLOB_RECURSE _files LIST_DIRECTORIES false RELATIVE "${SQT_NATIVE_DIR}"
     "${SQT_NATIVE_DIR}/*")
list(FILTER _files INCLUDE REGEX
     "(\\.(cpp|cc|cxx|c|hpp|hh|hxx|h|inl|ipp|cmake)|(^|/)CMakeLists\\.txt)$")
list(SORT _files)

set(_manifest "")
set(_count 0)
foreach(_rel IN LISTS _files)
    # CRLF is read as LF, as the Python side does: a checkout that only
    # rewrote line endings is the same code, and hashing raw bytes called a
    # correct build stale. Text-mode READ keeps the bytes as they are (the
    # sources hold no NUL), and the quoted expansions keep any `;` intact.
    file(READ "${SQT_NATIVE_DIR}/${_rel}" _content)
    string(REPLACE "\r\n" "\n" _content "${_content}")
    string(SHA256 _file_hash "${_content}")
    string(APPEND _manifest "${_file_hash}  ${_rel}\n")
    math(EXPR _count "${_count} + 1")
endforeach()
string(SHA256 _digest "${_manifest}")

set(_header "// Generated at build time by _cpp/cmake/source_digest.cmake. Do not edit.
#pragma once
#define SQT_SOURCE_DIGEST \"${_digest}\"
#define SQT_SOURCE_FILES ${_count}
#define SQT_BUILD_COMPILER \"${SQT_FACT_COMPILER}\"
#define SQT_BUILD_NATIVE_ARCH ${SQT_FACT_NATIVE_ARCH}
#define SQT_BUILD_OPENMP \"${SQT_FACT_OPENMP}\"
#define SQT_BUILD_PGO \"${SQT_FACT_PGO}\"
")

# Rewritten only when the text changes, so touching a file without changing
# it does not recompile the translation unit that includes the header.
set(_existing "")
if(EXISTS "${SQT_OUTPUT}")
    file(READ "${SQT_OUTPUT}" _existing)
endif()
if(NOT _existing STREQUAL _header)
    file(WRITE "${SQT_OUTPUT}" "${_header}")
endif()
