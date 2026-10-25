#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/rust/hearth/vendor/cubecl/crates/cubecl-wgpu/src/compute/server.rs"

if [[ ! -f "$TARGET" ]]; then
  echo "Hearth CubeCL source is missing: $TARGET" >&2
  echo "Check out submodules recursively before building." >&2
  exit 1
fi

python3 - "$TARGET" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1])
source = target.read_text()
marker = "// Ponko Android SPIR-V cache compatibility: std_io is desktop-only in CubeCL."

if marker in source:
    print("CubeCL Android SPIR-V cache compatibility patch already applied.")
    raise SystemExit(0)

old = '''            #[cfg(feature = "spirv")]
            spirv_cache: config.compilation.cache.as_ref().map(|cache_config| {
                Cache::new(
                    "spirv",
                    CacheOption::default()
                        .name("wgpu-spirv")
                        .root(cache_config.root()),
                )
            }),
'''
new = '''            #[cfg(feature = "spirv")]
            spirv_cache: {
                // Ponko Android SPIR-V cache compatibility: std_io is desktop-only in CubeCL.
                #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
                {
                    config.compilation.cache.as_ref().map(|cache_config| {
                        Cache::new(
                            "spirv",
                            CacheOption::default()
                                .name("wgpu-spirv")
                                .root(cache_config.root()),
                        )
                    })
                }
                #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
                {
                    None
                }
            },
'''

if old not in source:
    raise SystemExit(
        "Expected CubeCL SPIR-V cache initializer was not found. "
        "Refusing to patch an unknown upstream source revision."
    )

target.write_text(source.replace(old, new, 1))
print("Applied Android-only CubeCL SPIR-V cache compatibility patch.")
PY
