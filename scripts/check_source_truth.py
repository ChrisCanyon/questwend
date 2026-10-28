"""Reject untracked production sources, retired files, and local-path leaks.

This is intentionally independent of the native toolchain so it can run on
every branch and pull request before expensive model benchmarks.
"""

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = json.loads((ROOT / "production-layout.json").read_text(encoding="utf-8"))
ERRORS = []

cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
match = re.search(r"add_library\(questwend_core STATIC\s+(.*?)\)", cmake, re.S)
if not match:
    ERRORS.append("Cannot find the production core source list in CMakeLists.txt")
else:
    compiled = set(re.findall(r"core/[\w.-]+\.cpp", match.group(1)))
    declared = set(LAYOUT["core_sources"])
    if compiled != declared:
        ERRORS.append(f"CMake/layout mismatch: CMake-only={sorted(compiled-declared)}, "
                      f"layout-only={sorted(declared-compiled)}")

declared = set(LAYOUT["core_sources"])
actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "core").glob("*.cpp")}
if actual != declared:
    ERRORS.append(f"Core source mismatch: undeclared={sorted(actual-declared)}, "
                  f"missing={sorted(declared-actual)}")

for relative in LAYOUT["retired_sources"]:
    if (ROOT / relative).exists():
        ERRORS.append(f"Retired source returned: {relative}")

for path in [ROOT / "README.md", *ROOT.glob("core/*.cpp"), *ROOT.glob("server/*")]:
    if not path.is_file():
        continue
    body = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"[A-Za-z]:[/\\](?:Users|workspace)[/\\]", body, re.I):
        ERRORS.append(f"Local absolute path in {path.relative_to(ROOT)}")

for flag in ("QWEN_STATIC_PREFILL_CACHE", "QWEN_COLD_PREFILL_WORKSPACE",
             "QWEN_COALESCE", "QWEN_COALESCE_GAP"):
    for path in (ROOT / "core").glob("*"):
        if path.is_file() and flag in path.read_text(encoding="utf-8", errors="replace"):
            ERRORS.append(f"Retired experiment switch {flag} in {path.relative_to(ROOT)}")

if ERRORS:
    print("Source boundary check failed:\n" + "\n".join(f"- {e}" for e in ERRORS), file=sys.stderr)
    sys.exit(1)
print(f"Source boundary check passed: {len(declared)} core files, "
      f"{len(LAYOUT['retired_sources'])} retired paths excluded")
