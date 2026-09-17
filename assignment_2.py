"""Compatibility entry point; implementation lives in assignment_2/codes/."""

import runpy
from pathlib import Path

_implementation = runpy.run_path(
    str(Path(__file__).resolve().parent / "assignment_2" / "codes" / "assignment_2.py")
)
globals().update(
    {
        name: value
        for name, value in _implementation.items()
        if not name.startswith("__")
    }
)

if __name__ == "__main__":
    _implementation["main"]()
