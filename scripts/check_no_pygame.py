"""Fail if pygame or SDL2 library files are present in this Python environment."""

from importlib import metadata, util
from pathlib import Path
import sysconfig


def main():
    problems = []
    for name in ("pygame", "pygame-ce"):
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        problems.append(f"Installed distribution: {name}")
    if util.find_spec("pygame") is not None:
        problems.append("The pygame module is importable")
    roots = {Path(sysconfig.get_path(key)) for key in ("purelib", "platlib")}
    for root in roots:
        for path in root.rglob("*"):
            name = path.name.lower()
            if (
                path.is_file()
                and "sdl2" in name
                and (name.endswith((".dll", ".dylib")) or ".so" in name)
            ):
                problems.append(f"SDL2 library: {path}")
    if problems:
        raise SystemExit("\n".join(problems))
    print("PASS: no pygame distribution/module or SDL2 libraries in this environment")


if __name__ == "__main__":
    main()
