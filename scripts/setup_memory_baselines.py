import os
import argparse
from pathlib import Path
import subprocess
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("common", "memos"))
    parser.add_argument("--freeze-only", action="store_true", help="export installed environment without changing packages")
    args = parser.parse_args()
    for name in ([args.only] if args.only else ["common", "memos"]):
        destination = Path(os.path.join(Path(os.path.join(ROOT, '.venv-baselines')), name))
        if not args.freeze_only:
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(destination)
        executable = ('Scripts', 'python.exe') if os.name == 'nt' else ('bin', 'python')
        python = Path(os.path.join(destination, *executable))
        if not args.freeze_only:
            subprocess.run([str(python), "-m", "pip", "install", "-r",
                            str(Path(os.path.join(ROOT, 'scripts', 'requirements', f'baselines-{name}.txt')))], check=True)
        result = subprocess.run([str(python), "-m", "pip", "freeze"], text=True, capture_output=True, check=True)
        output = Path(os.path.join(ROOT, 'tests', 'sol', 'baseline-environment'))
        output.mkdir(parents=True, exist_ok=True)
        (Path(os.path.join(output, f'{name}-freeze.txt'))).write_text(result.stdout, encoding="utf-8")


if __name__ == "__main__":
    main()
