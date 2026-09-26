"""Explicit setup/diagnostics for the optional, isolated Browser Use environment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv

from config_service.paths import default_data_dir

PIN = 'browser-use==0.13.10'
INSTALL_TIMEOUT_SECONDS = 900


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='Create an isolated environment and install the pinned library')
    parser.add_argument('--check', action='store_true', help='Check dependencies and local browser without launching it')
    parser.add_argument('--environment', type=Path, default=default_data_dir() / 'browser-env')
    args = parser.parse_args(argv)
    root = args.environment.expanduser().absolute()
    python = root / 'bin' / 'python'
    if args.install:
        if sys.version_info < (3, 11):
            parser.error('Python 3.11 or newer is required')
        if root.exists() and not (root / 'pyvenv.cfg').is_file():
            parser.error('Refusing to install into an existing directory that is not a virtual environment')
        venv.EnvBuilder(with_pip=True).create(root)
        try:
            subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                            '-r', str(Path(__file__).with_name('browser-requirements.txt'))],
                           check=True, timeout=INSTALL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f'Browser dependency install exceeded {INSTALL_TIMEOUT_SECONDS} seconds; rerun --install.', file=sys.stderr)
            return 1
    if not python.is_file():
        print('Browser environment missing. Run this command with --install.', file=sys.stderr)
        return 1
    probe = subprocess.run([str(python), '-I', '-c',
                            'from importlib.metadata import version; print(version("browser-use"))'],
                           capture_output=True, text=True, timeout=15)
    candidates = [shutil.which(name) for name in ('chromium', 'chromium-browser', 'google-chrome')]
    candidates += ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                   '/Applications/Chromium.app/Contents/MacOS/Chromium']
    executable = next((str(path) for path in candidates if path and Path(path).is_file()), None)
    ready = probe.returncode == 0 and probe.stdout.strip() == PIN.split('==')[1]
    print(json.dumps({'dependency_ready': ready, 'python_path': str(python),
                      'browser_executable': executable,
                      'guidance': 'Select paths and headless mode in /settings. Install Chrome/Chromium separately if not found.'}, indent=2))
    return 0 if ready and executable else 1


if __name__ == '__main__':
    raise SystemExit(main())
