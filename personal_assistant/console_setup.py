"""Explicitly build the fixed Linux console image using local Docker."""
import argparse
from importlib.resources import files, as_file
from pathlib import Path
import subprocess
import tempfile

from personal_assistant.services.docker_console import DockerConsole, IMAGE, _host_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    docker, endpoint = DockerConsole(Path.cwd(), Path.cwd()).connection()
    with as_file(files("personal_assistant").joinpath("console_image")) as definition:
        with tempfile.TemporaryDirectory(prefix="pa-docker-config-") as config:
            return subprocess.run([docker, "--config", config, "--host", endpoint, "build", "--tag", IMAGE, str(definition)],
                                  env=_host_environment(), timeout=600).returncode


if __name__ == "__main__":
    raise SystemExit(main())
