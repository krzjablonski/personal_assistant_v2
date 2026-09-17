# Official multi-platform Python 3.12.14 / Debian 12 image, resolved 2026-09-10.
FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
# Frozen Debian index keeps utility/dependency selection reproducible.
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260901T000000Z/ bookworm main\n' > /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends ripgrep findutils coreutils && \
    rm -rf /var/lib/apt/lists/*
COPY exec.py /opt/personal-assistant-console/exec.py
RUN chmod 0555 /opt/personal-assistant-console/exec.py && \
    mkdir /workspace /outputs /inputs && chmod 0555 /outputs /inputs
ENV PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 TZ=UTC PYTHONDONTWRITEBYTECODE=1
USER 1000:1000
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/python", "-I", "/opt/personal-assistant-console/exec.py"]
