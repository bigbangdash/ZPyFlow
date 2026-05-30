# syntax=docker/dockerfile:1.4
#
# ZPyFlow development / CI image
# ─────────────────────────────
# Contains: Rust (stable, minimal) + Python 3.11 + maturin + dev deps
# Used by docker-compose.yml for build / test / bench services.

FROM python:3.11-slim AS base

# ── System packages ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
        libssl-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Rust (stable, minimal profile) ────────────────────────────────────────
# CARGO_HOME / RUSTUP_HOME are set to /usr/local so they persist in the image
# layer and can be shared across users.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH} \
    # Incremental compilation speeds up rebuilds inside the named volume.
    CARGO_INCREMENTAL=1

RUN curl https://sh.rustup.rs -sSf \
    | sh -s -- -y \
        --no-modify-path \
        --profile minimal \
        --default-toolchain stable \
    && rustup component add rustfmt clippy \
    && cargo --version && rustc --version

# ── Python tooling ─────────────────────────────────────────────────────────
# maturin develop requires a virtualenv; create one at /opt/venv and make it
# the active environment for all subsequent RUN and CMD steps.
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# maturin[patchelf]: patchelf fixes RPATH of the .so on Linux.
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir \
        "maturin[patchelf]>=1.5,<2" \
        -r /tmp/requirements-dev.txt

WORKDIR /app

# Default: drop to bash (overridden by docker-compose services).
CMD ["bash"]
