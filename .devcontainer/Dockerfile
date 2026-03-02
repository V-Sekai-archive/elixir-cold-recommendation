# Elixir (hexpm) base for mix/elixir. GPU tools (nvidia-smi) are injected at
# runtime by the NVIDIA Container Toolkit when the container is run with --gpus all.
FROM hexpm/elixir:1.18.0-erlang-27.2.1-debian-bookworm-20251103

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl gnupg git ca-certificates dos2unix \
    && rm -rf /var/lib/apt/lists/*

RUN mix local.hex --force && mix local.rebar --force
