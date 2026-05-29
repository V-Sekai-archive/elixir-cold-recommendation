# Build-only container for verifying compilation on Linux (XLA has Linux prebuilds, not Windows).
# Use for CI/compilation check only. Training requires host GPU (WSL2 or Linux bare metal).
FROM hexpm/elixir:1.18.0-erlang-27.2.1-debian-bookworm-20251103 AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential git
ENV MIX_ENV=dev
COPY mix.exs mix.lock ./
RUN mix local.hex --force && mix local.rebar --force && mix deps.get
COPY config config
COPY lib lib
COPY priv priv
COPY test test
RUN mix compile
