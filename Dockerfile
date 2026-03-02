FROM hexpm/elixir:1.18.0-erlang-27.2.1-debian-bookworm-20251103 AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
ENV MIX_ENV=prod
COPY mix.exs mix.lock ./
RUN mix local.hex --force && mix local.rebar --force && mix deps.get --only prod
COPY config config
COPY lib lib
COPY priv priv
RUN mix compile && mix release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates
WORKDIR /app
COPY --from=builder /app/_build/prod/rel/recgpt .
ENV RECGPT_FIXTURE=/data/fixture.json
ENV RECGPT_CKPT_EXPORT=/data/ckpt
EXPOSE 50051 50052
CMD ["bin/recgpt", "eval", "RecGPT.ReleaseTasks.serve()"]
