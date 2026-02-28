Application.ensure_all_started(:nx)

# Excluded tags: integration (disk I/O), e2e_serve/serve_parity (reflex-logic-other).
ExUnit.configure(
  exclude: [
    integration: true,
    eval: true,
    e2e_serve: true,
    serve_parity: true
  ]
)
