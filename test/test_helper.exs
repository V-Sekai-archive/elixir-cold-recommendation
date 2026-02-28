Application.ensure_all_started(:nx)

# Excluded tags: embedding (HF download), integration (disk I/O), e2e_serve/serve_parity (reflex-logic-other).
ExUnit.configure(
  exclude: [
    embedding: true,
    integration: true,
    eval: true,
    e2e_serve: true,
    serve_parity: true,
    pt_fixture: true
  ]
)
