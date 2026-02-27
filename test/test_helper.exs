Application.ensure_all_started(:nx)

# Excluded tags: embedding (HF download), compare_python / compare_embedding (fixtures), integration (disk I/O), e2e_steam / steam_parity (M:\reflex-logic-other)
ExUnit.configure(
  exclude: [
    embedding: true,
    compare_python: true,
    compare_embedding: true,
    integration: true,
    e2e_steam: true,
    steam_parity: true
  ]
)
