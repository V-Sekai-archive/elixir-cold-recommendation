Application.ensure_all_started(:nx)

# Excluded tags: integration (disk I/O).
ExUnit.configure(
  exclude: [
    integration: true,
    eval: true
  ]
)
