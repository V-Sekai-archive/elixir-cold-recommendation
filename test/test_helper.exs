Application.ensure_all_started(:nx)
Application.ensure_all_started(:recgpt)

# Excluded tags: integration (disk I/O), eval (slow).
ExUnit.configure(
  exclude: [
    integration: true,
    eval: true
  ]
)
