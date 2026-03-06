Application.ensure_all_started(:nx)
Application.ensure_all_started(:recgpt)
Ecto.Migrator.run(RecGPT.Repo, :up, all: true)

# Excluded tags: integration (disk I/O).
ExUnit.configure(
  exclude: [
    integration: true,
    eval: true
  ]
)
