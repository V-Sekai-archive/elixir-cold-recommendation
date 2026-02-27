# RecGPT — Credo config
%{
  configs: [
    %{
      name: "default",
      files: %{included: ["lib/", "test/"], excluded: []},
      checks: [
        # Allow mixed line endings (Windows vs Unix) so CI and local both pass
        {Credo.Check.Consistency.LineEndings, false}
      ]
    }
  ]
}
