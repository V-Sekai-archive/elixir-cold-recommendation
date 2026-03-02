# RecGPT — Credo config
%{
  configs: [
    %{
      name: "default",
      files: %{included: ["lib/", "test/", "mix/"], excluded: []},
      checks: [
        # Allow mixed line endings (Windows vs Unix) so CI and local both pass
        {Credo.Check.Consistency.LineEndings, false},
        # Relax refactoring thresholds for inference/eval/decode/serve/pt_loader
        {Credo.Check.Refactor.Nesting, [max_nesting: 4]},
        # Lib and test: complexity 15; mix tasks excluded (see mix config)
        {Credo.Check.Refactor.CyclomaticComplexity, [max_complexity: 15, files: %{excluded: ["mix/"]}]}
      ]
    },
    %{
      name: "mix",
      files: %{included: ["mix/"], excluded: []},
      checks: [
        {Credo.Check.Refactor.CyclomaticComplexity, [max_complexity: 25]}
      ]
    }
  ]
}
