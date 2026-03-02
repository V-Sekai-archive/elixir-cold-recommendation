# RecGPT — Credo config
%{
  configs: [
    %{
      name: "default",
      files: %{included: ["lib/", "test/", "mix/"], excluded: []},
      checks: [
        # Allow mixed line endings (Windows vs Unix) so CI and local both pass
        {Credo.Check.Consistency.LineEndings, false},
        # Relax refactoring thresholds
        {Credo.Check.Refactor.Nesting, [max_nesting: 5]},
        {Credo.Check.Refactor.CyclomaticComplexity,
         [
           max_complexity: 24,
           files: %{excluded: ["mix/"]}
         ]},
        {Credo.Check.Refactor.FunctionArity, [max_arity: 13]},
        {Credo.Check.Design.AliasUsage, false},
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
