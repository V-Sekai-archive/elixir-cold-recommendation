fixture = File.read!("priv/figgie_fixture.json") |> Jason.decode!()
sequences = fixture["sequences"]

# For simplicity, use all as train
train_sequences = %{
  "num_items" => 4,
  "sequences" => sequences
}

File.mkdir_p!("data/figgie")
File.write!("data/figgie/train_sequences.json", Jason.encode!(train_sequences, pretty: true))

# Create empty test for now
test_sequences = %{
  "num_items" => 4,
  "test_cases" => []
}
File.write!("data/figgie/test_sequences.json", Jason.encode!(test_sequences, pretty: true))

# Cold versions
cold_train = %{"num_items" => 4, "sequences" => []}
File.write!("data/figgie/cold_train_sequences.json", Jason.encode!(cold_train, pretty: true))

cold_test = %{"num_items" => 4, "test_cases" => []}
File.write!("data/figgie/cold_test_sequences.json", Jason.encode!(cold_test, pretty: true))

IO.puts("Figgie sequence files created in data/figgie/")
