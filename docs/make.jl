using Documenter, NeuralQuantumState

makedocs(modules = [NeuralQuantumState],
    authors= "Manu Compen"
    sitename="NeuralQuantumState.jl")

deploydocs(
    repo = "github.com/mcompen/NeuralQuantumState.jl.git",
    versions = ["stable" => "v^", "v#.#"]
)
