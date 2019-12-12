using Documenter, NeuralQuantumState

makedocs(modules = [NeuralQuantumState],
    sitename="NeuralQuantumState.jl")

deploydocs(
    repo = "github.com/mcompen/NeuralQuantumState.jl.git",
    devbranch = "master",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", devurl => devurl]
)
