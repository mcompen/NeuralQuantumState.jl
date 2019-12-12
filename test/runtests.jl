using Distributed
using Test
addprocs(2)
@everywhere using Random
@everywhere using NeuralQuantumState

@testset "NeuralQuantumState.jl" begin
    @everywhere Random.seed!(1234321)
    NetSettings = NETSETTINGS(modelname = "U_afh",
        repetitions =1000,
        n = 6,
        α = 3,
        mag0 = true,
        γ_decay = 0.997,
        mc_trials = 500,
        writetofile=true,
        init_therm_steps = 100,
        therm_steps = 50,
        stat_samples = 2000)

    energy = run_NQS(NetSettings)
    @test isapprox(energy[end], -11.211102550927972, rtol=1e-3)
end
