using Distributed
using Test
addprocs(2)
@everywhere using Random
@everywhere using NeuralQuantumState

const energy_AFH_6_spins = -11.211102550927972

@testset "NeuralQuantumState.jl" begin
    @everywhere Random.seed!(1234321)
    NetSettings = NETSETTINGS(modelname = "U_afh",
        repetitions = 600,
        n = 6,
        α = 3,
        mag0 = true,
        γ_decay = 0.997,
        mc_trials = 200,
        writetofile = true,
        init_therm_steps = 100,
        therm_steps = 50,
        stat_samples = 2000,
        use_meter=false)

    energy = run_NQS(NetSettings)
    @test isapprox(energy[end], energy_AFH_6_spins, rtol=1e-3)

    NetSettings = NETSETTINGS(NetSettings, mag0=false) # Now without mag0
    energy = run_NQS(NetSettings)
    @test isapprox(energy[end], energy_AFH_6_spins, rtol=1e-2)
end
