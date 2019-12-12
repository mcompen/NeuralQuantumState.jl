"""
    NETSETTINGS

Generate settings by specifying values for non-default fields. Parsed with
`Parameters.jl`.

# Arguments
- `modelname::String`: name of model (see `src/weights.jl`).
- `test_iter::Int`: Number of independent consecutive Carleo/Troyer runs.
- `repetitions::Int`: Number of wavefunction optimizations.
- `n::Int`: Number of visible neurons.
- `α::Int`: Hidden unit density (No. hidden / No. visible).
- `γ_init::Float`: Learning parameter (default: `0.05`).
- `γ_decay::Float`: Decay of learning parameter (default: `1.0`).
- `mc_trials::Int`: Number of variational derivatives PER repetition PER CPU.
- `mc_steps::Int`: Number of MH steps per sample (default: `n`).
- `regulator::Float`: Regularization parameter of the S-matrix.
- `init_therm_steps::Int`: Burn-in steps on first repetition.
- `therm_steps::Int`: Burn-in on other repetitions.
- `nflips::Int`: Number of flips to be performed per mc_trial (default: `n`).
- `pbc::Bool`: Periodic boundary conditions for lattice (default: `true`).
- `mag0::Bool`: Sample only in sector ∑s_z = 0 (default: `false`).
- `mfree::Bool`: Matrix-free inversion (default: `true`). Overridden to `false`.
    when `iterative_inverse = false`.
- `iterative_inverse::Bool`: Invert S-matrix iteratively (default: `true`).
    Assumed true when `mfree = true`.
- `calc_stat::Bool`: Calculate single-site statistics and correlations after
    last repetition.
- `stat_samples::Int`: No. of samples PER CPU for statistics calculation.
- `save_figures::Bool `: Save statistics in figure (default: `true`).
- `writetofile::Bool`: Write optimized NQS parameters, energy[, statistics] to a
    file.

# Examples
```jldoctest
julia> NetSettings = NETSETTINGS(
         modelname = \"U_afh\",
         repetitions =1000,
         n = 6,
         α = 3,
         mc_trials = 500,
         init_therm_steps = 100,
         therm_steps = 50,
         stat_samples = 2000)
NETSETTINGS{String,Int64,Float64,Bool}
modelname: String \"U_afh\"
test_iter: Int64 1
repetitions: Int64 1000
n: Int64 6
α: Int64 3
m: Int64 18
dim_rbm: Int64 132
γ_init: Float64 0.05
γ_decay: Float64 1.0
mc_trials: Int64 500
mc_steps: Int64 6
regulator: Float64 0.1
init_therm_steps: Int64 100
therm_steps: Int64 50
nflips: Int64 1
pbc: Bool true
mag0: Bool false
mfree: Bool true
iterative_inverse: Bool true
calc_stat: Bool true
stat_samples: Int64 5000
save_figures: Bool true
writetofile: Bool true
```
"""
@with_kw struct NETSETTINGS{S<:String, T<:Int, U<:Float64, V<:Bool}
    modelname::S
    test_iter::T = 1            # Number of independent consecutive Carleo/Troyer runs
    repetitions::T              # Number of wavefunction optimizations
    n::T                        # Number of visible neurons
    α::T                        # No. hidden / No. visible
    m::T = α * n                # Number of hidden neurons
    dim_rbm::T = m * n + m + n  # Total amount of free parameters
    γ_init::U = 0.05            # Learning parameter
    γ_decay::U = 1.0
    mc_trials::T                # Number of var derivatives per repetition
    mc_steps::T = n             # Number of MH steps per sample
    regulator::U = 0.1
    init_therm_steps::T         # burn-in / thermalization first iter
    therm_steps::T              # burn-in / thermalization
    nflips::T = 1               # Number of flips to be performed per MH step
    pbc::V = true               # Periodic boundary conditions. See also `weights.jl`.
    mag0::V = false             # Sample only in sector ∑s_z = 0
    mfree::V = true             # assumed false when iterative_inverse = false
    iterative_inverse::V = true # assumed true when mfree = true. true/false when mfree = false.
    calc_stat::V = true         # Calculate statistics
    stat_samples::T             # No. of samples for statistics calculation.
    save_figures::V = true      # Save statistics as figure
    writetofile::V = true       # write network parameters / energy to a file
end

mutable struct WEIGHTS{S<:Array{Float64, 1}, T<:Array{Float64, 2}}
    w_x::S
    w_y::S
    w_z::S
    w_xx::T
    w_yy::T
    w_zz::T
end

mutable struct STATISTICS{S<:Array{Float64, 1},
    T<:Array{Float64, 2}}
    s_x::S
    s_y::S
    s_z::S
    s_xx::T
    s_yy::T
    s_zz::T
end

mutable struct NETPARAMS{S<:SharedArray{Complex{Float64}, 1},
    T<:SharedArray{Complex{Float64}, 2}}  # Struct for typing of all network parameters
    a::S           # Bias on visible neurons
    b::S           # Hidden neurons
    w::T           # Connection matrix
end

mutable struct NETSTATE{S<:Array{Int, 1},
        T<:Array{Complex{Float64}, 1},
        U<:Complex{Float64}}
    state::S       # State of visible neurons
    theta::T       # Look-up table
    theta_upd::T   # Updated look-up table for a MH step
    logcoshsum::U    # cosh product lookup
    logcoshsum_upd::U
end

mutable struct VARDERIVS{S<:SharedArray{Complex{Float64}, 3},
    T<:Float64}
    o_tot::S
    reg::T
end

mutable struct OPTPARAMS{S<:Array{Complex{Float64}, 2},
    T<:Array{Complex{Float64}, 1}}  # Sets the types of the optimization variables
    s::S           # S-matrix
    f::T           # Forces
    dw_tot::T
end

mutable struct OPTPARAMSMFREE{S<:Array{Complex{Float64}, 1}}
    f::S
    dw_tot::S
end

mutable struct CGPARAMS{S<:Array{Complex{Float64}, 1},
        T<:SharedArray{Complex{Float64}, 1},
        U<:SharedArray{Complex{Float64}, 2},
        V<:Float64}
    r::S
    dir::T
    dw::S
    Sdir_w::U
    Sdir::S
    res::V
    res_prev::V
end
