"""
Methods for initializing the RBM and sampling from its distribution.
"""

#=
    f_logcoshsum(theta)
Inline function for calculating the log of '∏_i cosh(θ_i(NetState))'
=#
@inbounds @views @fastmath f_logcoshsum(theta) = log(prod(cosh.(theta)))

"""
    update_theta!(NetState, NetParams, flips, NetSettings)
In-place update of theta_upd for proposed spin flips.
"""
function update_theta!(NetState, NetParams, flips, NetSettings)
    @unpack m = NetSettings
    @unpack theta_upd, state = NetState
    @unpack w = NetParams

     for j=1:m
        for i in flips
             theta_upd[j] -= 2 * w[i,j] * state[i]
        end
    end
    NetState.logcoshsum_upd = f_logcoshsum(NetState.theta_upd)

end

"""
    psi_mh_revert(NetState, NetParams, flips, NetSettings)

Returns the ψ'/ψ ratio, given flips and network parameters. Does not modify
the network state. Used for energy calculation: see `find_energy`.
"""
function psi_mh_revert(NetState, NetParams, flips, NetSettings)
    @unpack theta, theta_upd, state = NetState
    @unpack a = NetParams
    update_theta!(NetState, NetParams, flips, NetSettings)
    logpsi_frac = 0.0im
    for i in flips
        logpsi_frac -= 2 * a[i] * state[i]
    end
    logpsi_frac += NetState.logcoshsum_upd - NetState.logcoshsum
    theta_upd .= theta
    return @fastmath exp(logpsi_frac)
end

"""
    psi_mh(NetState, NetParams, flips, NetSettings)

Returns the ψ'/ψ ratio, given the network state, the network parameters and the
flips. See also `flipper`  and `flipper_exchange`.
"""
function psi_mh(NetState, NetParams, flips, NetSettings)
    @unpack a = NetParams
    @unpack state = NetState

    update_theta!(NetState, NetParams, flips, NetSettings)
    logpsi_frac = 0.0im
    for i in flips
        logpsi_frac -= 2 * a[i] * state[i]
    end
    logpsi_frac += NetState.logcoshsum_upd - NetState.logcoshsum
    return @fastmath exp(logpsi_frac)
end

"""
    flipper_exchange(NetState, NetParams, NetSettings)

Performs a Metropolis-Hastings update. See also `psi_mh` and `flipper`.
Preserves magnetization.
"""
function flipper_exchange(NetState, NetParams, NetSettings)
    @unpack n = NetSettings
    @unpack state, theta, theta_upd = NetState

    flips = zeros(Int, 2)
    flips[1] = rand(1:n)
    flips[2] = flips[1] % n + 1  # ASSUMES PBC, exchange with Hamming dist 1

    while state[flips[1]] == state[flips[2]]
        flips[1] = flips[1] % n + 1
        flips[2] = flips[2] % n + 1
    end
    if abs2(psi_mh(NetState, NetParams, flips, NetSettings)) > rand()
        theta .= theta_upd
        NetState.logcoshsum = NetState.logcoshsum_upd
        state[flips] .= reverse(state[flips])
    else
        theta_upd .= theta
    end
end

"""
    flipper(NetState, NetParams, NetSettings)

Performs a Metropolis-Hastings update. See also `psi_mh`.
"""
function flipper(NetState, NetParams, NetSettings)
    @unpack nflips, n = NetSettings
    @unpack theta, state, theta_upd = NetState

    if nflips == 1
        flips  = rand(1:n)
    elseif nflips == 2
        if rand(Bool)
            flips = rand(1:n)
        else
            flips = sample(1:n, 2, replace=false)
        end
    end
    if @fastmath abs2(psi_mh(NetState, NetParams, flips, NetSettings)) > rand()
        theta .= theta_upd
        NetState.logcoshsum = NetState.logcoshsum_upd
        state[flips] *= -1
    else
        theta_upd .= theta
    end
end

"""
    find_energy(NetState, NetParams, HamiltWeights, NetSettings)

Saves the local energy in-place given Hamiltonian weight. See also
`mc_sampling()`.
"""
function find_energy(NetState, NetParams, HamiltWeights, NetSettings)
    @unpack w_x, w_y, w_z, w_xx, w_yy, w_zz = HamiltWeights
    @unpack state = NetState
    @unpack n = NetSettings

    eloc = 0.0im
    factor_tmp = 0.0im
    for i=1:n
        factor_tmp = w_x[i] + 1.0im * w_y[i] * state[i]
        if !iszero(factor_tmp)  # don't perform costly psi'/psi computation when multiplied by 0 anyway
            eloc += factor_tmp * psi_mh_revert(NetState, NetParams, i, NetSettings)
        end
        eloc += w_z[i] * state[i]
        for j=i+1:n
            state_ij_tmp = state[i] * state[j]
            factor_tmp = w_xx[i, j] - w_yy[i, j] * state_ij_tmp
            if !iszero(factor_tmp)
                eloc += factor_tmp * psi_mh_revert(NetState, NetParams, [i, j], NetSettings)
            end
            eloc += w_zz[i, j] * state_ij_tmp
        end
    end
    return eloc
end

"""
    derivs!(o_tot, NetState, NetSettings, trial, id)

In-place saving of variational derivatives `adjoint(O)` in `o_tot` for trial number `trial`.
"""
function derivs!(o_tot, NetState, NetSettings, trial, id)
    @unpack n, m = NetSettings
    @unpack state, theta = NetState

    o_tot[1:n, trial, id] .= state
    o_tot[n + 1:n + m, trial, id] .= conj.(tanh.(theta))
    @views o_tot[n + m + 1:end, trial, id] .= kron(o_tot[n + 1:n + m, trial, id], state)
end

"""
    reinit_state!(NetState, NetParams, NetSettings)

In-place reinitialization of spin state.
"""
function reinit_state!(NetState, NetParams, NetSettings)
    @unpack mag0, n = NetSettings
    @unpack state = NetState

    if mag0
        for i=1:Int(n/2)
            state[i] = -1
            state[end - i + 1] = +1
        end
        shuffle!(state)
    else
        state .= 2 * rand(0:1, n) .- 1
    end
    reset_theta!(NetState, NetParams)
end

"""
    reset_theta!(NetState, NetParams)

In-place recalculation look-up variables.
"""
function reset_theta!(NetState, NetParams)
    @unpack b, w = NetParams
    @unpack theta, theta_upd, state = NetState

    theta .= b + transpose(w) * state
    theta_upd .= theta
    NetState.logcoshsum = f_logcoshsum(theta)
end

"""
    initial_thermalize!(NetState, NetParams, NetSettings)

Initial thermalization for when the network parameters are just initialized.
"""
function initial_thermalize(NetState, NetParams, NetSettings)
    @unpack init_therm_steps, mc_steps, mag0 = NetSettings
    for i=1:init_therm_steps
        for j=1:mc_steps
            if mag0
                flipper_exchange(NetState, NetParams, NetSettings)
            else
                flipper(NetState, NetParams, NetSettings)
            end
        end
    end
end

"""
    thermalize(NetState, NetParams, NetSettings)

Thermalization after network parameter update.
"""
function thermalize(NetState, NetParams, NetSettings)
    @unpack therm_steps, mc_steps, mag0 = NetSettings
    for i=1:therm_steps
        for j=1:mc_steps
            if mag0
                flipper_exchange(NetState, NetParams, NetSettings)
            else
                flipper(NetState, NetParams, NetSettings)
            end
        end
    end
end

"""
    mc_sampling!(NetParams, o_tot, elocs, HamiltWeights, NetSettings[, warm_state])

# Arguments

- `NetParams`: the network parameters W={a,b,w}.
- `VarDervs`: a VARDERIVS-type with variational derivatives (updated in-place).
- `elocs`: a SharedArray-type, of which column `id` is updated in-place.
- `HamiltWeights`: the weights of the 1/2-local Hamiltonian. See `weights.jl`.
- `NetSettings`: The settings for optimization.
# Keywords

- `warm_state`: The `NetState` struct is dereferenced when the function is
called. However, the state could be passed to the master process without much
overhead to give the next Markov Chain a "warm start" when the network
parameters are updated.
"""
function mc_sampling!(NetParams, o_tot, elocs, HamiltWeights, NetSettings, id; warm_state=zeros(Int, 1))
    @unpack n, mc_trials, mc_steps, mag0 = NetSettings
    # Initialization
    NetState = init_state(NETSTATE, NetSettings, warm_state=warm_state)
    reset_theta!(NetState, NetParams)

    # Thermalization
    if warm_state[1] == 0  # first iter
        initial_thermalize(NetState, NetParams, NetSettings)
    else
        thermalize(NetState, NetParams, NetSettings)
    end

    # Sampling
    for t=1:mc_trials
        for i=1:mc_steps
            if mag0
                flipper_exchange(NetState, NetParams, NetSettings)
            else
                flipper(NetState, NetParams, NetSettings)
            end
        end
        derivs!(o_tot, NetState, NetSettings, t, id)
        elocs[t, id] = find_energy(NetState, NetParams, HamiltWeights, NetSettings)
    end
    return NetState.state
end

"""
    init_params(ParamStruct, NetSettings)
Constructs a `NETPARAMS` type with random network parameters. Note that these
parameters should be accessible from all processors and are thus SharedArrays.
"""
function init_params(ParamStruct, NetSettings)
    @unpack n, m, dim_rbm = NetSettings
    μ_params = 0.0
    σ_params = 0.01 / dim_rbm
    a = complex(1,0) * rand(Normal(μ_params, σ_params), n) +
        complex(0,1) * rand(Normal(μ_params, σ_params), n)
    b = complex(1,0) * rand(Normal(μ_params, σ_params), m) +
        complex(0,1) * rand(Normal(μ_params, σ_params), m)
    w = complex(1,0) * rand(Normal(μ_params, σ_params), n, m) +
        complex(0,1) * rand(Normal(μ_params, σ_params), n, m)
    netparameters = ParamStruct(SharedArray(a), SharedArray(b), SharedArray(w))
    return netparameters
end

"""
    init_state(StateStruct, NetSettings[, warm_state])
Constructs a `NETSTATE` type with a random/warm network state and the
corresponding look-up tables (θ and its cosh-product).
"""
function init_state(StateStruct, NetSettings; warm_state=zeros(Int, 1))
    @unpack n, mag0, m = NetSettings
    if warm_state[1] != 0
        state = warm_state
    elseif mag0 && warm_state[1] == 0
        state = zeros(Int, n)
        for i=1:Int(n/2)
            state[i] = -1
            state[end - i + 1] = +1
        end
        shuffle!(state)
    elseif !mag0 && warm_state[1] == 0
        state = 2 * rand(0:1, n) .- 1
    end
    theta = zeros(Complex{Float64}, m)
    theta_upd = copy(theta)
    logcoshsum = 0.0im
    logcoshsum_upd = 0.0im
    netstate = StateStruct(state, theta, theta_upd, logcoshsum, logcoshsum_upd)
end

"""
    init_vardervs(VarDervStruct, NetSettings)

Takes a Variational Derivative type as argument. Consequently initializes the
variational derivative matrix and the regularization parameter. NOTE that
adjoint(O) is saved during sampling, instead of O. This is done so that samples
can be stored in C-order in `derivs!` in `network.jl`. Generates a variational-
derivative matrix for each worker.
"""
function init_vardervs(VarDervStruct, NetSettings)       # Loads an Opt constructor and initiates
    @unpack regulator, dim_rbm, mc_trials = NetSettings
    o_tot = SharedArray(zeros(Complex{Float64}, dim_rbm, mc_trials, nworkers()))
    reg = regulator
    varderv_inst = VarDervStruct(o_tot, reg)
    return varderv_inst
end
