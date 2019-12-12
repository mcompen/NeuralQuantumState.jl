"""
Contains functions for calculating spin-statistics exactly when given a density
matrix, or through sampling when given RBM network parameters
"""

function init_stat(statStruct, n)
    s_x = zeros(Float64, n)
    s_y = zeros(Float64, n)
    s_z = zeros(Float64, n)
    s_xx = zeros(Float64, n, n)
    s_yy = zeros(Float64, n, n)
    s_zz = zeros(Float64, n, n)
    stat = statStruct(s_x, s_y, s_z, s_xx, s_yy, s_zz)
    return stat
end

"""
    stats_calculator(NetParams, HamiltWeights, warm_states)

Saves the statistics given (optimized) network parameters, the model hamiltonian
weights, the entire parallel RNG object, and "warm states" from the last markov
chain.
"""
function stats_calculator(NetParams, HamiltWeights, warm_states, NetSettings)
    @unpack n = NetSettings
    stats = init_stat(STATISTICS, n)
    eloc = 0.0
    mystats = [eloc, stats]
    @sync for j=1:nworkers()
        @async mystats .+= remotecall_fetch(remote_stats_sampler,
            workers()[j],
            NetParams,
            HamiltWeights,
            NetSettings,
            warm_state=warm_states[:, j])
    end

    mystats[1] /= nworkers()
    mystats[2] /= nworkers()
    mystats[2].s_xx .+= transpose(mystats[2].s_xx)
    mystats[2].s_yy .+= transpose(mystats[2].s_yy)
    mystats[2].s_zz .+= transpose(mystats[2].s_zz)
    return mystats
end

"""
    remote_stats_sampler(NetParams, HamiltWeights, NetSettings[, warm_state])

Returns sampled statistics given (optimized) RBM network parameters. See also
`find_stats!. Designed to be run in parallel.
"""
function remote_stats_sampler(NetParams, HamiltWeights, NetSettings; warm_state=zeros(Int, 1))
    @unpack stat_samples, mc_steps, n, mag0 = NetSettings
    NetState = init_state(NETSTATE, NetSettings, warm_state=warm_state)
    Statistics = init_stat(STATISTICS, n)
    reset_theta!(NetState, NetParams)
    thermalize(NetState, NetParams, NetSettings)
    eloc = 0.0

    for i=1:stat_samples
        for j=1:mc_steps
            if mag0
                flipper_exchange(NetState, NetParams, NetSettings)
            else
                flipper(NetState, NetParams, NetSettings)
            end
        end
        eloc = find_stats!(NetState, NetParams, Statistics, HamiltWeights, eloc, NetSettings)
    end
    eloc /= stat_samples
    Statistics /= stat_samples
    return eloc, Statistics
end

"""
    find_stats!(NetState, NetParams, Statistics, HamiltWeights, eloc, NetSettings)

In-place addition of local statistics to a Statistics struct, given RBM
network parameters and the spin state. Combinated with sampling of the local
energy to save computing time through re-usage of psi'/psi fractions. See also
 `sampled_statistics`.
"""

function find_stats!(NetState, NetParams, Statistics, HamiltWeights, eloc, NetSettings)
    @unpack n = NetSettings
    @unpack state = NetState
    @unpack w_x, w_y, w_z, w_xx, w_yy, w_zz = HamiltWeights
    @unpack s_x, s_y, s_z, s_xx, s_yy, s_zz = Statistics

    for i=1:n
        factor_i_tmp = w_x[i] + 1.0im * w_y[i] * state[i]
        psi_i_tmp = psi_mh_revert(NetState, NetParams, i, NetSettings)
        eloc += real(factor_i_tmp * psi_i_tmp)
        eloc += w_z[i] * state[i]
        s_x[i] += real(psi_i_tmp)
        s_y[i] += real(1.0im * state[i] * psi_i_tmp)
        s_z[i] += state[i]
        for j=i+1:n
            state_ij_tmp = state[i] * state[j]
            psi_ij_tmp = psi_mh_revert(NetState, NetParams, [i, j], NetSettings)
            factor_ij_tmp = w_xx[i, j] - w_yy[i, j] * state_ij_tmp
            eloc += real(factor_ij_tmp * psi_ij_tmp)
            eloc += w_zz[i, j] * state_ij_tmp
            s_xx[i, j] += real(psi_ij_tmp)
            s_yy[i, j] -= state_ij_tmp * real(psi_ij_tmp)
            s_zz[i, j] += state_ij_tmp
        end
    end
    return eloc
end

"""
    Base.:/(S::STATISTICS, x::Number)

Define division of Statistics for averaging over CPUs.
"""
function Base.:/(S::STATISTICS, x::Number)
    divS = init_stat(STATISTICS, size(S.s_x, 1))
    for k in fieldnames(STATISTICS)
        setfield!(divS, k, getfield(S, k) ./ x)
    end
    return divS
end

"""
    Base.:+(S1::STATISTICS, S2::STATISTICS)

Define subtraction of Statistics for QBM learning rule
"""
function Base.:+(S1::STATISTICS, S2::STATISTICS)
    addS = init_stat(STATISTICS, size(S1.s_x, 1))
    for k in fieldnames(STATISTICS)
        setfield!(addS, k, getfield(S1, k) + getfield(S2, k))  # CONVERTS SharedArray to Array..
    end
    return addS
end
