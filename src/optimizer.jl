"""
    optimize!(CGP, VarDervs, OptParams, elocs, forces_w, γ[, iterative])

Optimize the network parameters in-place with Stochastic Reconfiguration by
MATRIX-FREE inversion of the S-matrix.

# Arguments
- `CGP::CGPARAMS`: A conjugate gradient parameter struct for when Conjugate
Gradient is performed instead of explicit inversion of the S-matrix.
- `VarDervs::VARDERIVS`: contains sampled variational derivatives and
regularisation parameter.
- `OptParams::OPTPARAMSMFREE`: contains optimization variables: forces,
wave function parameter update vector ΔW (NOT the S-matrix).
- `elocs`: Each column stores the sampled local energies from a worker.
- `forces_w`: Stores the result of adjoint(O) * E (both standardized) of each
worker in a column, the summed forces are stored in `OptParams.f`.
- `γ::Float64`: learning parameter
- `NetSettings`: struct containing all parameters of the RBM network

"""
function optimize!(CGP::CGPARAMS, VarDervs::VARDERIVS, OptParams::OPTPARAMSMFREE,
    elocs, forces_w, γ, NetSettings)
    @unpack o_tot = VarDervs
    @unpack f = OptParams
    o_tot .-=  mean(mean(o_tot, dims=2), dims=3) # Mean over all vardervs on all workers.
    elocs .-= mean(elocs)  # Mean over all eloc samples on all workers.
    mul!(f, forces_w, VarDervs, elocs, NetSettings)

    cg_iters = my_cg!(CGP, VarDervs, OptParams, γ, NetSettings, tol=1e-3)
    return cg_iters
end

"""
    optimize!(CGP, VarDervs, OptParams, elocs, forces_w, γ[, iterative])

Optimize the network parameters in-place with Stochastic Reconfiguration using
the EXPLICITLY BUILT S-matrix.

# Arguments

- `CGP::CGPARAMS`: A conjugate gradient parameter struct for when Conjugate
Gradient is performed instead of explicit inversion of the S-matrix.
- `VarDervs::VARDERIVS`: contains sampled variational derivatives and
regularisation parameter.
- `OptParams::OPTPARAMS`: contains optimization variables: S, S^(-1), forces,
wave function parameter update vector ΔW.
- `elocs`: Each column stores the sampled local energies from a worker.
- `forces_w`: Stores the result of adjoint(O) * E (both stanardized) of each
worker in a column, the summed forces are stored in `OptParams.f`.
- `γ::Float64`: learning parameter
- `NetSettings`: network (optimization/sampling) settings

"""
function optimize!(CGP::CGPARAMS, VarDervs::VARDERIVS, OptParams::OPTPARAMS,
    elocs, forces_w, γ, NetSettings)
    @unpack iterative_inverse, mc_trials, dim_rbm, regulator = NetSettings
    @unpack f, s, dw_tot = OptParams
    @unpack o_tot = VarDervs
    o_tot .-= mean(mean(o_tot, dims=2), dims=3)
    elocs .-= mean(elocs)

    mul!(f, forces_w, VarDervs, elocs, NetSettings)

    mul!(s, reshape(o_tot, dim_rbm, mc_trials * nworkers()),
        adjoint(reshape(o_tot, dim_rbm, mc_trials * nworkers())) )
    lmul!(1 / (mc_trials * nworkers()), s)

    if iterative_inverse
        my_cg!(CGP,
            VarDervs.reg * Matrix{Complex{Float64}}(I, dim_rbm, dim_rbm) + s,
            OptParams,
            γ,
            NetSettings,
            tol=1e-3)
    else
        inv_s = zeros(Complex{Float64}, 0, 0)
        try
            inv_s = inv(Hermitian(VarDervs.reg *
                Matrix{Complex{Float64}}(I, dim_rbm, dim_rbm) + s))
        catch SingularException
            println("increased λ")
            VarDervs.reg = 2 * VarDervs.reg
            inv_s = inv(Hermitian(VarDervs.reg *
                Matrix{Complex{Float64}}(I, dim_rbm, dim_rbm) + s))
            VarDervs.reg = regulator  # Revert to original in the case λ was raised.
        end
        dw_tot .= -γ * inv_s * f
    end
    return ""
end

"""
    revert_netparams!(NetParams, OptParams, NetSettings)

Revert the network parameters using the previous shift ΔW saved in
OptParams.dw_tot. Run this function after NaNs occur.
"""
@views function revert_netparams!(NetParams, OptParams, NetSettings)
    @unpack a, b, w = NetParams
    @unpack dw_tot = OptParams
    @unpack n, m = NetSettings
    a .-= dw_tot[1 : n]
    b .-= dw_tot[n + 1: n + m]
    w .-= reshape(dw_tot[n + m + 1: end], n, m)
end

"""
    update_netparams!(NetParams, OptParams, NetSettings)

Update the network parameters using the proposed network shift ΔW saved in
OptParams.dw_tot. Run this function after optimization.
"""
@views function update_netparams!(NetParams, OptParams, NetSettings)
    @unpack a, b, w = NetParams
    @unpack dw_tot = OptParams
    @unpack n, m = NetSettings
    a .+= dw_tot[1 : n]
    b .+= dw_tot[n + 1: n + m]
    w .+= reshape(dw_tot[n + m + 1: end], n, m)
end

"""
    init_optim(OptParamStruct)

Takes a `OPTPARAMS` struct as an argument. Consequently initializes the optimization
parameters (WITH S-matrix).
"""
function init_optim(OptParamStruct, NetSettings)
    @unpack dim_rbm = NetSettings
    s = zeros(Complex{Float64}, dim_rbm, dim_rbm)
    f = zeros(Complex{Float64}, dim_rbm)
    dw_tot = zeros(Complex{Float64}, dim_rbm)
    optimparameters_inst = OptParamStruct(s, f, dw_tot, inv_s)
    return optimparameters_inst
end

"""
    init_optim_mfree(OptParamStruct)

Takes an `OPTPARAMSMFREE` struct as argument. Consequently initializes the
optimization parameters (WITHOUT S-matrix).
"""
function init_optim_mfree(OptParamStruct, NetSettings)
    @unpack dim_rbm = NetSettings
    f = zeros(Complex{Float64}, dim_rbm)
    dw_tot = zeros(Complex{Float64}, dim_rbm)
    optparam_inst = OptParamStruct(f, dw_tot)
end


"""
    distributed_o_eloc(forces_w, VarDervs, elocs)

Writes the part of adjoint(O)⋅E that is sampled by worker number `id`
to a SharedArray `forces_w`. See also `mul!`.
"""
function distributed_o_eloc(forces_w, VarDervs, elocs, id)
    mul!(view(forces_w, :, id),
        view(VarDervs.o_tot, :, :, id),
        view(elocs, :, id))
end


"""
    mul!(forces, forces_w, VarDervs, elocs)

Overloaded function for the calculation of
F = (mc_trials * nworkers)^(-1) * adjoint(O) ⋅ Eloc,  sums over all workers.
"""
function LinearAlgebra.mul!(forces, forces_w, VarDervs::VARDERIVS,
    elocs::SharedArray{Complex{Float64}, 2}, NetSettings)
    @unpack mc_trials = NetSettings
    if nworkers() > 30
        @sync for (id, worker) in enumerate(workers())
            @async remotecall_wait(distributed_o_eloc,
                worker,
                forces_w,
                VarDervs,
                elocs,
                id)
            end
    else
        for id=1:nworkers()
            mul!(view(forces_w, :, id),
                view(VarDervs.o_tot, :, :, id),
                view(elocs, :, id))
            end
    end
    forces .= dropdims(sum(forces_w, dims=2), dims=2)
    # overwrites v with adjoint(O) * (O * b)
    lmul!(1 / (mc_trials * nworkers()), forces)
    # overwrites v with scaled v
end
