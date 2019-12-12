"""
    init_cg_params(CGParamStruct, NetSettings)

Takes a Conjugate Gradient Parameter struct as argument, Initialize the Conjugate
Gradient Parameters.
"""
function init_cg_params(CGParamStruct, NetSettings)
    @unpack dim_rbm = NetSettings
    r = zeros(Complex{Float64}, dim_rbm)
    dir = SharedArray(zeros(Complex{Float64}, dim_rbm))
    dw = zeros(Complex{Float64}, dim_rbm)
    Sdir_w = SharedArray(zeros(Complex{Float64}, dim_rbm, nworkers()))
    Sdir = zeros(Complex{Float64}, dim_rbm)
    res = 0.0
    res_prev = 1.0
    return CGParamStruct(r, dir, dw, Sdir_w, Sdir, res, res_prev)
end

"""
    distributed_o_dir!(Sdir_w, VarDervs, dir)

Multiply the part of adjoint(O)*O*dir that is sampled by processor `myid() - 1`.
Views are used to prevent allocation-expensive copies upon slicing. `Sdir_w` is
updated in-place.
"""
function distributed_o_dir!(Sdir_w, VarDervs, dir)
    @fastmath mul!(view(Sdir_w, :, myid() - 1),  # view to ensure writing to Sdir-slice, instead of a copy
        view(VarDervs.o_tot, :, :, myid() - 1),
        BLAS.gemv('C', view(VarDervs.o_tot, :, :, myid() - 1), dir))
end


"""
    LinearAlgebra.mul!(Sdir, Sdir_w, VarDervs, dir)

Definition of 'VarDervs * dir' multiplication, so that calculation of entire
S-matrix is prevented: S * dir = adjoint(OptParams.O) * [OptParams.O * dir], see
(Carleo, Troyer, 2017). Note that the VARDERIVS type stores adjoint(O) instead
of O itself. This function is called directly from `my_cg!` for calculating
matrix-vector products. This is done in parallel, after which the parallel
results are summed and scaled. The result is stored in-place in `Sdir`.
See also `init_vardervs` and `my_cg!`.
"""
function LinearAlgebra.mul!(Sdir::Array{Complex{Float64}, 1},
        Sdir_w::SharedArray{Complex{Float64}, 2},
        VarDervs::VARDERIVS,
        dir::SharedArray{Complex{Float64}, 1},
        NetSettings)
    @unpack mc_trials = NetSettings
    if nworkers() > 30
        @sync for i=1:nworkers()
            @async remotecall_wait(distributed_o_dir!,
                workers()[i],
                Sdir_w,
                VarDervs,
                dir)
        end
    else
        for i=1:nworkers()
            @fastmath mul!(view(Sdir_w, :, i),  # view to ensure writing to Sdir-slice, instead of a copy
                view(VarDervs.o_tot, :, :, i),
                BLAS.gemv('C', view(VarDervs.o_tot, :, :, i), dir))
        end
    end
    Sdir .= dropdims(sum(Sdir_w, dims=2), dims=2)
    # overwrites Sdir with adjoint(O) * (O * dir)
    lmul!(1 / (mc_trials * nworkers()), Sdir)
    # overwrites Sdir with scaled Sdir
    axpy!(VarDervs.reg, dir, Sdir)
    # overwrites Sdir with regularized Sdir.
end

""" --- NOT USED ---
    diag_precond!(VarDervs, CGP, NetSettings)

Takes the variational derivatives and calculates only the diagonal of the
S-matrix to be used for preconditioning in the CG algorithm.
See also the matrix-free version of `my_cg!`.

"""
function diag_precond!(VarDervs::VARDERIVS, CGP::CGPARAMS, NetSettings)
    @unpack dim_rbm, mc_trials = NetSettings
    for i=1:dim_rbm
        for j=1:nworkers()
            CGP.m[i] += norm(view(VarDervs.o_tot, i, :, j))^2
        end
    end
    CGP.m ./= (mc_trials * nworkers())
end

"""
    my_cg!(CGP, VarDervs, OptParams, lrate, NetSettings[, tol] )

Calculates the change in network parameters iteratively by conjugate gradient.
Matrix-vector products are calculated without an explictly built S-matrix.
`OptParams.dw` is updated in-place.
"""
function my_cg!(CGP::CGPARAMS, VarDervs::VARDERIVS, OptParams::OPTPARAMSMFREE,
    lrate::Float64, NetSettings; tol=1e-3)
    @unpack dim_rbm = NetSettings
    i_max = dim_rbm

    fill!(OptParams.dw_tot, zero(eltype(OptParams.dw_tot)))
    fill!(CGP.dir, zero(eltype(CGP.dir)))
    CGP.res_prev = 1.0

    CGP.r .= -lrate .* OptParams.f
    res_0 = norm(CGP.r)
    CGP.res = res_0

    i = 0
    while i < i_max && CGP.res / res_0 > tol
        β = CGP.res^2 / CGP.res_prev^2
        CGP.dir .= CGP.r .+ β .* CGP.dir
        mul!(CGP.Sdir, CGP.Sdir_w, VarDervs, CGP.dir, NetSettings)
        α = CGP.res^2 / dot(CGP.dir, CGP.Sdir)
        OptParams.dw_tot .+= α .* CGP.dir
        CGP.r .-= α .* CGP.Sdir
        CGP.res_prev = CGP.res
        CGP.res = norm(CGP.r)
        i += 1
    end
    return i
end

"""
    my_cg!(CGP, VarDervs, OptParams, lrate, NetSettings[, tol] )

Calculates the change in network parameters iteratively by conjugate gradient.
Matrix-vector products are calculated with the explicitly built S-matrix.
`OptParams.dw` is updated in-place.
"""
function my_cg!(CGP::CGPARAMS,
        S::Array{Complex{Float64}, 2},
        OptParams::OPTPARAMS,
        lrate::Float64,
        NetSettings;
        tol=1e-3)
    @unpack dim_rbm = NetSettings
    i_max = dim_rbm

    fill!(OptParams.dw_tot, zero(eltype(OptParams.dw_tot)))
    fill!(CGP.dir, zero(eltype(CGP.dir)))
    CGP.res_prev = 1.0

    CGP.r .= -lrate .* OptParams.f
    res_0 = norm(CGP.r)
    CGP.res = res_0

    i = 0
    while i < i_max && (CGP.res / res_0) > tol
        β = CGP.res^2 / CGP.res_prev^2
        CGP.dir .= CGP.r .+ β .* CGP.dir
        mul!(CGP.Sdir, S, CGP.dir)
        α = CGP.res^2 / dot(CGP.dir, CGP.Sdir)
        OptParams.dw_tot .+= α .* CGP.dir
        CGP.r .-= α .* CGP.Sdir
        CGP.res_prev = CGP.res
        CGP.res = norm(CGP.r)
        i += 1
    end
end

# Supporting Base functions for VARDERIVS type.
Base.size(VarDervs::VARDERIVS) = (size(VarDervs.o_tot, 1), size(VarDervs.o_tot, 1)) # should return size of S
Base.size(VarDervs::VARDERIVS, i::Int) = size(VarDervs.o_tot, 1)  # S is square
Base.eltype(VarDervs::VARDERIVS) = eltype(VarDervs.o_tot)
