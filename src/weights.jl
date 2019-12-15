"""
    init_weights(WeightsStruct, modelname, n[, pbc=true])
Generates single site and pair interactions Hamiltonians. Interacts with
`WEIGHTS` type, modelname passed as string.
"""
function init_weights(WeightsStruct, modelname, n, pbc=true)
    if modelname == "afh"
        w_x = zeros(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] = 1; w_yy[i, j] = 1; w_zz[i, j] = 1
        end
        w_xx[1, n] = 1; w_yy[1, n] = 1; w_zz[1, n] = 1
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "U_afh"  # Marshall-Peierls Heisenberg model
        w_x = zeros(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] = -1; w_yy[i, j] = -1; w_zz[i, j] = 1
        end
        w_xx[1, n] = -1; w_yy[1, n] = -1; w_zz[1, n] = 1
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "constr_x"
        w_x = -0.5 * ones(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n, n) ; w_yy = zeros(n,n) ; w_zz = zeros(n,n)
        for i=1:n -1
            j = i + 1
            w_zz[i, j] = 0
        end
        w_zz[1, n] = 0
        w_zz += w_zz'

    elseif modelname == "tfi_h1/2"
        w_x = -(1/2)*ones(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n, n) ; w_yy = zeros(n,n) ; w_zz = zeros(n,n)
        for i=1:n -1
            j = i + 1
            w_zz[i, j] = -1
        end
        w_zz[1, n] = - 1
        w_zz += w_zz'

    elseif modelname == "tfi_h1"
        w_x = -ones(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n, n) ; w_yy = zeros(n,n) ; w_zz = zeros(n,n)
        for i=1:n -1
            j = i + 1
            w_zz[i, j] = -1
        end
        w_zz[1, n] = - 1
        w_zz += w_zz'

    elseif modelname == "tfi_h2"
        w_x = -2 * ones(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n, n) ; w_yy = zeros(n,n) ; w_zz = zeros(n,n)
        for i=1:n -1
            j = i + 1
            w_zz[i, j] = -1
        end
        w_zz[1, n] = - 1
        w_zz += w_zz'

    elseif modelname == "afh_conn"
        w_x = zeros(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = -ones(n,n) ; w_yy = -ones(n,n) ; w_zz = -ones(n,n)
        w_xx -= Diagonal(w_xx); w_yy -= Diagonal(w_yy); w_zz -= Diagonal(w_zz)

    elseif modelname == "afh_ext_y"
        w_x = zeros(n); w_y = -ones(n); w_z = zeros(n)
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] = -1; w_yy[i, j] = -1; w_zz[i, j] = -1
        end
        w_xx[1, n] = -1; w_yy[1, n] = -1; w_zz[1, n] = -1
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "afh_xyz"
        w_x = zeros(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] = -0.2; w_yy[i, j] = -0.6; w_zz[i, j] = -0.4
        end
        w_xx[1, n] = -0.2; w_yy[1, n] = -0.6; w_zz[1, n] = -0.4
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "afh_xyz_ext"
        w_x = rand(Normal(0, 1/sqrt(n)), n); w_y = rand(Normal(0, 1/sqrt(n)), n); w_z = rand(Normal(0, 1/sqrt(n)), n)
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] = -0.2; w_yy[i, j] = -0.6; w_zz[i, j] = -0.4
        end
        w_xx[1, n] = -0.2; w_yy[1, n] = -0.6; w_zz[1, n] = -0.4
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "afh_noisy"
        w_x = 0.01 * rand(n); w_y = 0.01 * rand(n); w_z = 0.01 * rand(n)
        w_xx = 0.01 * rand(n,n) ; w_yy = 0.01 * rand(n,n); w_zz = 0.01 * rand(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] -= 1; w_yy[i, j] -= 1; w_zz[i, j] -= 1
        end
        w_xx[1, n] -= 1; w_yy[1, n] -= 1; w_zz[1, n] -= 1
        w_xx = (w_xx + transpose(w_xx)) / 2; w_yy = (w_yy + transpose(w_yy)) / 2; w_zz = (w_zz + transpose(w_zz)) / 2
        w_xx -= Diagonal(w_xx); w_yy -= Diagonal(w_yy); w_zz -= Diagonal(w_zz)

    elseif modelname == "random_normal"
        w_x = rand(Normal(0,1), n); w_y = rand(Normal(0, 1), n); w_z = rand(Normal(0, 1), n)
        w_xx = rand(Normal(0, 1/sqrt(n)), n, n); w_yy = rand(Normal(0, 1/sqrt(n)), n, n); w_zz = rand(Normal(0, 1/sqrt(n)), n, n)
        w_xx = (w_xx + transpose(w_xx)) / 2; w_yy = (w_yy + transpose(w_yy)) / 2; w_zz = (w_zz + transpose(w_zz)) / 2
        w_xx -= Diagonal(w_xx); w_yy -= Diagonal(w_yy); w_zz -= Diagonal(w_zz)

    elseif modelname == "random_normal_ext0_symm"
        w_x = zeros(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = w_yy = w_zz = rand(Normal(0, 1e-5), n, n)
        w_xx = (w_xx + transpose(w_xx)) / 2; w_yy = (w_yy + transpose(w_yy)) / 2; w_zz = (w_zz + transpose(w_zz)) / 2

    elseif modelname == "random"
        w_x = rand(n); w_y = rand(n); w_z = rand(n)
        w_xx = rand(n, n); w_yy = rand(n, n); w_zz = rand(n, n)
        w_xx = (w_xx + transpose(w_xx)) / 2; w_yy = (w_yy + transpose(w_yy)) / 2; w_zz = (w_zz + transpose(w_zz)) / 2
        w_xx -= Diagonal(w_xx); w_yy -= Diagonal(w_yy); w_zz -= Diagonal(w_zz)

    elseif modelname == "random_ext0"
        w_x = zeros(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = rand(n, n); w_yy = rand(n, n); w_zz = rand(n, n)
        w_xx = (w_xx + transpose(w_xx)) / 2; w_yy = (w_yy + transpose(w_yy)) / 2; w_zz = (w_zz + transpose(w_zz)) / 2
        w_xx -= Diagonal(w_xx); w_yy -= Diagonal(w_yy); w_zz -= Diagonal(w_zz)

    else
        error("no model with that name")
    end

    if ~pbc && n>2
        w_xx[1, n] = w_yy[1, n] = w_zz[1, n] = 0
        w_xx[n, 1] = w_yy[n, 1] = w_zz[n, 1] = 0
    end

    weights = WeightsStruct(w_x, w_y, w_z, w_xx, w_yy, w_zz)
    return weights
end

"""
    Base.+(W1::WEIGHTS, W2::WEIGHTS)

Define additions of weight-structs for QBM learning rule.
"""
function Base.:+(W1::WEIGHTS, W2::WEIGHTS)
    SumW = init_weights(WEIGHTS, "zeros", size(W1.w_x, 1))
    for k in fieldnames(WEIGHTS)
        W1_tmp = getfield(W1, k)
        W2_tmp = getfield(W2, k)
        setfield!(SumW, k, W1_tmp + W2_tmp)
    end
    return SumW
end

"""
    Base.-(W1::WEIGHTS, W2::WEIGHTS)

Define additions of weight-structs for QBM learning rule.
"""
function Base.:-(W1::WEIGHTS, W2::WEIGHTS)
    subtrW = init_weights(WEIGHTS, "zeros", size(W1.w_x, 1))
    for k in fieldnames(WEIGHTS)
        W1_tmp = getfield(W1, k)
        W2_tmp = getfield(W2, k)
        setfield!(subtrW, k, W1_tmp - W2_tmp)
    end
    return subtrW
end

"""
    Base.*(x::Float64, W2::WEIGHTS)

Define multipliciation of weight-structs by a scalar for QBM learning rule.
"""
function Base.:*(x::Float64, W::WEIGHTS)
    timesW = init_weights(WEIGHTS, "zeros", size(W.w_x, 1))
    for k in fieldnames(WEIGHTS)
        timesW_tmp = x * getfield(W, k)
        setfield!(timesW, k, timesW_tmp)
    end
    return timesW
end
