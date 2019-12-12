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
    elseif modelname == "random_y0"
        w_x = rand(n); w_y = zeros(n); w_z = rand(n)
        w_xx = rand(n, n); w_yy = rand(n, n); w_zz = rand(n, n)
        w_xx = (w_xx + transpose(w_xx)) / 2; w_yy = (w_yy + transpose(w_yy)) / 2; w_zz = (w_zz + transpose(w_zz)) / 2
        w_xx -= Diagonal(w_xx); w_yy -= Diagonal(w_yy); w_zz -= Diagonal(w_zz)

    elseif modelname == "ones"
        w_x = ones(n); w_y = ones(n); w_z = ones(n)
        w_xx = ones(n, n); w_yy = ones(n, n); w_zz = ones(n, n)
        w_xx -= Diagonal(w_xx); w_yy -= Diagonal(w_yy); w_zz -= Diagonal(w_zz)

    elseif modelname == "constr_test"
        w_x = zeros(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] = -0.3; w_yy[i, j] = 0.2; w_zz[i, j] = 0.1
        end
        w_xx[1, n] = -0.3; w_yy[1, n] = 0.2; w_zz[1, n] = 0.1
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "constr_test2"
        w_x = -0.5 * ones(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] = -0.9; w_yy[i, j] = -0.1; w_zz[i, j] = 0.3
        end
        w_xx[1, n] = -0.9; w_yy[1, n] = -0.1; w_zz[1, n] = 0.3
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "constr_test3"
        w_x = -abs.(rand(Normal(0, 1e-6), n)); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] = -0.2; w_yy[i, j] = -0.1; w_zz[i, j] = 0.9
        end
        w_xx[1, n] = -0.2; w_yy[1, n] = -0.1; w_zz[1, n] = 0.9
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "constr_connected"
        w_x = -0.05 * Array(1:n); w_y = zeros(n); w_z = 0.025 * Array(1:n)
        w_xx = -(0.5 / n^2) * reshape(Array(1:n^2), n, n); w_yy = 0.25 .* w_xx; w_zz = (-0.8 / n^2) .* reshape(reverse(Array(1:n^2)), n, n)
        w_xx  = Matrix(Symmetric(w_xx, :L)); w_yy = Matrix(Symmetric(w_yy, :L)); w_zz = Matrix(Symmetric(w_zz, :L))
        w_xx .-= Diagonal(w_xx); w_yy .-= Diagonal(w_yy); w_zz .-= Diagonal(w_zz)

    elseif modelname == "constr_connected_frustr"
        w_x = -0.05 * Array(1:n); w_y = zeros(n); w_z = 0.025 * Array(1:n)
        w_xx = -(0.5 / n^2) * reshape(Array(1:n^2), n, n); w_yy = 0.25 .* w_xx; w_zz = (-0.8 / n^2) .* reshape(reverse(deleteat!(Array(-0.5 * n^2: 0.5 * n^2), Int(0.5 * n^2))), n, n)
        w_xx  = Matrix(Symmetric(w_xx, :L)); w_yy = Matrix(Symmetric(w_yy, :L)); w_zz = Matrix(Symmetric(w_zz, :L))
        w_xx .-= Diagonal(w_xx); w_yy .-= Diagonal(w_yy); w_zz .-= Diagonal(w_zz)

    elseif modelname == "constr_connected_frustr2"
        w_x = -0.05 * Array(1:n); w_y = zeros(n); w_z = 0.025 * deleteat!(Array(-0.5*n : 0.5*n), Int(0.5*n))
        w_xx = -(0.5 / n^2) * reshape(Array(1:n^2), n, n); w_yy = 0.25 .* w_xx; w_zz = (-0.8 / n^2) .* reshape(reverse(Array(1:n^2)), n, n)
        for i=1:n
            w_zz[i, :] *= (-1)^i
        end
        w_xx  = Matrix(Symmetric(w_xx, :L)); w_yy = Matrix(Symmetric(w_yy, :L)); w_zz = Matrix(Symmetric(w_zz, :L))
        w_xx .-= Diagonal(w_xx); w_yy .-= Diagonal(w_yy); w_zz .-= Diagonal(w_zz)

    elseif modelname == "constr_connected2"
        w_x = -0.1 * Array(1:n); w_y = zeros(n); w_z = 0.2 * Array(1:n)
        w_xx = -(0.6 / n^2) * reshape(Array(1:n^2), n, n); w_yy = -0.25 * w_xx; w_zz = (-0.05 / n^2) .* reshape(reverse(Array(1:n^2)), n, n)
        w_xx  = Matrix(Symmetric(w_xx, :L)); w_yy = Matrix(Symmetric(w_yy, :L)); w_zz = Matrix(Symmetric(w_zz, :L))
        w_xx .-= Diagonal(w_xx); w_yy .-= Diagonal(w_yy); w_zz .-= Diagonal(w_zz)

    elseif modelname == "constr_therm"
        w_x = -abs.(rand(Normal(0, 1e-6), n)); w_y = zeros(n); w_z = abs.(rand(Normal(0, 1e-6), n))
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_yy[i, j] = rand(Normal(0, 1e-6)); w_zz[i, j] = rand(Normal(0, 1e-6))
            w_xx[i, j] = rand(Uniform(-abs(w_yy[i, j]) - 1e-5, -abs(w_yy[i, j])))
        end
        w_yy[1, n] = rand(Normal(0, 1e-6)); w_zz[1, n] = rand(Normal(0, 1e-6))
        w_xx[1, n] = rand(Uniform(-abs(w_yy[1, n]) - 1e-5, -abs(w_yy[1, n])))
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "zeros"
        w_x = w_y = w_z = zeros(n);
        w_xx = w_yy = w_zz = zeros(n,n);
    elseif modelname == "small_stoq"
        w_x = -1/sqrt(n) .* abs.(rand(Normal(0, 1e-3), n))
        w_y = zeros(n)
        w_z = 1/sqrt(n) .* rand(Normal(0, 1e-3), n)
        w_zz = 1/sqrt(n) .* rand(Normal(0, 1e-3), n, n)
        w_yy = 1/sqrt(n) .* rand(Normal(0, 1e-3), n, n)
        w_xx = -abs.(w_yy) - abs.(1/sqrt(n) .* rand(Normal(0, 1e-3), n, n))
        w_xx .-= Diagonal(w_xx); w_yy .-= Diagonal(w_yy); w_zz .-= Diagonal(w_zz)
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    elseif modelname == "stoquastic"
        w_x = -1.0*ones(n);
        w_xx = -1.0*(ones(n, n) - I);
        w_z = w_y = zeros(n);
        w_zz = w_yy = zeros(n, n);

    elseif modelname == "close_to_zero"
        w_x = -0.1*ones(n);
        w_xx = -0.1*(ones(n, n) - I);
        w_z = w_y = zeros(n);
        w_zz = w_yy = zeros(n, n);

    elseif modelname == "NN"
        w_x = -1*ones(n); w_y = zeros(n); w_z = zeros(n)
        w_xx = zeros(n,n) ; w_yy = zeros(n,n); w_zz = zeros(n,n)
        for i=1:n - 1
            j=i + 1
            w_xx[i, j] = 1; w_yy[i, j] = 0.1; w_zz[i, j] = 1
        end
        w_xx[1, n] = 1; w_yy[1, n] = 0.1; w_zz[1, n] = 1
        w_xx += w_xx'; w_yy += w_yy'; w_zz += w_zz'

    # Triangular weights provide the classic triangular frustrated spin model.
    elseif modelname =="triangular"
        if mod(sqrt(n/2), 1) != 0
            # n must be of the form 2*i^2 for i integer.
            print("Not possible to create uniform triangular lattice with given dimensions")
        else
            w_x = w_y = zeros(n)
            w_xx = w_yy = zeros(n, n)
            w_z = (0.20/exp(n))*[exp(i) for i in 1:n] # ranges exponentially up to 0.2 to break degeneracy
            w_zz = zeros(n, n)
            l1 = 1:Int(n/2)
            l2 = Int(n/2)+1:n
            # two scaling constants of the lattice
            a = Int(sqrt(n/2))
            b = Int(n/2)
            # Horizontal connections
            for i in l1
                if mod(i, a) != 0
                    w_zz[i, i+1] = 1
                    w_zz[i+b, i+b+1] = 1
                else
                    w_zz[i, i-a+1] = 1
                    w_zz[i+b, i+b-a+1] = 1
                end
            end
            #diagonal connections
            for i in l2
                w_zz[i, i-b] = 1 # connection up-left
                upright = i-b+1 # connection up-right
                if mod(upright, a) != 1
                    w_zz[i, upright] = 1
                else
                    w_zz[i, upright-a] = 1
                end
                downleft = i-b+a
                if downleft > b
                    downleft -= b
                end
                w_zz[i, downleft] = 1 # connection down-left
                downright = i-b+a+1
                if mod(downright,a) == 1
                    downright -= a
                end
                if downright > b
                    downright -= b
                end
                w_zz[i, downright] = 1
            end
            w_zz += w_zz'
        end
    else
        println("No modelname with that name")
    end

    if ~pbc && n>2
        w_xx[1, n] = w_yy[1, n] = w_zz[1, n] = 0
        w_xx[n, 1] = w_yy[n, 1] = w_zz[n, 1] = 0
    end

    weights = WeightsStruct(w_x, w_y, w_z, w_xx, w_yy, w_zz)
    return weights
end

function absmax(W1::WEIGHTS)
    max = 0
    for k in fieldnames(WEIGHTS)
        max_k = maximum(abs.(getfield(W1, k)))
        max_k > max ? max = max_k : nothing
    end
    return max
end

function Base.abs(W1::WEIGHTS)
    absW = init_weights(WEIGHTS, "zeros", size(W1.w_x, 1))
    for k in fieldnames(WEIGHTS)
        W1_tmp = getfield(W1, k)
        setfield!(absW, k, abs.(W1_tmp))
    end
    return absW
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
