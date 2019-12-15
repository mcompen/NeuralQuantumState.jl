"""
Runs the Carleo & Troyer (2017) algorithm.
"""

function main(NetSettings)
    @unpack mfree, mc_trials, dim_rbm, n, repetitions, calc_stat, γ_init, γ_decay,
    save_figures, modelname, use_meter = NetSettings

    HamiltWeights = init_weights(WEIGHTS, modelname, n)
    NetParams = init_params(NETPARAMS, NetSettings)
    CGParams = init_cg_params(CGPARAMS, NetSettings)
    VarDervs = init_vardervs(VARDERIVS, NetSettings)

    if mfree
        OptParams = init_optim_mfree(OPTPARAMSMFREE, NetSettings)
    else
        OptParams = init_optim(OPTPARAMS, NetSettings)
    end

    elocs = SharedArray(zeros(Complex{Float64}, mc_trials, nworkers()))
    # Stores the energy samples of each worker in a column.
    forces_w = SharedArray(zeros(Complex{Float64}, dim_rbm, nworkers()))
    # Stores the forces of each worker in a column.
    warm_states = zeros(Int, n, nworkers())
    # Saves last visited state in a chain.
    mean_e = zeros(repetitions)
    γ = γ_init

    progress_gui = Progress(repetitions)  # Progress bar
    for i=1:repetitions
        @sync for (id, worker) in enumerate(workers())
            @async warm_states[:, id] = remotecall_fetch(mc_sampling!,
                worker,
                NetParams,
                VarDervs.o_tot,
                elocs,
                HamiltWeights,
                NetSettings,
                id,
                warm_state=warm_states[:, id]
                )
            end

        mean_e[i] = mean(real(elocs))
        cg_iters = optimize!(CGParams, VarDervs, OptParams, elocs, forces_w, γ, NetSettings)
        update_netparams!(NetParams, OptParams, NetSettings)
        γ *= γ_decay
        if use_meter
            ProgressMeter.next!(progress_gui; showvalues = [(:iter,i),
                (:E,mean_e[i]), (:CG_iters, cg_iters)])
        end
    end
    @everywhere GC.gc()
    eloc_and_stats = [0.0, init_stat(STATISTICS, n)]
    if calc_stat
        eloc_and_stats = stats_calculator(NetParams, HamiltWeights, warm_states, NetSettings)
    end
    return mean_e, NetParams, eloc_and_stats[1], eloc_and_stats[2]
end

"""
    run_NQS(NetSettings)

# Arguments
    - `NetSettings`: The settings struct containing all possible network settings.
"""
function run_NQS(NetSettings)
    @unpack modelname, n, test_iter, calc_stat, α, mag0, writetofile, save_figures = NetSettings
    init_print(NetSettings)

    energy = zeros(Float64, 0)
    for i=1:test_iter
        println("RUN $i")
        energy, NetParams, final_eloc, GroundStats = main(NetSettings)

        if writetofile
            if calc_stat
                namebase = "$modelname-n=$n-mag0=$mag0-alpha=$α-run=$i-"
                for k in fieldnames(STATISTICS)
                    writedlm(namebase * "statistics-$k.txt", getfield(GroundStats, k))
                end
                writedlm(namebase * "mc_and_final_eloc.txt", [energy; final_eloc])
                if save_figures
                    plot_statistics(GroundStats, namebase, n)
                end
            else
                writedlm(namebase * "mc_eloc.txt", energy)
            end
            for k in fieldnames(NETPARAMS)
                writedlm(namebase * "netparams-$k.txt", getfield(NetParams, k))
            end
        end
    end
    return energy
end

function init_print(NetSettings)
    @unpack n, modelname, pbc, γ_init, γ_decay, regulator, mag0, α, repetitions,
        mc_trials, therm_steps, stat_samples, mfree, iterative_inverse,
        writetofile =  NetSettings
    println("## Initializing $n spin system
## model = $modelname
## pbc = $pbc
## learning rate = $γ_init
## lrate-decay = $γ_decay
## S-matrix diag shift = $regulator
## workers = $(nworkers())
## ∑s_z=0-sector sampling = $mag0
## α = $α
## iterations = $repetitions
## samples per worker = $mc_trials
## thermalization steps = $therm_steps
## final samples per worker = $stat_samples
## matrixfree = $mfree
## conjugate gradient inversion = $iterative_inverse
## saving results = $writetofile")
end
