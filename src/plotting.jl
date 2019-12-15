"""
Plotting spin statistics and Hamiltonian weights. Statistics of local fields and
w_i^k are plotted rather ugly. Contains also functions for plotting
wavefunctions, and learning rates and likelihoods as a function of QBM
Witeration.
"""

function plot_weights(Weights, modelname, n)
    all_weights = []
    for k in fieldnames(WEIGHTS)
        weights_k = transpose(getfield(Weights, k))
        weights_k[abs.(weights_k) .< 1e-6] .= 0
        push!(all_weights, weights_k)
    end
    posneg_col = [:pu_or; (-absmax(Weights), +absmax(Weights))]
    pos_col = [:inferno; (0, +absmax(Weights))]
    check_col(weights) = any(weights .< 0) ? posneg_col : pos_col
    heatmap_plots = []
    aspect_ratio = [repeat([n], 3); repeat([1], 3)]

    for i=1:6
        p = plot( heatmap(all_weights[i], aspect_ratio=aspect_ratio[i],
            framestyle=:none, color=check_col(all_weights[i])[1]),
            size=(600,600),
            clims= check_col(all_weights[i])[2] )
        push!(heatmap_plots, p)
    end

    p_k = plot(heatmap_plots[1:3]..., layout=(1,3))
    p_kk = plot(heatmap_plots[4:6]..., layout=(1,3))

    plot(p_k, p_kk, layout=(2,1))
    savefig("$modelname-$n-weights-all.pdf")
end

function plot_statistics(Statistics, namebase, n)
    all_stats = []
    for k in fieldnames(STATISTICS)
        modelstats = transpose(getfield(Statistics, k))
        modelstats[abs.(modelstats) .< 1e-6] .= 0.0
        push!(all_stats, modelstats)
    end
    posneg_col = [:pu_or; (-1.0, 1.0)]
    pos_col = [:inferno; (0.0, 1.0)]
    check_col(stats) = any(stats .< 0) ? posneg_col : pos_col
    heatmap_plots = []
    aspect_ratio = [repeat([0.5 * n], 3); repeat([1], 3)]

    for i=1:6
        p = plot( heatmap(all_stats[i], aspect_ratio=aspect_ratio[i],
            framestyle=:none, color=check_col(all_stats[i])[1]),
            size=(600,600),
            clims= check_col(all_stats[i])[2] )
        push!(heatmap_plots, p)
    end

    p_k = plot(heatmap_plots[1:3]..., layout=(1,3))
    p_kk = plot(heatmap_plots[4:6]..., layout=(1,3))

    plot(p_k, p_kk, layout=(2,1))
    savefig("$namebase-stats-all.pdf")
end
