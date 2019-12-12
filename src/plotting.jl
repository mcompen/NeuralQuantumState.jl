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

    p_x = plot( heatmap(all_weights[1], aspect_ratio=n, framestyle=:none, color=check_col(all_weights[1])[1]),
        xticks=0:2:n,
        xtickfont=font(6),
        yticks=false,
        size=(600,600),
        clims= check_col(all_weights[1])[2]
    )
    p_y = plot( heatmap(all_weights[2], aspect_ratio=n, framestyle=:none, color=check_col(all_weights[2])[1]),
        xticks=0:2:n,
        xtickfont=font(6),
        yticks=false,
        size=(600,600),
        clims=check_col(all_weights[2])[2]
    )
    p_z = plot( heatmap(all_weights[3], aspect_ratio=n, framestyle=:none, color=check_col(all_weights[3])[1]),
        xticks=0:2:n,
        xtickfont=font(6),
        yticks=false,
        size=(600,600),
        clims=check_col(all_weights[3])[2]
    )
    p_xx = plot( heatmap(all_weights[4], aspect_ratio=1, framestyle=:none, color=check_col(all_weights[4])[1]),
        xticks=0:2:n,
        xtickfont=font(6),
        yticks=false,
        size=(600,600),
        clims=check_col(all_weights[4])[2]
    )
    p_yy = plot( heatmap(all_weights[5], aspect_ratio=1, framestyle=:none, color=check_col(all_weights[5])[1]),
        xticks=0:2:n,
        xtickfont=font(6),
        yticks=false,
        size=(600,600),
        clims=check_col(all_weights[5])[2]
    )
    p_zz = plot( heatmap(all_weights[6], aspect_ratio=1, framestyle=:none, color=check_col(all_weights[6])[1]),
        xticks=0:2:n,
        xtickfont=font(6),
        yticks=false,
        size=(600,600),
        clims=check_col(all_weights[6])[2]
    )

    p_k = plot(p_x, p_y, p_z, layout=(1,3))
    p_kk = plot(p_xx, p_yy, p_zz, layout=(1,3))

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

    p_x = plot( heatmap(all_stats[1], aspect_ratio=n, framestyle=:none, color=check_col(all_stats[1])[1]),
        size=(600,600),
        clims= check_col(all_stats[1])[2])
    p_y = plot( heatmap(all_stats[2], aspect_ratio=n, framestyle=:none, color=check_col(all_stats[2])[1]),
        size=(600,600),
        clims=check_col(all_stats[2])[2])
    p_z = plot( heatmap(all_stats[3], aspect_ratio=n, framestyle=:none, color=check_col(all_stats[3])[1]),
        size=(600,600),
        clims=check_col(all_stats[3])[2])
    p_xx = plot( heatmap(all_stats[4], aspect_ratio=1, framestyle=:none, color=check_col(all_stats[4])[1]),
        size=(600,600),
        clims=check_col(all_stats[4])[2])
    p_yy = plot( heatmap(all_stats[5], aspect_ratio=1, framestyle=:none, color=check_col(all_stats[5])[1]),
        size=(600,600),
        clims=check_col(all_stats[5])[2])
    p_zz = plot( heatmap(all_stats[6], aspect_ratio=1, framestyle=:none, color=check_col(all_stats[6])[1]),
        size=(600,600),
        clims=check_col(all_stats[6])[2])

    p_k = plot(p_x, p_y, p_z, layout=(1,3))
    p_kk = plot(p_xx, p_yy, p_zz, layout=(1,3))

    plot(p_k, p_kk, layout=(2,1))
    savefig("$namebase-stats-all.pdf")
end
