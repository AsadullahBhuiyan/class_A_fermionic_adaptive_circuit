using JLD2
using Plots

const BENCHMARK_RESULTS_FILE = "benchmark_convolution_results.jld2"

function plot_convolution_comparison(
    ωs::AbstractVector,
    ks::AbstractVector,
    out_direct,
    out_fft::AbstractMatrix,
    out_tail::AbstractMatrix,
    slice_indices::AbstractVector,
    active_indices::AbstractVector;
    direct_offset_fraction::Float64=0.0,
)
    active_order = sortperm(@view ωs[active_indices])
    ωorder = active_indices[active_order]
    ωplot = ωs[ωorder]
    method_colors = Dict(
        "direct" => :blue,
        "fft" => :red,
        "tail" => :green,
    )
    k_styles = [:solid, :dash, :dot, :dashdot]
    linewidth = 2.5
    direct_linewidth = 3.5
    direct_label = direct_offset_fraction == 0.0 ? "direct" : "direct (+offset)"
    real_scale = max(maximum(abs.(real.(out_fft))), maximum(abs.(real.(out_tail))))
    imag_scale = max(maximum(abs.(imag.(out_fft))), maximum(abs.(imag.(out_tail))))
    direct_real_offset = direct_offset_fraction * (real_scale > 0 ? real_scale : 1.0)
    direct_imag_offset = direct_offset_fraction * (imag_scale > 0 ? imag_scale : 1.0)

    plt_real = plot(title="Real Part", xlabel="ω", ylabel="Re convolution")
    plt_imag = plot(title="Imag Part", xlabel="ω", ylabel="Im convolution")

    for (style_index, kidx) in enumerate(slice_indices)
        linestyle = k_styles[mod1(style_index, length(k_styles))]
        kval = ks[kidx]

        plot!(
            plt_real,
            ωplot,
            real.(out_fft[ωorder, kidx]),
            color=method_colors["fft"],
            linestyle=linestyle,
            linewidth=linewidth,
            label="fft, k=$(round(kval; digits=3))",
        )
        plot!(
            plt_real,
            ωplot,
            real.(out_tail[ωorder, kidx]),
            color=method_colors["tail"],
            linestyle=linestyle,
            linewidth=linewidth,
            label="tail, k=$(round(kval; digits=3))",
        )
        if out_direct !== nothing
            plot!(
                plt_real,
                ωplot,
                real.(out_direct[ωorder, kidx]) .+ direct_real_offset,
                color=method_colors["direct"],
                linestyle=linestyle,
                linewidth=direct_linewidth,
                marker=:circle,
                markersize=3,
                markerstrokewidth=0,
                label="$(direct_label), k=$(round(kval; digits=3))",
            )
        end

        plot!(
            plt_imag,
            ωplot,
            imag.(out_fft[ωorder, kidx]),
            color=method_colors["fft"],
            linestyle=linestyle,
            linewidth=linewidth,
            label="fft, k=$(round(kval; digits=3))",
        )
        plot!(
            plt_imag,
            ωplot,
            imag.(out_tail[ωorder, kidx]),
            color=method_colors["tail"],
            linestyle=linestyle,
            linewidth=linewidth,
            label="tail, k=$(round(kval; digits=3))",
        )
        if out_direct !== nothing
            plot!(
                plt_imag,
                ωplot,
                imag.(out_direct[ωorder, kidx]) .+ direct_imag_offset,
                color=method_colors["direct"],
                linestyle=linestyle,
                linewidth=direct_linewidth,
                marker=:circle,
                markersize=3,
                markerstrokewidth=0,
                label="$(direct_label), k=$(round(kval; digits=3))",
            )
        end
    end

    return plt_real, plt_imag
end

function plot_benchmark_main(;
    results_file::AbstractString=BENCHMARK_RESULTS_FILE,
    real_png::AbstractString="benchmark_convolution_real.png",
    imag_png::AbstractString="benchmark_convolution_imag.png",
    direct_offset_fraction::Float64=0.00,
)
    results = load(results_file)
    ωs = results["ωs"]
    ks = results["ks"]
    out_direct = results["out_direct"]
    out_fft = results["out_fft"]
    out_tail = results["out_tail"]
    kslice_indices = results["kslice_indices"]
    params = results["params"]
    active_indices = params.active_indices

    plt_real, plt_imag = plot_convolution_comparison(
        ωs,
        ks,
        out_direct,
        out_fft,
        out_tail,
        kslice_indices,
        active_indices,
        direct_offset_fraction=direct_offset_fraction,
    )

    # savefig(plt_real, real_png)
    # savefig(plt_imag, imag_png)
    display(plt_real)
    display(plt_imag)

    return plt_real, plt_imag
end

main(; kwargs...) = plot_benchmark_main(; kwargs...)

running_in_vscode_repl() = isinteractive() && isdefined(Main, :VSCodeServer)

if abspath(PROGRAM_FILE) == (@__FILE__) || running_in_vscode_repl()
    Base.invokelatest(plot_benchmark_main)
end
