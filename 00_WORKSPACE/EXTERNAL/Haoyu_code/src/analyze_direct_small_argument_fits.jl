using JLD2
using LinearAlgebra
using Plots
using Printf
using Statistics

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function fit_linear_model(
    y::AbstractVector{<:Real},
    columns::AbstractVector{<:AbstractVector{<:Real}},
)
    X = hcat(map(col -> Float64.(col), columns)...)
    yvec = Float64.(y)
    coeffs = X \ yvec
    fitted = X * coeffs
    residuals = yvec - fitted
    abs_rms = sqrt(mean(abs2, residuals))
    rel_rms = abs_rms / max(maximum(abs.(yvec)), eps())
    return (coeffs=coeffs, fitted=fitted, abs_rms=abs_rms, rel_rms=rel_rms)
end

function model_specs(kind::Symbol, xmax::Float64)
    xvar = kind == :k ? "k" : "Omega"
    return [
        (
            key="quadratic",
            label="a $(xvar)^2",
            design=(x -> [x .^ 2]),
        ),
        (
            key="logplusquadratic",
            label="a $(xvar)^2 log($(xvar)max/$(xvar)) + b $(xvar)^2",
            design=(x -> [x .^ 2 .* log.(xmax ./ x), x .^ 2]),
        ),
        (
            key="quartic",
            label="a $(xvar)^2 + b $(xvar)^4",
            design=(x -> [x .^ 2, x .^ 4]),
        ),
    ]
end

function fit_models(x::Vector{Float64}, y::Vector{Float64}, kind::Symbol, xmax::Float64)
    fits = Dict{String, NamedTuple}()
    for spec in model_specs(kind, xmax)
        result = fit_linear_model(y, spec.design(x))
        fits[spec.key] = (
            key=spec.key,
            label=spec.label,
            coeffs=result.coeffs,
            fitted=result.fitted,
            abs_rms=result.abs_rms,
            rel_rms=result.rel_rms,
        )
    end
    return fits
end

function best_model_key(fits::Dict{String, <:NamedTuple})
    return first(sort(collect(keys(fits)); by=key -> fits[key].rel_rms))
end

function load_direct_k_summary(input_dir::AbstractString, Γ::Float64; kfit_max::Float64)
    data = load(joinpath(input_dir, "gamma_$(gamma_token(Γ)).jld2"))
    x = Float64.(data["k_values"])
    y = real.(Γ .* (data["Πk"] .- data["Π00"]))
    sel = findall(<=(kfit_max), x)
    return (Γ=Γ, x=x[sel], y=y[sel], fits=fit_models(x[sel], y[sel], :k, kfit_max))
end

function load_direct_omega_summary(input_dir::AbstractString, Γ::Float64; Ωfit_max::Float64)
    data = load(joinpath(input_dir, "gamma_$(gamma_token(Γ)).jld2"))
    x = Float64.(data["Ω_values"])
    y = real.(Γ .* (data["ΠΩ"] .- data["Π00"]))
    sel = findall(<=(Ωfit_max), x)
    return (Γ=Γ, x=x[sel], y=y[sel], fits=fit_models(x[sel], y[sel], :Ω, Ωfit_max))
end

function write_fit_table(path::AbstractString, summaries::AbstractVector, model_order::Vector{String})
    open(path, "w") do io
        println(io, "\\begin{tabular}{crrr}")
        println(io, "\\toprule")
        println(io, "\$\\Gamma\$ & \\texttt{quadratic} & \\texttt{logplusquadratic} & \\texttt{quartic} \\\\")
        println(io, "\\midrule")
        for summary in summaries
            best = best_model_key(summary.fits)
            entries = String[]
            for key in model_order
                item = @sprintf("%.2f\\%%", 100 * summary.fits[key].rel_rms)
                if key == best
                    item = "\\textbf{$item}"
                end
                push!(entries, item)
            end
            println(io, @sprintf("%.0f", summary.Γ), " & ", join(entries, " & "), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function plot_fit_overlays(
    summaries::AbstractVector,
    xlabel::AbstractString,
    ylabel::AbstractString,
    model_order::Vector{String},
    output_pdf::AbstractString,
    fit_max::Float64,
)
    colors = Dict("quadratic" => :red, "logplusquadratic" => :blue, "quartic" => :green)
    panels = Plots.Plot[]
    for summary in summaries
        plt = plot(
            summary.x,
            summary.y;
            label="data",
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=:log10,
            linewidth=1.8,
            linestyle=:solid,
            marker=:circle,
            markersize=4,
            markerstrokewidth=0,
            color=:black,
            title="Γ = $(round(summary.Γ; digits=3)), fit window ≤ $(round(fit_max; digits=4))",
        )
        for key in model_order
            fit = summary.fits[key]
            plot!(plt, summary.x, fit.fitted; label="$(fit.label), rel RMS = $(round(100 * fit.rel_rms; digits=2))%", linewidth=2.5, color=colors[key])
        end
        push!(panels, plt)
    end
    plt = plot(panels...; layout=(length(panels), 1), size=(980, 340 * length(panels)))
    savefig(plt, output_pdf)
end

function plot_ratio_diagnostic(
    summaries::AbstractVector,
    xlabel::AbstractString,
    ylabel::AbstractString,
    output_pdf::AbstractString,
)
    panels = Plots.Plot[]
    for summary in summaries
        x = summary.x
        plt = plot(
            x,
            summary.y ./ (x .^ 2);
            label="data",
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=:log10,
            linewidth=1.8,
            linestyle=:solid,
            marker=:circle,
            markersize=4,
            markerstrokewidth=0,
            color=:black,
            title="Γ = $(round(summary.Γ; digits=3))",
        )
        plot!(plt, x, summary.fits["quadratic"].fitted ./ (x .^ 2); label="quadratic", linewidth=2.5, color=:red)
        plot!(plt, x, summary.fits["logplusquadratic"].fitted ./ (x .^ 2); label="log+quadratic", linewidth=2.5, color=:blue)
        plot!(plt, x, summary.fits["quartic"].fitted ./ (x .^ 2); label="quadratic+quartic", linewidth=2.5, color=:green)
        push!(panels, plt)
    end
    plt = plot(panels...; layout=(length(panels), 1), size=(980, 340 * length(panels)))
    savefig(plt, output_pdf)
end

function main(;
    k_dir::AbstractString="data/smallk_direct_lines_Nw16384_Nq4096_wmax320",
    Ω_dir::AbstractString="data/smallomega_direct_lines_Nw32768_Nq2048_wmax640",
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    kfit_max::Float64=0.0025,
    Ωfit_max::Float64=0.0015,
    output_prefix::AbstractString="note/direct_small_argument",
)
    resolved_Γs = collect(Float64.(Γs))
    model_order = ["quadratic", "logplusquadratic", "quartic"]

    k_summaries = [load_direct_k_summary(k_dir, Γ; kfit_max=kfit_max) for Γ in resolved_Γs]
    Ω_summaries = [load_direct_omega_summary(Ω_dir, Γ; Ωfit_max=Ωfit_max) for Γ in resolved_Γs]

    plot_fit_overlays(k_summaries, "k", "Re[Γ(Π(0,k)-Π(0,0))]", model_order, "$(output_prefix)_k_fits.pdf", kfit_max)
    plot_ratio_diagnostic(k_summaries, "k", "Re[Γ(Π(0,k)-Π(0,0))] / k^2", "$(output_prefix)_k_ratio.pdf")
    plot_fit_overlays(Ω_summaries, "Omega", "Re[Γ(Π(Ω,0)-Π(0,0))]", model_order, "$(output_prefix)_omega_fits.pdf", Ωfit_max)
    plot_ratio_diagnostic(Ω_summaries, "Omega", "Re[Γ(Π(Ω,0)-Π(0,0))] / Ω^2", "$(output_prefix)_omega_ratio.pdf")

    write_fit_table("note/direct_small_argument_k_fit_table.tex", k_summaries, model_order)
    write_fit_table("note/direct_small_argument_omega_fit_table.tex", Ω_summaries, model_order)

    println("saved $(output_prefix)_k_fits.pdf")
    println("saved $(output_prefix)_k_ratio.pdf")
    println("saved $(output_prefix)_omega_fits.pdf")
    println("saved $(output_prefix)_omega_ratio.pdf")
    println("saved note/direct_small_argument_k_fit_table.tex")
    println("saved note/direct_small_argument_omega_fit_table.tex")

    for (name, summaries) in (("k", k_summaries), ("omega", Ω_summaries))
        println(name, " direction:")
        for summary in summaries
            println("  Γ=", summary.Γ)
            for key in model_order
                fit = summary.fits[key]
                println("    ", key, ": rel_rms=", @sprintf("%.6f", fit.rel_rms), ", coeffs=", fit.coeffs)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
