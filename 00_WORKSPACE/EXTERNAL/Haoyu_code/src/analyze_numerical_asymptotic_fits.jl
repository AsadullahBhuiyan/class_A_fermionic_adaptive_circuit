using JLD2
using LinearAlgebra
using Plots
using Printf
using Statistics

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function fit_linear_model(
    x::AbstractVector{<:Real},
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
    return (
        coeffs=coeffs,
        fitted=fitted,
        residuals=residuals,
        abs_rms=abs_rms,
        rel_rms=rel_rms,
    )
end

function evenized_line(
    x_values::AbstractVector{<:Real},
    line_values::AbstractVector{ComplexF64},
    Γ::Real;
    xmax::Float64,
    xzero::Float64=0.0,
)
    zero_index = findfirst(==(xzero), x_values)
    @assert !isnothing(zero_index) "zero must lie on the sampled line"
    base = line_values[zero_index]

    x_positive = Float64[]
    y_even = Float64[]
    for (idx, x) in pairs(x_values)
        xf = Float64(x)
        if 0 < xf <= xmax
            mirror_index = findfirst(y -> isapprox(y, -xf; atol=1e-12), x_values)
            @assert !isnothing(mirror_index) "missing negative partner for x=$(xf)"
            value = 0.5 * real(Γ * (line_values[idx] - base) + Γ * (line_values[mirror_index] - base))
            push!(x_positive, xf)
            push!(y_even, value)
        end
    end

    order = sortperm(x_positive)
    return x_positive[order], y_even[order]
end

function load_k_direction_line(
    input_dir::AbstractString,
    Γ::Real;
    kmax::Float64,
)
    data = load(joinpath(input_dir, "gamma_$(gamma_token(Γ)).jld2"))
    return evenized_line(data["k_values"], data["kline"], Γ; xmax=kmax)
end

function load_omega_direction_line(
    input_dir::AbstractString,
    Γ::Real;
    Ωmax::Float64,
)
    data = load(joinpath(input_dir, "gamma_$(gamma_token(Γ)).jld2"))
    Ω_values = data["Ω_values"]
    line = vec(data["conv_small"][:, 1])
    return evenized_line(Ω_values, line, Γ; xmax=Ωmax)
end

function model_specs(kind::Symbol, xmax::Float64)
    if kind == :k
        xvar = "k"
    elseif kind == :Ω
        xvar = "\\Omega"
    else
        error("unknown kind $kind")
    end

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

function fit_candidate_models(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, kind::Symbol, xmax::Float64)
    specs = model_specs(kind, xmax)
    fits = Dict{String, NamedTuple}()
    for spec in specs
        columns = spec.design(Float64.(x))
        result = fit_linear_model(x, y, columns)
        fits[spec.key] = (
            key=spec.key,
            label=spec.label,
            coeffs=result.coeffs,
            fitted=result.fitted,
            residuals=result.residuals,
            abs_rms=result.abs_rms,
            rel_rms=result.rel_rms,
        )
    end
    return fits
end

function best_model_key(fits::Dict{String, <:NamedTuple})
    ordered = sort(collect(keys(fits)); by=key -> fits[key].rel_rms)
    return first(ordered)
end

function format_coeff_vector(coeffs::AbstractVector{<:Real})
    return join([@sprintf("%.5e", c) for c in coeffs], ", ")
end

function write_fit_table(
    path::AbstractString,
    summaries::Vector{NamedTuple},
    model_order::Vector{String},
    xlabel::AbstractString,
)
    open(path, "w") do io
        println(io, "\\begin{tabular}{c", repeat("r", length(model_order)), "}")
        println(io, "\\toprule")
        println(io, "\$\\Gamma\$ & ", join(model_order .|> (m -> "\\texttt{$m}"), " & "), " \\\\")
        println(io, "\\midrule")
        for summary in summaries
            best = best_model_key(summary.fits)
            cells = String[]
            for key in model_order
                value = 100 * summary.fits[key].rel_rms
                entry = @sprintf("%.2f\\%%", value)
                if key == best
                    entry = "\\textbf{$entry}"
                end
                push!(cells, entry)
            end
            println(io, @sprintf("%.0f", summary.Γ), " & ", join(cells, " & "), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io)
        println(io, "% ", xlabel, " fits in the chosen small-argument window. Entries are relative RMS residuals.")
    end
end

function plot_fit_overlays(
    summaries::Vector{NamedTuple},
    model_order::Vector{String},
    xlabel::AbstractString,
    ylabel::AbstractString;
    output_pdf::AbstractString,
    fit_max::Float64,
)
    panels = Plots.Plot[]
    colors = Dict(
        "quadratic" => :red,
        "logplusquadratic" => :blue,
        "quartic" => :green,
    )

    for summary in summaries
        x = summary.x
        y = summary.y
        plt = scatter(
            x,
            y;
            label="data",
            xlabel=xlabel,
            ylabel=ylabel,
            title="Γ = $(round(summary.Γ; digits=3)), fit window ≤ $(round(fit_max; digits=3))",
            markersize=4,
            markerstrokewidth=0,
            color=:black,
        )
        for key in model_order
            fit = summary.fits[key]
            plot!(plt, x, fit.fitted; label="$(fit.label), rel RMS = $(round(100 * fit.rel_rms; digits=2))%", linewidth=2.5, color=colors[key])
        end
        push!(panels, plt)
    end

    rows = length(panels)
    plt = plot(panels...; layout=(rows, 1), size=(980, 340 * rows))
    savefig(plt, output_pdf)
end

function plot_ratio_diagnostic(
    summaries::Vector{NamedTuple},
    xlabel::AbstractString,
    ylabel::AbstractString;
    output_pdf::AbstractString,
)
    panels = Plots.Plot[]
    for summary in summaries
        x = summary.x
        yratio = summary.y ./ (x .^ 2)
        quad = summary.fits["quadratic"]
        logfit = summary.fits["logplusquadratic"]
        quartic = summary.fits["quartic"]

        plt = scatter(
            x,
            yratio;
            label="data",
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=:log10,
            title="Γ = $(round(summary.Γ; digits=3))",
            markersize=4,
            markerstrokewidth=0,
            color=:black,
        )
        plot!(plt, x, quad.fitted ./ (x .^ 2); label="quadratic", linewidth=2.5, color=:red)
        plot!(plt, x, logfit.fitted ./ (x .^ 2); label="log+quadratic", linewidth=2.5, color=:blue)
        plot!(plt, x, quartic.fitted ./ (x .^ 2); label="quadratic+quartic", linewidth=2.5, color=:green)
        push!(panels, plt)
    end

    rows = length(panels)
    plt = plot(panels...; layout=(rows, 1), size=(980, 340 * rows))
    savefig(plt, output_pdf)
end

function summarize_direction(
    Γs::AbstractVector{<:Real},
    loader::Function,
    kind::Symbol,
    xmax::Float64,
)
    summaries = NamedTuple[]
    for Γ in Γs
        x, y = loader(Γ, xmax)
        fits = fit_candidate_models(x, y, kind, xmax)
        push!(summaries, (Γ=Float64(Γ), x=x, y=y, fits=fits))
    end
    return summaries
end

function write_summary_tex(
    path::AbstractString,
    k_summaries::Vector{NamedTuple},
    Ω_summaries::Vector{NamedTuple},
    model_order::Vector{String},
)
    open(path, "w") do io
        println(io, "% Auto-generated by analyze_numerical_asymptotic_fits.jl")
        println(io, "\\newcommand{\\NumericalAsymptoticModelOrder}{", join(model_order, ", "), "}")
        println(io, "\\newcommand{\\NumericalAsymptoticBestK}{")
        for summary in k_summaries
            best = best_model_key(summary.fits)
            println(io, "  \\Gamma=", @sprintf("%.0f", summary.Γ), ": \\texttt{", best, "} (", @sprintf("%.2f", 100 * summary.fits[best].rel_rms), "\\%%)\\\\")
        end
        println(io, "}")
        println(io, "\\newcommand{\\NumericalAsymptoticBestOmega}{")
        for summary in Ω_summaries
            best = best_model_key(summary.fits)
            println(io, "  \\Gamma=", @sprintf("%.0f", summary.Γ), ": \\texttt{", best, "} (", @sprintf("%.2f", 100 * summary.fits[best].rel_rms), "\\%%)\\\\")
        end
        println(io, "}")
    end
end

function main(;
    k_dirs::Dict{Float64, String}=Dict(
        20.0 => "data/smallk_kline_fft_hiBW_Nw12288_Nk2048",
        40.0 => "data/smallk_kline_fft_hiBW_Nw12288_Nk2048",
        80.0 => "data/smallk_kline_fft_hiBW2_Nw12288_Nk2048",
    ),
    Ω_dir::AbstractString="data/smallomega_scan_hiBW_Nw65536_Nk128_wmax320",
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    kfit_max::Float64=0.04,
    Ωfit_max::Float64=0.15,
    output_prefix::AbstractString="note/numerical_asymptotic",
)
    resolved_Γs = collect(Float64.(Γs))
    model_order = ["quadratic", "logplusquadratic", "quartic"]

    k_loader = (Γ, xmax) -> load_k_direction_line(k_dirs[Γ], Γ; kmax=xmax)
    Ω_loader = (Γ, xmax) -> load_omega_direction_line(Ω_dir, Γ; Ωmax=xmax)

    k_summaries = summarize_direction(resolved_Γs, k_loader, :k, kfit_max)
    Ω_summaries = summarize_direction(resolved_Γs, Ω_loader, :Ω, Ωfit_max)

    plot_fit_overlays(
        k_summaries,
        model_order,
        "k",
        "Re{Γ[Π(0,k)-Π(0,0)]}_even";
        output_pdf="$(output_prefix)_k_fits.pdf",
        fit_max=kfit_max,
    )
    plot_ratio_diagnostic(
        k_summaries,
        "k",
        "Re{Γ[Π(0,k)-Π(0,0)]}_even / k^2";
        output_pdf="$(output_prefix)_k_ratio.pdf",
    )
    plot_fit_overlays(
        Ω_summaries,
        model_order,
        "Ω",
        "Re{Γ[Π(Ω,0)-Π(0,0)]}_even";
        output_pdf="$(output_prefix)_omega_fits.pdf",
        fit_max=Ωfit_max,
    )
    plot_ratio_diagnostic(
        Ω_summaries,
        "Ω",
        "Re{Γ[Π(Ω,0)-Π(0,0)]}_even / Ω^2";
        output_pdf="$(output_prefix)_omega_ratio.pdf",
    )

    write_fit_table("note/numerical_asymptotic_k_fit_table.tex", k_summaries, model_order, "k")
    write_fit_table("note/numerical_asymptotic_omega_fit_table.tex", Ω_summaries, model_order, "\\Omega")
    write_summary_tex("note/numerical_asymptotic_summary.tex", k_summaries, Ω_summaries, model_order)

    println("saved $(output_prefix)_k_fits.pdf")
    println("saved $(output_prefix)_k_ratio.pdf")
    println("saved $(output_prefix)_omega_fits.pdf")
    println("saved $(output_prefix)_omega_ratio.pdf")
    println("saved note/numerical_asymptotic_k_fit_table.tex")
    println("saved note/numerical_asymptotic_omega_fit_table.tex")
    println("saved note/numerical_asymptotic_summary.tex")

    for direction in (("k", k_summaries), ("omega", Ω_summaries))
        println(direction[1], " direction:")
        for summary in direction[2]
            println("  Γ=", summary.Γ)
            for key in model_order
                fit = summary.fits[key]
                println("    ", key, ": rel_rms=", @sprintf("%.6f", fit.rel_rms), ", coeffs=[", format_coeff_vector(fit.coeffs), "]")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
