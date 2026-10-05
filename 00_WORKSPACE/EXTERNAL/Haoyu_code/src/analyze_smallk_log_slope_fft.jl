using JLD2
using LinearAlgebra
using Printf
using Plots
using LaTeXStrings
using Statistics

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function fixed_slope_intercept(x::AbstractVector, y::AbstractVector, slope::Float64)
    return mean(y .- slope .* x)
end

function free_line_fit(x::AbstractVector, y::AbstractVector)
    X = hcat(x, ones(length(x)))
    coeffs = X \ y
    return (slope=coeffs[1], intercept=coeffs[2], fitted=X * coeffs)
end

function analyze_smallk_log_slope_fft(;
    input_dir::AbstractString,
    Γs::Union{Nothing, AbstractVector}=nothing,
    kfit_max::Float64=0.08,
    output_pdf::AbstractString="smallk_log_slope_fft.pdf",
    output_jld2::AbstractString="smallk_log_slope_fft.jld2",
)
    meta = load(joinpath(input_dir, "metadata.jld2"))
    resolved_Γs = isnothing(Γs) ? collect(Float64.(meta["Γs"])) : collect(Float64.(Γs))
    fixed_slope = -1 / (4π)

    panels = Plots.Plot[]
    summaries = NamedTuple[]

    for Γ in resolved_Γs
        data = load(joinpath(input_dir, "gamma_$(gamma_token(Γ)).jld2"))
        ks = data["k_positive"]
        vals = data["kline_positive"]
        y = real.(Γ .* vals .- Γ .* data["kline"][findfirst(==(0.0), data["k_values"])]) ./ (ks .^ 2)
        x = log.(Γ ./ ks)

        sel = findall(ks .<= kfit_max)
        xfit = x[sel]
        yfit = y[sel]
        free = free_line_fit(xfit, yfit)
        fixed_intercept = fixed_slope_intercept(xfit, yfit, fixed_slope)
        fixed_fit = fixed_slope .* xfit .+ fixed_intercept
        fixed_rms = sqrt(mean((yfit .- fixed_fit) .^ 2))
        free_rms = sqrt(mean((yfit .- free.fitted) .^ 2))

        plt = scatter(
            xfit,
            yfit;
            label="data",
            xlabel=L"\log(\Gamma/k)",
            ylabel=L"\mathrm{Re}[\Gamma \Pi(0,k)-\Gamma \Pi(0,0)]/k^2",
            title="Γ = $(Γ), ωband = $(round(meta["ωband"]; digits=2))",
            markersize=4,
        )
        plot!(plt, xfit, free.fitted; linewidth=2.5, label=@sprintf("free fit: %.4f x %+ .4f", free.slope, free.intercept))
        plot!(plt, xfit, fixed_fit; linewidth=2.5, linestyle=:dash, label=@sprintf("fixed slope -1/(4π): b = %.4f", fixed_intercept))
        push!(panels, plt)

        push!(summaries, (
            Γ=Γ,
            kfit_max=kfit_max,
            nfit=length(sel),
            free_slope=free.slope,
            free_intercept=free.intercept,
            free_rms=free_rms,
            fixed_slope=fixed_slope,
            fixed_intercept=fixed_intercept,
            fixed_rms=fixed_rms,
        ))
    end

    plot_rows = length(panels)
    plt = plot(panels...; layout=(plot_rows, 1), size=(900, 320 * plot_rows))
    savefig(plt, output_pdf)

    jldsave(
        output_jld2;
        input_dir,
        Γs=resolved_Γs,
        ωband=meta["ωband"],
        dω=meta["dω"],
        kfit_max,
        fixed_slope,
        summaries,
        output_pdf,
    )

    println("saved ", output_pdf)
    println("saved ", output_jld2)
    for summary in summaries
        println(summary)
    end
end

main(; kwargs...) = analyze_smallk_log_slope_fft(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    input_dir = isempty(ARGS) ? error("pass input directory") : ARGS[1]
    analyze_smallk_log_slope_fft(input_dir=input_dir)
end
