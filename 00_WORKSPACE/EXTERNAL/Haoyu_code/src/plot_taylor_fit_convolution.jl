using JLD2
using Plots
using LaTeXStrings

const TAYLOR_FIT_RESULTS_FILE = "taylor_fit_convolution_results.jld2"
const TAYLOR_FIT_FORMULA = L"C(t,x)=\mathrm{Tr}\left[G_R(t,x)G_A(-t,-x)\right]"
const TAYLOR_EXPANSION_FORMULA = L"\Gamma C(\Omega,k)=C+i\,\frac{C_0\Omega}{\Gamma/2}-i\,\frac{C_1\sin k}{\Gamma/2}-\frac{C_{00}\Omega^2+C_{11}\sin^2 k-C_{01}\Omega\sin k}{(\Gamma/2)^2}"

function coefficient_filename_token(name::AbstractString)
    return lowercase(replace(name, "{" => "", "}" => "", "," => "", " " => "", "/" => "_"))
end

function coefficient_gamma_power(name::AbstractString)
    powers = Dict(
        "C" => 0,
        "C_0" => 0,
        "C_1" => -1,
        "C_00" => -2,
        "C_01" => -2,
        "C_11" => -2,
    )
    return powers[name]
end

function coefficient_plot_label(name::AbstractString)
    power = coefficient_gamma_power(name)
    latex_name = replace(name, "_" => "_{")
    if occursin("_", name)
        head, tail = split(name, "_"; limit=2)
        latex_name = string(head, "_{", tail, "}")
    end
    expr = power == 0 ? latex_name : power == 1 ? string(latex_name, "\\Gamma") : string(latex_name, "\\Gamma^{", power, "}")
    return latexstring(expr)
end

default_results_file(::Val{:fft}) = TAYLOR_FIT_RESULTS_FILE
default_results_file(::Val{:tail}) = TAYLOR_FIT_RESULTS_FILE

default_output_pdf(::Val{:fft}, Nω::Integer, Nk::Integer) = "taylor_fit_coefficients_fft_Nw$(Nω)_Nk$(Nk).pdf"
default_output_pdf(::Val{:tail}, Nω::Integer, Nk::Integer) = "taylor_fit_coefficients_tail_Nw$(Nω)_Nk$(Nk).pdf"

coefficient_uses_log_yscale(name::AbstractString) = name in ("C_00", "C_01", "C_11")
format_domega(dω::Real) = string(round(dω; sigdigits=2))
format_decimal_tick(value::Real) = replace(string(round(value; sigdigits=2)), r"\.0$" => "")

function coefficient_ylims(name::AbstractString, values::AbstractVector)
    ymin = minimum(values)
    ymax = maximum(values)
    if name == "C"
        upper = 1.05 * max(ymax, 2.0)
        lower = min(0.0, ymin - 0.05 * (upper - ymin))
        return (lower, upper)
    end
    if coefficient_uses_log_yscale(name)
        return (0.9 * ymin, 1.1 * ymax)
    end
    span = ymax - ymin
    padding = iszero(span) ? max(abs(ymax), 1.0) * 0.05 : 0.05 * span
    return (ymin - padding, ymax + padding)
end

function coefficient_yticks(name::AbstractString, values::AbstractVector)
    if !coefficient_uses_log_yscale(name)
        return :auto
    end
    ymin, ymax = coefficient_ylims(name, values)
    min_decade = floor(Int, log10(ymin))
    max_decade = ceil(Int, log10(ymax))
    ticks = Float64[]
    for decade in min_decade:max_decade
        scale = 10.0^decade
        for mantissa in (1.0, 2.0, 5.0)
            tick = mantissa * scale
            if ymin <= tick <= ymax
                push!(ticks, tick)
            end
        end
    end
    labels = [format_decimal_tick(tick) for tick in ticks]
    return (ticks, labels)
end

function plot_coefficient_sweeps(
    Nω::Integer,
    Nk::Integer,
    dω::Real,
    Γs::AbstractVector,
    coefficients::AbstractMatrix,
    coefficient_names::AbstractVector;
    output_pdf::AbstractString="taylor_fit_coefficients.pdf",
)
    linewidth = 2.5
    markersize = 4
    plots = Vector{Any}(undef, length(coefficient_names))

    for idx in eachindex(coefficient_names)
        name = coefficient_names[idx]
        scaled_label = coefficient_plot_label(name)
        scaled_values = coefficients[idx, :] .* (Γs .^ coefficient_gamma_power(name))
        plt = plot(
            title=scaled_label,
            xlabel=L"\Gamma",
            ylabel=scaled_label,
            legend=:best,
            yscale=coefficient_uses_log_yscale(name) ? :log10 : :identity,
            ylims=coefficient_ylims(name, scaled_values),
            yticks=coefficient_yticks(name, scaled_values),
            minorgrid=coefficient_uses_log_yscale(name),
            minorticks=5,
        )
        plot!(
            plt,
            Γs,
            scaled_values,
            color=:blue,
            linewidth=linewidth,
            marker=:circle,
            markersize=markersize,
            markerstrokewidth=0,
            label=scaled_label,
        )
        if name == "C"
            hline!(plt, [2.0], color=:black, linestyle=:dash, linewidth=1.5, label=L"C=2")
        end
        plots[idx] = plt
    end

    combined = plot(
        plots...;
        layout=(3, 2),
        size=(1200, 1200),
        plot_title=string(
            TAYLOR_FIT_FORMULA,
            "\n",
            latexstring("N_{\\omega}=", string(Nω), ",\\ N_k=", string(Nk), ",\\ d\\omega=", format_domega(dω)),
            "\n",
            TAYLOR_EXPANSION_FORMULA,
        ),
        plot_titlevspan=0.12,
    )
    savefig(combined, output_pdf)
    display(combined)

    return combined
end

function plot_taylor_fit_main(;
    method::Val=Val(:fft),
    results_file::Union{Nothing, AbstractString}=nothing,
    output_pdf::Union{Nothing, AbstractString}=nothing,
)
    resolved_results_file = isnothing(results_file) ? default_results_file(method) : results_file

    results = load(resolved_results_file)
    Nω = haskey(results, "Nω") ? results["Nω"] : results["params"].Nω
    Nk = haskey(results, "Nk") ? results["Nk"] : results["params"].Nk
    dω = haskey(results, "dω") ? results["dω"] : results["params"].dω
    resolved_output_pdf = isnothing(output_pdf) ? default_output_pdf(method, Nω, Nk) : output_pdf
    Γs = results["Γs"]
    coefficients = results["coefficients"]
    coefficient_names = results["coefficient_names"]

    return plot_coefficient_sweeps(
        Nω,
        Nk,
        dω,
        Γs,
        coefficients,
        coefficient_names;
        output_pdf=resolved_output_pdf,
    )
end

main(method::Val; kwargs...) = plot_taylor_fit_main(; method=method, kwargs...)
main(; kwargs...) = plot_taylor_fit_main(; kwargs...)

running_in_vscode_repl() = isinteractive() && isdefined(Main, :VSCodeServer)

if abspath(PROGRAM_FILE) == (@__FILE__) || running_in_vscode_repl()
    plot_taylor_fit_main()
    nothing
end
