using JLD2
using LinearAlgebra
using Printf
using Plots
using LaTeXStrings

include("taylor_fit_convolution.jl")
include("touching_model.jl")

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function load_sparse_scan_metadata(output_dir::AbstractString)
    return load(joinpath(output_dir, "metadata.jld2"))
end

function gamma_scan_filepath(output_dir::AbstractString, Γ::Real)
    return joinpath(output_dir, "gamma_$(gamma_token(Γ)).jld2")
end

function local_model_filepath(output_dir::AbstractString, Γ::Real)
    return joinpath(output_dir, "local_touching_gamma_$(gamma_token(Γ)).jld2")
end

function comparison_plot_filepath(output_dir::AbstractString, Γ::Real)
    return joinpath(output_dir, "comparison_gamma_$(gamma_token(Γ)).pdf")
end

function matched_model_filepath(output_dir::AbstractString, Γ::Real)
    return joinpath(output_dir, "matched_model_gamma_$(gamma_token(Γ)).jld2")
end

function locate_cell(grid::AbstractVector{<:Real}, x::Real)
    if x < grid[1] || x > grid[end]
        return nothing
    end
    hi = searchsortedfirst(grid, x)
    if hi == 1
        return 1, 1, 0.0
    elseif hi > length(grid)
        return length(grid), length(grid), 0.0
    elseif grid[hi] == x
        return hi, hi, 0.0
    else
        lo = hi - 1
        t = (x - grid[lo]) / (grid[hi] - grid[lo])
        return lo, hi, t
    end
end

function indices_with_abs_cut(grid::AbstractVector{<:Real}, cut::Real)
    indices = findall(abs.(grid) .<= cut)
    return isempty(indices) ? nearest_zero_indices(grid, 1) : indices
end

function center_line(values::AbstractVector{ComplexF64})
    return values .- values[cld(length(values), 2)]
end

function default_remainder_k_fit_cut(Γ::Float64)
    return min(0.5, 2.5 * Γ)
end

function bilinear_interpolate(
    Ω_grid::AbstractVector{<:Real},
    k_grid::AbstractVector{<:Real},
    values::AbstractMatrix{ComplexF64},
    Ω::Real,
    k::Real,
)
    ωcell = locate_cell(Ω_grid, Ω)
    kcell = locate_cell(k_grid, k)
    if isnothing(ωcell) || isnothing(kcell)
        return ComplexF64(NaN, NaN)
    end
    i0, i1, tx = ωcell
    j0, j1, ty = kcell
    if i0 == i1 && j0 == j1
        return values[i0, j0]
    elseif i0 == i1
        return (1 - ty) * values[i0, j0] + ty * values[i0, j1]
    elseif j0 == j1
        return (1 - tx) * values[i0, j0] + tx * values[i1, j0]
    else
        v00 = values[i0, j0]
        v10 = values[i1, j0]
        v01 = values[i0, j1]
        v11 = values[i1, j1]
        return (1 - tx) * (1 - ty) * v00 +
               tx * (1 - ty) * v10 +
               (1 - tx) * ty * v01 +
               tx * ty * v11
    end
end

function build_regular_remainder_design(Ω_fit::AbstractVector, k_fit::AbstractVector)
    nrows = length(Ω_fit) * length(k_fit)
    design = Matrix{Float64}(undef, 2 * nrows, 6)
    row = 1
    @inbounds for kval in k_fit
        for Ω in Ω_fit
            design[row, 1] = 1.0
            design[row, 2] = 0.0
            design[row, 3] = 0.0
            design[row, 4] = Ω^2
            design[row, 5] = Ω * kval
            design[row, 6] = kval^2

            imag_row = row + nrows
            design[imag_row, 1] = 0.0
            design[imag_row, 2] = Ω
            design[imag_row, 3] = kval
            design[imag_row, 4] = 0.0
            design[imag_row, 5] = 0.0
            design[imag_row, 6] = 0.0
            row += 1
        end
    end
    return design
end

function evaluate_regular_remainder(coeffs::AbstractVector, Ω::Float64, k::Float64)
    return coeffs[1] +
           im * coeffs[2] * Ω +
           im * coeffs[3] * k +
           coeffs[4] * Ω^2 +
           coeffs[5] * Ω * k +
           coeffs[6] * k^2
end

function fit_regular_remainder(
    Ω_values::AbstractVector,
    k_values::AbstractVector,
    residual_grid::AbstractMatrix{ComplexF64};
    NΩ_fit::Int=61,
    Nk_fit::Int=21,
    k_fit_cut::Union{Nothing, Float64}=nothing,
)
    Ω_indices = nearest_zero_indices(Ω_values, min(NΩ_fit, length(Ω_values)))
    k_indices = isnothing(k_fit_cut) ?
        nearest_zero_indices(k_values, min(Nk_fit, length(k_values))) :
        indices_with_abs_cut(k_values, k_fit_cut)
    Ω_fit = Ω_values[Ω_indices]
    k_fit = k_values[k_indices]
    design = build_regular_remainder_design(Ω_fit, k_fit)
    y = vec(@view residual_grid[Ω_indices, k_indices])
    yfit = vcat(real.(y), imag.(y))
    coeffs = qr(design) \ yfit
    return (
        coeffs=coeffs,
        Ω_indices=Ω_indices,
        k_indices=k_indices,
        Ω_fit=Ω_fit,
        k_fit=k_fit,
    )
end

function build_matched_model_grid(
    Ω_values::AbstractVector,
    k_values::AbstractVector,
    local_grid::AbstractMatrix{ComplexF64},
    coeffs::AbstractVector,
)
    matched = similar(local_grid)
    @inbounds for j in eachindex(k_values), i in eachindex(Ω_values)
        matched[i, j] = local_grid[i, j] + evaluate_regular_remainder(coeffs, Ω_values[i], k_values[j])
    end
    return matched
end

function sample_line(
    Ω_grid::AbstractVector{<:Real},
    k_grid::AbstractVector{<:Real},
    values::AbstractMatrix{ComplexF64},
    Ω_line::AbstractVector{<:Real},
    k_line::AbstractVector{<:Real},
)
    @assert length(Ω_line) == length(k_line)
    out = Vector{ComplexF64}(undef, length(Ω_line))
    @inbounds for i in eachindex(Ω_line)
        out[i] = bilinear_interpolate(Ω_grid, k_grid, values, Ω_line[i], k_line[i])
    end
    return out
end

function build_local_touching_sparse_grid(
    metadata,
    Γ::Float64;
    pcut::Float64=TOUCHING_DEFAULT_PCUT,
    ωcut::Float64=TOUCHING_DEFAULT_OMEGA_CUT,
    cutoff_power::Int=TOUCHING_DEFAULT_CUTOFF_POWER,
)
    ωs = metadata["ωs"]
    active_indices = metadata["active_indices"]
    ω_save_indices = metadata["ω_save_indices"]
    k_save_indices = metadata["k_save_indices"]
    ks = metadata["ks"]
    η = metadata["η"]
    dω = metadata["dω"]
    method = Val(Symbol(metadata["method"]))
    ϵ = metadata["ϵ"]
    tailorder = metadata["tailorder"]
    tailindices = metadata["tailindices"]

    total_Nω = length(ωs)
    Nk = length(ks)
    gr1 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    gr2 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    out_fft_local = zeros(ComplexF64, total_Nω, Nk)
    build_local_touching_pair!(
        gr1,
        gr2,
        ωs,
        ks,
        active_indices;
        Γ=Γ,
        η=η,
        pcut=pcut,
        ωcut=ωcut,
        cutoff_power=cutoff_power,
    )
    run_convolution!(method, out_fft_local, gr1, gr2, dω, ωs, active_indices, tailindices, ϵ, tailorder)
    conv_small = Array{ComplexF64}(undef, length(ω_save_indices), length(k_save_indices))
    @views conv_small .= out_fft_local[active_indices[ω_save_indices], k_save_indices]
    return conv_small
end

function save_local_touching_sparse_grid(
    output_dir::AbstractString,
    Γ::Float64,
    Ω_values::AbstractVector,
    k_values::AbstractVector,
    conv_small::AbstractMatrix{ComplexF64};
    pcut::Float64,
    ωcut::Float64,
    cutoff_power::Int,
)
    filepath = local_model_filepath(output_dir, Γ)
    jldsave(
        filepath;
        Γ,
        Ω_values,
        k_values,
        conv_small,
        pcut,
        ωcut,
        cutoff_power,
    )
    return filepath
end

function line_domain_for_alpha(
    Ωmax::Real,
    kmax::Real,
    α::Real;
    npts::Int=300,
)
    if α != 0
        kmax = min(kmax, Ωmax / abs(α))
    end
    return collect(range(-kmax, kmax; length=npts))
end

function comparison_plot(
    Γ::Float64,
    Ω_values::AbstractVector,
    k_values::AbstractVector,
    numeric_grid::AbstractMatrix{ComplexF64},
    model_grid::AbstractMatrix{ComplexF64};
    alphas::AbstractVector=[0.5, 1.0, 2.0],
    Ω_plot_cut::Float64=0.2,
    k_plot_cut::Float64=0.2,
    center_lines::Bool=true,
)
    plt = plot(layout=(2 + length(alphas), 2), size=(1200, 320 * (2 + length(alphas))), legend=:best)

    Ω_indices = indices_with_abs_cut(Ω_values, Ω_plot_cut)
    k_indices = indices_with_abs_cut(k_values, k_plot_cut)

    ω0_numeric = numeric_grid[:, findfirst(==(0.0), k_values)]
    ω0_model = model_grid[:, findfirst(==(0.0), k_values)]
    if center_lines
        ω0_numeric = center_line(ω0_numeric)
        ω0_model = center_line(ω0_model)
    end
    plot!(plt[1], Ω_values[Ω_indices], real.(ω0_numeric[Ω_indices]), label="numeric", color=:blue, linewidth=2, xlabel=L"\Omega", ylabel="Re")
    plot!(plt[1], Ω_values[Ω_indices], real.(ω0_model[Ω_indices]), label="matched asymptotics", color=:red, linestyle=:dash, linewidth=2)
    plot!(plt[2], Ω_values[Ω_indices], imag.(ω0_numeric[Ω_indices]), label="numeric", color=:blue, linewidth=2, xlabel=L"\Omega", ylabel="Im")
    plot!(plt[2], Ω_values[Ω_indices], imag.(ω0_model[Ω_indices]), label="matched asymptotics", color=:red, linestyle=:dash, linewidth=2)
    title!(plt[1], L"(\Omega,0)")
    title!(plt[2], L"(\Omega,0)")

    k0_numeric = numeric_grid[findfirst(==(0.0), Ω_values), :]
    k0_model = model_grid[findfirst(==(0.0), Ω_values), :]
    if center_lines
        k0_numeric = center_line(k0_numeric)
        k0_model = center_line(k0_model)
    end
    plot!(plt[3], k_values[k_indices], real.(k0_numeric[k_indices]), label="numeric", color=:blue, linewidth=2, xlabel="k", ylabel="Re")
    plot!(plt[3], k_values[k_indices], real.(k0_model[k_indices]), label="matched asymptotics", color=:red, linestyle=:dash, linewidth=2)
    plot!(plt[4], k_values[k_indices], imag.(k0_numeric[k_indices]), label="numeric", color=:blue, linewidth=2, xlabel="k", ylabel="Im")
    plot!(plt[4], k_values[k_indices], imag.(k0_model[k_indices]), label="matched asymptotics", color=:red, linestyle=:dash, linewidth=2)
    title!(plt[3], L"(0,k)")
    title!(plt[4], L"(0,k)")

    for (offset, α) in enumerate(alphas)
        kline = line_domain_for_alpha(Ω_plot_cut, k_plot_cut, α)
        Ωline = α .* kline
        numeric_line = sample_line(Ω_values, k_values, numeric_grid, Ωline, kline)
        model_line = sample_line(Ω_values, k_values, model_grid, Ωline, kline)
        if center_lines
            numeric_line = center_line(numeric_line)
            model_line = center_line(model_line)
        end
        row = offset + 2
        plot!(plt[2 * row - 1], kline, real.(numeric_line), label="numeric", color=:blue, linewidth=2, xlabel="k", ylabel="Re")
        plot!(plt[2 * row - 1], kline, real.(model_line), label="matched asymptotics", color=:red, linestyle=:dash, linewidth=2)
        plot!(plt[2 * row], kline, imag.(numeric_line), label="numeric", color=:blue, linewidth=2, xlabel="k", ylabel="Im")
        plot!(plt[2 * row], kline, imag.(model_line), label="matched asymptotics", color=:red, linestyle=:dash, linewidth=2)
        title!(plt[2 * row - 1], latexstring("\\Omega=", string(α), "k"))
        title!(plt[2 * row], latexstring("\\Omega=", string(α), "k"))
    end

    plot_title = center_lines ?
        "Γ = $(Γ), centered asymptotic lines" :
        "Γ = $(Γ), asymptotic lines"
    plot!(plt; plot_title=plot_title)
    return plt
end

function compare_touching_model(;
    output_dir::AbstractString,
    Γs::Union{Nothing, AbstractVector}=nothing,
    pcut::Float64=TOUCHING_DEFAULT_PCUT,
    ωcut::Float64=TOUCHING_DEFAULT_OMEGA_CUT,
    cutoff_power::Int=TOUCHING_DEFAULT_CUTOFF_POWER,
    NΩ_remainder_fit::Int=61,
    Nk_remainder_fit::Int=21,
    k_remainder_fit_cut_scale::Float64=2.5,
    max_k_remainder_fit_cut::Float64=0.5,
    alphas::AbstractVector=[0.5, 1.0, 2.0],
    Ω_plot_cut::Float64=0.2,
    k_plot_cut::Float64=0.2,
    center_lines::Bool=true,
)
    metadata = load_sparse_scan_metadata(output_dir)
    resolved_Γs = isnothing(Γs) ? metadata["Γs"] : collect(Float64.(Γs))
    Ω_values = metadata["Ω_values"]
    k_values = metadata["k_values"]

    plot_files = String[]
    for Γ in resolved_Γs
        numeric = load(gamma_scan_filepath(output_dir, Γ))
        local_conv_small = build_local_touching_sparse_grid(
            metadata,
            Γ;
            pcut=pcut,
            ωcut=ωcut,
            cutoff_power=cutoff_power,
        )
        save_local_touching_sparse_grid(
            output_dir,
            Γ,
            Ω_values,
            k_values,
            local_conv_small;
            pcut=pcut,
            ωcut=ωcut,
            cutoff_power=cutoff_power,
        )
        remainder_fit = fit_regular_remainder(
            Ω_values,
            k_values,
            numeric["conv_small"] .- local_conv_small;
            NΩ_fit=NΩ_remainder_fit,
            Nk_fit=Nk_remainder_fit,
            k_fit_cut=min(max_k_remainder_fit_cut, k_remainder_fit_cut_scale * Γ),
        )
        matched_grid = build_matched_model_grid(Ω_values, k_values, local_conv_small, remainder_fit.coeffs)
        jldsave(
            matched_model_filepath(output_dir, Γ);
            Γ,
            Ω_values,
            k_values,
            matched_grid,
            remainder_coeffs=remainder_fit.coeffs,
            remainder_Ω_fit=remainder_fit.Ω_fit,
            remainder_k_fit=remainder_fit.k_fit,
            remainder_k_fit_cut=min(max_k_remainder_fit_cut, k_remainder_fit_cut_scale * Γ),
        )
        plt = comparison_plot(
            Γ,
            Ω_values,
            k_values,
            numeric["conv_small"],
            matched_grid;
            alphas=alphas,
            Ω_plot_cut=Ω_plot_cut,
            k_plot_cut=k_plot_cut,
            center_lines=center_lines,
        )
        filepath = comparison_plot_filepath(output_dir, Γ)
        savefig(plt, filepath)
        push!(plot_files, filepath)
        println("saved comparison plot ", filepath)
    end

    jldsave(
        joinpath(output_dir, "comparison_index.jld2");
        Γs=resolved_Γs,
        plot_files,
        pcut,
        ωcut,
        cutoff_power,
        NΩ_remainder_fit,
        Nk_remainder_fit,
        k_remainder_fit_cut_scale,
        max_k_remainder_fit_cut,
        alphas=collect(Float64.(alphas)),
        Ω_plot_cut,
        k_plot_cut,
        center_lines,
    )
    return (
        output_dir=output_dir,
        Γs=resolved_Γs,
        plot_files=plot_files,
    )
end

main(; kwargs...) = compare_touching_model(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    output_dir = isempty(ARGS) ? "sparse_convolution_scan_Nw65536_Nk128" : ARGS[1]
    compare_touching_model(output_dir=output_dir)
end
