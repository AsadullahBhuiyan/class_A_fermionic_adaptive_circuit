using JLD2
using Plots
using Printf
using Statistics
using Base.Threads: @threads, maxthreadid, threadid

include("collect_cubic_triangle_data.jl")
include("convolution.jl")

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function midpoint_frequency_grid(Nω::Int, ωmax::Float64)
    dω = 2 * ωmax / Nω
    ωs = Vector{Float64}(undef, Nω)
    @inbounds for i in 1:Nω
        ωs[i] = -ωmax + (i - 0.5) * dω
    end
    return ωs, dω
end

function uniform_momentum_grid(Nq::Int)
    dq = 2π / Nq
    qs = Vector{Float64}(undef, Nq)
    @inbounds for j in 1:Nq
        qs[j] = (j - 1) * dq
    end
    return qs, dq
end

@inline function padded_fft_index(i::Int, N::Int)
    pivot = fld(N - 1, 2) + 1
    return i <= pivot ? i - 1 : i - 1 - N
end

function build_fft_frequency_grid(Nω::Int, dω::Float64; padding_factor::Int=4)
    total_Nω = padding_factor * Nω
    ωs = Vector{Float64}(undef, total_Nω)
    @inbounds for i in 1:total_Nω
        ωs[i] = dω * padded_fft_index(i, total_Nω)
    end
    return ωs
end

function padded_fft_indices(N::Int)
    return vcat(collect(0:fld(N - 1, 2)), collect(-fld(N, 2):-1))
end

function effective_frequency_indices(effective_nus::AbstractVector, total_Nω::Int)
    return sort(mod1.(effective_nus .+ 1, total_Nω))
end

signed_k_grid(ks::AbstractVector) = [k <= π ? k : k - 2π for k in ks]

function nearest_zero_indices(values::AbstractVector, npoints::Int)
    order = sortperm(abs.(values))
    selected = order[1:min(npoints, length(values))]
    return sort(selected; by=i -> values[i])
end

@inline function monitored_keldysh_green_entries(
    mode::Val,
    ω::Float64,
    k::Float64,
    r::Float64,
    Γ::Float64,
    η::Float64,
)
    g11, g12, g21, g22 = monitored_retarded_green_entries(ω, k, r, Γ, η)
    ga11 = conj(g11)
    ga12 = conj(g12)
    ga21 = conj(g21)
    ga22 = conj(g22)
    return keldysh_entries(mode, ω, Γ, g11, g12, g21, g22, ga11, ga12, ga21, ga22)
end

@inline function kk_trace_product(
    a11::ComplexF64,
    a12::ComplexF64,
    a21::ComplexF64,
    a22::ComplexF64,
    b11::ComplexF64,
    b12::ComplexF64,
    b21::ComplexF64,
    b22::ComplexF64,
)
    return a11 * b11 + a12 * b21 + a21 * b12 + a22 * b22
end

@inline function kk_trace_product_flat(data1_flat, i1, j1, data2_flat, i2, j2)
    return data1_flat[i1, j1, 1] * data2_flat[i2, j2, 1] +
           data1_flat[i1, j1, 2] * data2_flat[i2, j2, 3] +
           data1_flat[i1, j1, 3] * data2_flat[i2, j2, 2] +
           data1_flat[i1, j1, 4] * data2_flat[i2, j2, 4]
end

@inline function add_k_contributions!(
    partial_k::AbstractMatrix{ComplexF64},
    tid::Int,
    k_values::Vector{Float64},
    ω::Float64,
    q::Float64,
    r::Float64,
    Γ::Float64,
    η::Float64,
    mode::Val,
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
)
    @inbounds for idx in eachindex(k_values)
        h11, h12, h21, h22 = monitored_keldysh_green_entries(mode, ω, q - k_values[idx], r, Γ, η)
        partial_k[tid, idx] += kk_trace_product(h11, h12, h21, h22, g11, g12, g21, g22)
    end
    return nothing
end

@inline function add_omega_contributions!(
    partial_Ω::AbstractMatrix{ComplexF64},
    tid::Int,
    Ω_values::Vector{Float64},
    ω::Float64,
    q::Float64,
    r::Float64,
    Γ::Float64,
    η::Float64,
    mode::Val,
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
)
    @inbounds for idx in eachindex(Ω_values)
        h11, h12, h21, h22 = monitored_keldysh_green_entries(mode, ω - Ω_values[idx], q, r, Γ, η)
        partial_Ω[tid, idx] += kk_trace_product(h11, h12, h21, h22, g11, g12, g21, g22)
    end
    return nothing
end

function kk_direct_lines_for_gamma(
    Γ::Float64,
    ωs::Vector{Float64},
    dω::Float64,
    qs::Vector{Float64},
    k_values::Vector{Float64},
    Ω_values::Vector{Float64};
    r::Float64=1.0,
    η::Float64=1e-4,
    distribution_mode::Symbol=:dyson_sign,
)
    mode = Val(distribution_mode)
    nt = maxthreadid()
    nk = length(k_values)
    nΩ = length(Ω_values)
    partial_base = zeros(ComplexF64, nt)
    partial_k = zeros(ComplexF64, nt, nk)
    partial_Ω = zeros(ComplexF64, nt, nΩ)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        base_local = 0.0 + 0.0im

        @inbounds for ω in ωs
            g11, g12, g21, g22 = monitored_keldysh_green_entries(mode, ω, q, r, Γ, η)
            base_local += kk_trace_product(g11, g12, g21, g22, g11, g12, g21, g22)
            add_k_contributions!(partial_k, tid, k_values, ω, q, r, Γ, η, mode, g11, g12, g21, g22)
            add_omega_contributions!(partial_Ω, tid, Ω_values, ω, q, r, Γ, η, mode, g11, g12, g21, g22)
        end

        partial_base[tid] += base_local
    end

    prefactor = dω / (2π * length(qs))
    Π00 = prefactor * sum(partial_base)
    Πk = prefactor .* vec(sum(partial_k; dims=1))
    ΠΩ = prefactor .* vec(sum(partial_Ω; dims=1))
    return Π00, Πk, ΠΩ
end

function kk_direct_map_for_gamma(
    Γ::Float64,
    ωs::Vector{Float64},
    dω::Float64,
    qs::Vector{Float64},
    Ω_values::Vector{Float64},
    k_values::Vector{Float64};
    r::Float64=1.0,
    η::Float64=1e-4,
    distribution_mode::Symbol=:dyson_sign,
)
    mode = Val(distribution_mode)
    nt = maxthreadid()
    nΩ = length(Ω_values)
    nk = length(k_values)
    partial = zeros(ComplexF64, nt, nΩ, nk)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]

        @inbounds for ω in ωs
            g11, g12, g21, g22 = monitored_keldysh_green_entries(mode, ω, q, r, Γ, η)
            for iΩ in eachindex(Ω_values)
                Ω = Ω_values[iΩ]
                for ik in eachindex(k_values)
                    k = k_values[ik]
                    h11, h12, h21, h22 = monitored_keldysh_green_entries(mode, ω - Ω, q - k, r, Γ, η)
                    partial[tid, iΩ, ik] += kk_trace_product(h11, h12, h21, h22, g11, g12, g21, g22)
                end
            end
        end
    end

    prefactor = dω / (2π * length(qs))
    return prefactor .* dropdims(sum(partial; dims=1); dims=1)
end

function build_keldysh_pair!(
    data1::AbstractArray{ComplexF64,4},
    data2::AbstractArray{ComplexF64,4},
    ωs::AbstractVector,
    ks::AbstractVector,
    active_indices::AbstractVector;
    r::Float64=1.0,
    Γ::Float64=1.0,
    η::Float64=1e-4,
    distribution_mode::Symbol=:dyson_sign,
)
    mode = Val(distribution_mode)
    @inbounds for j in eachindex(ks)
        k = ks[j]
        for i in active_indices
            k11, k12, k21, k22 = monitored_keldysh_green_entries(mode, ωs[i], k, r, Γ, η)
            data1[i, j, 1, 1] = k11
            data1[i, j, 1, 2] = k12
            data1[i, j, 2, 1] = k21
            data1[i, j, 2, 2] = k22
            data2[i, j, 1, 1] = k11
            data2[i, j, 1, 2] = k12
            data2[i, j, 2, 1] = k21
            data2[i, j, 2, 2] = k22
        end
    end
    return nothing
end

function kk_fft_window_for_gamma(
    Γ::Float64;
    Nω::Int,
    Nk::Int,
    dω::Float64,
    NΩ_save::Int,
    Nk_save::Int,
    r::Float64=1.0,
    η::Float64=1e-4,
    distribution_mode::Symbol=:dyson_sign,
)
    ωs = build_fft_frequency_grid(Nω, dω)
    total_Nω = length(ωs)
    effective_nus = padded_fft_indices(Nω)
    active_indices = effective_frequency_indices(effective_nus, total_Nω)
    active_indices = active_indices[sortperm(@view ωs[active_indices])]
    ω_phys = ωs[active_indices]

    ks = collect(2π .* (0:(Nk - 1)) ./ Nk)
    k_signed = signed_k_grid(ks)

    ω_save_indices = nearest_zero_indices(ω_phys, NΩ_save)
    k_save_indices = nearest_zero_indices(k_signed, Nk_save)

    gk1 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    gk2 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    out = zeros(ComplexF64, total_Nω, Nk)

    build_keldysh_pair!(gk1, gk2, ωs, ks, active_indices; r=r, Γ=Γ, η=η, distribution_mode=distribution_mode)
    convolve_RRc_notail!(FlatBinary(kk_trace_product_flat), out, gk1, gk2, dω)

    conv_small = Array{ComplexF64}(undef, length(ω_save_indices), length(k_save_indices))
    @views conv_small .= out[active_indices[ω_save_indices], k_save_indices]

    zero_ω_idx = findfirst(iszero, ω_phys[ω_save_indices])
    zero_k_idx = findfirst(iszero, k_signed[k_save_indices])

    return (
        Γ=Γ,
        ω_phys=ω_phys,
        k_signed=k_signed,
        Ω_values=ω_phys[ω_save_indices],
        k_values=k_signed[k_save_indices],
        conv_small=conv_small,
        Π00=out[active_indices[findfirst(iszero, ω_phys)], findfirst(iszero, k_signed)],
        zero_ω_idx=zero_ω_idx,
        zero_k_idx=zero_k_idx,
        Nω=Nω,
        Nk=Nk,
        dω=dω,
        distribution_mode=String(distribution_mode),
    )
end

function plot_linecuts(results::AbstractVector, output_pdf::AbstractString)
    panels = Plots.Plot[]
    for row in results
        pltΩ = plot(
            row.Ω_values,
            real.(row.ΠΩ .- row.Π00);
            label="Re [ΠKK(Ω,0)-ΠKK(0,0)]",
            color=:blue,
            linewidth=2.2,
            marker=:circle,
            markersize=3,
            markerstrokewidth=0,
            xlabel="Ω",
            ylabel="difference",
            title="Γ = $(round(row.Γ; digits=1))",
        )
        plot!(pltΩ, row.Ω_values, imag.(row.ΠΩ .- row.Π00); label="Im difference", color=:red, linewidth=2.0, linestyle=:dash)

        pltk = plot(
            row.k_values,
            real.(row.Πk .- row.Π00);
            label="Re [ΠKK(0,k)-ΠKK(0,0)]",
            color=:blue,
            linewidth=2.2,
            marker=:circle,
            markersize=3,
            markerstrokewidth=0,
            xlabel="k",
            ylabel="difference",
            title="Γ = $(round(row.Γ; digits=1))",
        )
        plot!(pltk, row.k_values, imag.(row.Πk .- row.Π00); label="Im difference", color=:red, linewidth=2.0, linestyle=:dash)

        push!(panels, pltΩ)
        push!(panels, pltk)
    end

    plt = plot(panels...; layout=(length(results), 2), size=(1200, 320 * length(results)))
    savefig(plt, output_pdf)
end

function plot_heatmaps(results::AbstractVector, output_pdf::AbstractString)
    panels = Plots.Plot[]
    for row in results
        zero_val = row.Π00
        shifted = real.(row.map_values .- zero_val)
        vmax = max(quantile(abs.(vec(shifted)), 0.995), eps())
        plt = heatmap(
            row.k_values,
            row.Ω_values,
            shifted;
            color=:balance,
            clims=(-vmax, vmax),
            xlabel="k",
            ylabel="Ω",
            title="Re [ΠKK(Ω,k)-ΠKK(0,0)], Γ = $(round(row.Γ; digits=1))",
            aspect_ratio=:auto,
        )
        hline!(plt, [0.0]; color=:black, linewidth=1.0, linestyle=:dash, label=nothing)
        vline!(plt, [0.0]; color=:black, linewidth=1.0, linestyle=:dash, label=nothing)
        push!(panels, plt)
    end
    plt = plot(panels...; layout=(length(results), 1), size=(900, 320 * length(results)))
    savefig(plt, output_pdf)
end

function plot_pi00(results::AbstractVector, output_pdf::AbstractString)
    Γs = [row.Γ for row in results]
    Π00 = [real(row.Π00) for row in results]
    plt1 = plot(Γs, Π00; marker=:circle, linewidth=2.2, color=:blue, xlabel="Γ", ylabel="Re ΠKK(0,0)", label="raw")
    plt2 = plot(Γs, Γs .* Π00; marker=:circle, linewidth=2.2, color=:red, xlabel="Γ", ylabel="Γ Re ΠKK(0,0)", label="Γ-scaled")
    plt3 = plot(Γs, (Γs .^ 2) .* Π00; marker=:circle, linewidth=2.2, color=:green, xlabel="Γ", ylabel="Γ² Re ΠKK(0,0)", label="Γ²-scaled")
    plt = plot(plt1, plt2, plt3; layout=(3, 1), size=(800, 900))
    savefig(plt, output_pdf)
end

function main(;
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    distribution_mode::Symbol=:dyson_sign,
    r::Float64=1.0,
    η::Float64=1e-4,
    Nω_line::Int=8192,
    Nq_line::Int=1024,
    ωmax_line::Float64=320.0,
    line_max::Float64=0.03,
    nline::Int=31,
    Nω_map::Int=1024,
    Nq_map::Int=256,
    ωmax_map::Float64=320.0,
    map_max::Float64=0.06,
    nmap::Int=41,
    output_dir::AbstractString="data/full_KK_bubble_dyson_sign",
    output_prefix::AbstractString="note/full_KK_bubble_dyson_sign",
)
    mkpath(output_dir)

    ωs_line, dω_line = midpoint_frequency_grid(Nω_line, ωmax_line)
    qs_line, dq_line = uniform_momentum_grid(Nq_line)
    line_values = collect(range(0.0, line_max; length=nline))

    metadata = (
        Γs=Float64.(Γs),
        distribution_mode=String(distribution_mode),
        r,
        η,
        Nω_line,
        Nq_line,
        ωmax_line,
        dω_line,
        dq_line,
        line_values,
        Nω_map,
        Nq_map,
        ωmax_map,
        map_max,
        nmap,
    )
    jldsave(joinpath(output_dir, "metadata.jld2"); metadata...)

    ωs_map, dω_map = midpoint_frequency_grid(Nω_map, ωmax_map)
    qs_map, dq_map = uniform_momentum_grid(Nq_map)
    map_values = collect(range(-map_max, map_max; length=nmap))

    line_results = NamedTuple[]
    map_results = NamedTuple[]
    for Γraw in Γs
        Γ = Float64(Γraw)
        println("computing full KK direct line cuts for Γ=", Γ)
        Π00, Πk, ΠΩ = kk_direct_lines_for_gamma(
            Γ,
            ωs_line,
            dω_line,
            qs_line,
            line_values,
            line_values;
            r=r,
            η=η,
            distribution_mode=distribution_mode,
        )
        line_row = (Γ=Γ, Π00=Π00, k_values=line_values, Πk=Πk, Ω_values=line_values, ΠΩ=ΠΩ)
        push!(line_results, line_row)
        jldsave(joinpath(output_dir, "direct_gamma_$(gamma_token(Γ)).jld2"); line_row...)

        println("computing full KK direct map for Γ=", Γ)
        map_grid = kk_direct_map_for_gamma(
            Γ,
            ωs_map,
            dω_map,
            qs_map,
            map_values,
            map_values;
            r=r,
            η=η,
            distribution_mode=distribution_mode,
        )
        map_row = (
            Γ=Γ,
            Π00=Π00,
            Ω_values=map_values,
            k_values=map_values,
            map_values=map_grid,
            Nω_map=Nω_map,
            Nq_map=Nq_map,
            ωmax_map=ωmax_map,
            dω_map=dω_map,
            dq_map=dq_map,
        )
        push!(map_results, map_row)
        jldsave(joinpath(output_dir, "map_gamma_$(gamma_token(Γ)).jld2"); map_row...)
    end

    line_pdf = "$(output_prefix)_linecuts.pdf"
    heatmap_pdf = "$(output_prefix)_heatmaps.pdf"
    pi00_pdf = "$(output_prefix)_pi00.pdf"
    plot_linecuts(line_results, line_pdf)
    plot_heatmaps(map_results, heatmap_pdf)
    plot_pi00(line_results, pi00_pdf)

    jldsave(
        joinpath(output_dir, "summary.jld2");
        metadata,
        line_results,
        map_results,
        line_pdf,
        heatmap_pdf,
        pi00_pdf,
    )

    println("saved ", line_pdf)
    println("saved ", heatmap_pdf)
    println("saved ", pi00_pdf)
    for row in line_results
        println(
            @sprintf(
                "Γ=%.1f ReΠKK00=%.8e ImΠKK00=%.8e ΓReΠKK00=%.8e Γ²ReΠKK00=%.8e",
                row.Γ,
                real(row.Π00),
                imag(row.Π00),
                row.Γ * real(row.Π00),
                row.Γ^2 * real(row.Π00),
            ),
        )
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
