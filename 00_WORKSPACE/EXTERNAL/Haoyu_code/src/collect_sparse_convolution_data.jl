using JLD2
using LinearAlgebra
using Printf

include("taylor_fit_convolution.jl")

const SPARSE_SCAN_DIR_PREFIX = "sparse_convolution_scan"

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function select_small_window_indices(
    ω_phys::AbstractVector,
    k_signed::AbstractVector;
    NΩ_save::Int,
    Nk_save::Int,
)
    ω_indices = nearest_zero_indices(ω_phys, min(NΩ_save, length(ω_phys)))
    k_indices = nearest_zero_indices(k_signed, min(Nk_save, length(k_signed)))
    return ω_indices, k_indices
end

function flatten_sparse_grid(
    Ω_values::AbstractVector,
    k_values::AbstractVector,
    conv_small::AbstractMatrix{ComplexF64},
    Γ::Float64,
)
    nΩ = length(Ω_values)
    nk = length(k_values)
    nsamples = nΩ * nk
    sample_Ω = Vector{Float64}(undef, nsamples)
    sample_k = Vector{Float64}(undef, nsamples)
    sample_Γ = fill(Γ, nsamples)
    sample_conv = Vector{ComplexF64}(undef, nsamples)
    idx = 1
    @inbounds for j in eachindex(k_values)
        for i in eachindex(Ω_values)
            sample_Ω[idx] = Ω_values[i]
            sample_k[idx] = k_values[j]
            sample_conv[idx] = conv_small[i, j]
            idx += 1
        end
    end
    return sample_Ω, sample_k, sample_Γ, sample_conv
end

function sparse_scan_output_dir(Nω::Int, Nk::Int)
    return "$(SPARSE_SCAN_DIR_PREFIX)_Nw$(Nω)_Nk$(Nk)"
end

function save_sparse_scan_metadata(
    output_dir::AbstractString,
    metadata::NamedTuple,
)
    jldsave(joinpath(output_dir, "metadata.jld2"); metadata...)
    return nothing
end

function save_sparse_scan_gamma(
    output_dir::AbstractString,
    Γ::Float64,
    Ω_values::AbstractVector,
    k_values::AbstractVector,
    sin_k_values::AbstractVector,
    conv_small::AbstractMatrix{ComplexF64},
    ω_indices::AbstractVector,
    k_indices::AbstractVector,
)
    sample_Ω, sample_k, sample_Γ, sample_conv = flatten_sparse_grid(Ω_values, k_values, conv_small, Γ)
    filepath = joinpath(output_dir, "gamma_$(gamma_token(Γ)).jld2")
    jldsave(
        filepath;
        Γ,
        Ω_values,
        k_values,
        sin_k_values,
        conv_small,
        ω_indices=collect(ω_indices),
        k_indices=collect(k_indices),
        sample_Ω,
        sample_k,
        sample_Γ,
        sample_conv,
    )
    return filepath
end

function collect_sparse_convolution_data(;
    method::Val=Val(:fft),
    Nk::Int=128,
    Nω::Int=2^16,
    dω::Float64=(2π / Nk)^2,
    r::Float64=1.0,
    η::Float64=1e-4,
    Γs::AbstractVector=[0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0],
    ϵ::Float64=0.3,
    tailorder::Int=1,
    NΩ_save::Int=401,
    Nk_save::Int=128,
    output_dir::Union{Nothing, AbstractString}=nothing,
)
    Γs = collect(Float64.(Γs))
    resolved_output_dir = isnothing(output_dir) ? sparse_scan_output_dir(Nω, Nk) : String(output_dir)
    mkpath(resolved_output_dir)

    ωs = build_frequency_grid(Nω, dω)
    total_Nω = length(ωs)
    effective_nus = padded_fft_indices(Nω)
    active_indices = effective_frequency_indices(effective_nus, total_Nω)
    active_order = sortperm(@view ωs[active_indices])
    active_indices = active_indices[active_order]
    ω_phys = ωs[active_indices]
    tailindices = default_tailindices(effective_nus, total_Nω, tailorder)

    ks = collect(2π .* (0:(Nk - 1)) ./ Nk)
    k_signed = signed_k_grid(ks)
    sin_k = sin_k_grid(ks)

    ω_save_indices, k_save_indices = select_small_window_indices(
        ω_phys,
        k_signed;
        NΩ_save=NΩ_save,
        Nk_save=Nk_save,
    )

    Ω_values = ω_phys[ω_save_indices]
    k_values = k_signed[k_save_indices]
    sin_k_values = sin_k[k_save_indices]

    metadata = (
        method=convolution_method_name(method),
        Nω=Nω,
        total_Nω=total_Nω,
        Nk=Nk,
        dω=dω,
        r=r,
        η=η,
        ϵ=ϵ,
        tailorder=tailorder,
        Γs=Γs,
        ωs=ωs,
        ω_phys=ω_phys,
        ks=ks,
        k_signed=k_signed,
        sin_k=sin_k,
        active_indices=collect(active_indices),
        tailindices=collect(tailindices),
        ω_save_indices=collect(ω_save_indices),
        k_save_indices=collect(k_save_indices),
        Ω_values=Ω_values,
        k_values=k_values,
        sin_k_values=sin_k_values,
    )
    save_sparse_scan_metadata(resolved_output_dir, metadata)

    gr1 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    gr2 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    out_fft_local = zeros(ComplexF64, total_Nω, Nk)

    saved_files = String[]
    for Γ in Γs
        println("computing sparse convolution scan for Γ=", Γ)
        fill!(gr1, 0)
        fill!(gr2, 0)
        fill!(out_fft_local, 0)
        build_retarded_pair!(gr1, gr2, ωs, ks, active_indices; r=r, Γ=Γ, η=η)
        run_convolution!(method, out_fft_local, gr1, gr2, dω, ωs, active_indices, tailindices, ϵ, tailorder)
        conv_small = Array{ComplexF64}(undef, length(ω_save_indices), length(k_save_indices))
        @views conv_small .= out_fft_local[active_indices[ω_save_indices], k_save_indices]
        filepath = save_sparse_scan_gamma(
            resolved_output_dir,
            Γ,
            Ω_values,
            k_values,
            sin_k_values,
            conv_small,
            ω_save_indices,
            k_save_indices,
        )
        println("saved ", filepath)
        push!(saved_files, filepath)
    end

    jldsave(joinpath(resolved_output_dir, "index.jld2"); output_dir=resolved_output_dir, Γs, saved_files)
    return (
        output_dir=resolved_output_dir,
        Γs=Γs,
        Ω_values=Ω_values,
        k_values=k_values,
        saved_files=saved_files,
    )
end

main(; kwargs...) = collect_sparse_convolution_data(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
