using JLD2
using Printf
using Base.Threads: @threads, maxthreadid, nthreads, threadid

include("taylor_fit_convolution.jl")

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

function logspace_positive(xmin::Float64, xmax::Float64, n::Int)
    @assert 0 < xmin < xmax
    @assert n >= 2
    return collect(exp.(range(log(xmin), log(xmax); length=n)))
end

function maybe_logspace_positive(xmin::Float64, xmax::Float64, n::Int)
    if n == 0
        return Float64[]
    end
    return Float64.(logspace_positive(xmin, xmax, n))
end

@inline function overlap_entries(
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
    h11::ComplexF64,
    h12::ComplexF64,
    h21::ComplexF64,
    h22::ComplexF64,
)
    return g11 * conj(h11) + g12 * conj(h12) + g21 * conj(h21) + g22 * conj(h22)
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
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
)
    @inbounds for idx in eachindex(k_values)
        h11, h12, h21, h22 = monitored_retarded_green_entries(ω, q - k_values[idx], r, Γ, η)
        partial_k[tid, idx] += overlap_entries(g11, g12, g21, g22, h11, h12, h21, h22)
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
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
)
    @inbounds for idx in eachindex(Ω_values)
        h11, h12, h21, h22 = monitored_retarded_green_entries(ω - Ω_values[idx], q, r, Γ, η)
        partial_Ω[tid, idx] += overlap_entries(g11, g12, g21, g22, h11, h12, h21, h22)
    end
    return nothing
end

function direct_lines_for_gamma(
    Γ::Float64,
    ωs::Vector{Float64},
    dω::Float64,
    qs::Vector{Float64},
    k_values::Vector{Float64},
    Ω_values::Vector{Float64};
    r::Float64=1.0,
    η::Float64=1e-4,
)
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

        @inbounds for ωraw in ωs
            ω = ωraw
            g11, g12, g21, g22 = monitored_retarded_green_entries(ω, q, r, Γ, η)
            base_local += overlap_entries(g11, g12, g21, g22, g11, g12, g21, g22)
            add_k_contributions!(partial_k, tid, k_values, ω, q, r, Γ, η, g11, g12, g21, g22)
            add_omega_contributions!(partial_Ω, tid, Ω_values, ω, q, r, Γ, η, g11, g12, g21, g22)
        end

        partial_base[tid] += base_local
    end

    prefactor = dω / (2π * length(qs))
    Π00 = prefactor * sum(partial_base)
    Πk = prefactor .* vec(sum(partial_k; dims=1))
    ΠΩ = prefactor .* vec(sum(partial_Ω; dims=1))
    return Π00, Πk, ΠΩ
end

function save_direct_line_gamma(
    output_dir::AbstractString,
    Γ::Float64,
    Π00::ComplexF64,
    k_values::AbstractVector{<:Real},
    Πk::AbstractVector{ComplexF64},
    Ω_values::AbstractVector{<:Real},
    ΠΩ::AbstractVector{ComplexF64},
)
    filepath = joinpath(output_dir, "gamma_$(gamma_token(Γ)).jld2")
    jldsave(
        filepath;
        Γ,
        Π00,
        k_values=collect(Float64.(k_values)),
        Πk=collect(ComplexF64.(Πk)),
        Ω_values=collect(Float64.(Ω_values)),
        ΠΩ=collect(ComplexF64.(ΠΩ)),
    )
    return filepath
end

function collect_small_argument_direct_lines(;
    Nω::Int=4096,
    Nq::Int=2048,
    ωmax::Float64=320.0,
    r::Float64=1.0,
    η::Float64=1e-4,
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    kmin::Float64=5e-4,
    kmax::Float64=2e-2,
    nk_save::Int=20,
    Ωmin::Float64=5e-4,
    Ωmax::Float64=2e-2,
    nΩ_save::Int=20,
    output_dir::AbstractString="data/small_argument_direct_lines",
)
    resolved_Γs = collect(Float64.(Γs))
    mkpath(output_dir)

    ωs, dω = midpoint_frequency_grid(Nω, ωmax)
    qs, dq = uniform_momentum_grid(Nq)
    k_values = maybe_logspace_positive(kmin, kmax, nk_save)
    Ω_values = maybe_logspace_positive(Ωmin, Ωmax, nΩ_save)

    jldsave(
        joinpath(output_dir, "metadata.jld2");
        Nω,
        Nq,
        ωmax,
        dω,
        dq,
        r,
        η,
        Γs=resolved_Γs,
        k_values,
        Ω_values,
        nthreads=nthreads(),
    )

    saved_files = String[]
    for Γ in resolved_Γs
        println("collecting direct lines for Γ=", Γ, " with Nω=", Nω, ", Nq=", Nq, ", ωmax=", ωmax)
        Π00, Πk, ΠΩ = direct_lines_for_gamma(Γ, ωs, dω, qs, k_values, Ω_values; r=r, η=η)
        filepath = save_direct_line_gamma(output_dir, Γ, Π00, k_values, Πk, Ω_values, ΠΩ)
        println("saved ", filepath)
        push!(saved_files, filepath)
    end

    jldsave(joinpath(output_dir, "index.jld2"); output_dir, Γs=resolved_Γs, saved_files)
    return (output_dir=output_dir, Γs=resolved_Γs, saved_files=saved_files)
end

main(; kwargs...) = collect_small_argument_direct_lines(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
