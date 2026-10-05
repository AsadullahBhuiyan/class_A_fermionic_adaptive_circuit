using JLD2
using LinearAlgebra

include("convolution.jl")
include("quadratic_kernel.jl")

const BENCHMARK_RESULTS_FILE = "benchmark_convolution_results.jld2"

function build_frequency_grid(Nω::Int, dω::Float64; padding_factor::Int=4)
    total_Nω = padding_factor * Nω
    ωs = Vector{Float64}(undef, total_Nω)

    @inbounds for i in 1:total_Nω
        ν = padded_fft_index(i, total_Nω)
        ωs[i] = dω * ν
    end

    return ωs
end

function effective_frequency_indices(effective_nus::AbstractVector, total_Nω::Int)
    return sort(mod1.(effective_nus .+ 1, total_Nω))
end

function build_retarded_grid(
    ωs::AbstractVector,
    ks::AbstractVector,
    active_indices::AbstractVector;
    r::Float64=1.3,
    Γ::Float64=0.5,
    η::Float64=0.001,
)
    data = zeros(ComplexF64, length(ωs), length(ks), 2, 2)
    @inbounds for j in eachindex(ks)
        for i in active_indices
            z = ComplexF64(ωs[i], η)
            sk = sin(ks[j])
            ck = cos(ks[j])
            mass = r - ck
            Σ = Σbulk(z, ks[j], r)
            shift = im * Γ / 2
            a = z - mass - Σ / 2 + shift
            b = -sk + Σ / 2
            d = z + mass - Σ / 2 + shift
            detinv = inv(a * d - b * b)
            data[i, j, 1, 1] = d * detinv
            data[i, j, 1, 2] = -b * detinv
            data[i, j, 2, 1] = -b * detinv
            data[i, j, 2, 2] = a * detinv
        end
    end
    return data
end

trace_overlap(x, y) = x[1, 1] * conj(y[1, 1]) + x[1, 2] * conj(y[1, 2]) + x[2, 1] * conj(y[2, 1]) + x[2, 2] * conj(y[2, 2])
trace_overlap_flat(data1_flat, i1, j1, data2_flat, i2, j2) =
    data1_flat[i1, j1, 1] * conj(data2_flat[i2, j2, 1]) +
    data1_flat[i1, j1, 2] * conj(data2_flat[i2, j2, 2]) +
    data1_flat[i1, j1, 3] * conj(data2_flat[i2, j2, 3]) +
    data1_flat[i1, j1, 4] * conj(data2_flat[i2, j2, 4])

function direct_convolve_rrc!(
    binary!::F,
    out::AbstractArray,
    data1::AbstractArray,
    data2::AbstractArray,
    dω::Float64,
) where {F}
    return convolve_RRc_direct!(binary!, out, data1, data2, dω)
end

function direct_convolve_rrc!(
    out::AbstractMatrix,
    data1::AbstractMatrix,
    data2::AbstractMatrix,
    dω::Float64,
)
    return convolve_RRc_direct!(out, data1, data2, dω)
end

function default_tailindices(effective_nus::AbstractVector, total_Nω::Int, tailorder::Int)
    positive_support = [ν for ν in effective_nus if ν > 0]
    negative_support = [ν for ν in effective_nus if ν < 0]
    if isempty(positive_support) || isempty(negative_support)
        fallback_nus = effective_nus[1:(tailorder+1)]
        return sort(mod1.(fallback_nus .+ 1, total_Nω))
    end
    nfit = max(tailorder + 1, min(10, min(length(positive_support), length(negative_support))))
    selected_pos = positive_support[(end-nfit+1):end]
    selected_neg = negative_support[1:nfit]
    selected_nus = vcat(selected_pos, selected_neg)
    return sort(mod1.(selected_nus .+ 1, total_Nω))
end

function nearest_k_indices(ks::AbstractVector, targets::AbstractVector)
    return unique([argmin(abs.(ks .- target)) for target in targets])
end

direct_operation_count(Nω_total::Int, Nk::Int) = Int128(Nω_total) * Nk * Nω_total * Nk

function global_disagreement(x, y)
    xnorm = norm(x)
    ynorm = norm(y)
    if iszero(xnorm) || iszero(ynorm)
        return 0.0
    end
    overlap = clamp(abs(dot(vec(x), vec(y))) / (xnorm * ynorm), 0.0, 1.0)
    return 1 - overlap
end

function trimmed_physical_indices(active_indices::AbstractVector, trim_percent::Float64)
    @assert 0.0 <= trim_percent < 0.5 "trim_percent must lie in [0, 0.5)"
    nactive = length(active_indices)
    trim_count = floor(Int, trim_percent * nactive)
    if 2 * trim_count >= nactive
        return collect(active_indices)
    end
    return collect(active_indices[(trim_count + 1):(nactive - trim_count)])
end

function report_errors(
    out_direct,
    out_fft::AbstractArray,
    out_tail::AbstractArray,
    slice_indices::AbstractVector,
    active_indices::AbstractVector,
    trim_percent::Float64,
)
    error_indices = trimmed_physical_indices(active_indices, trim_percent)
    physical_fft = @view out_fft[error_indices, :]
    physical_tail = @view out_tail[error_indices, :]
    tail_vs_fft_physical_disagreement = global_disagreement(physical_tail, physical_fft)

    println("trimmed physical-grid global disagreement (tail vs fft): ", tail_vs_fft_physical_disagreement)
    println("error estimate trim percent: ", trim_percent)

    if out_direct === nothing
        println("direct convolution skipped; no direct-reference error report available")
        return
    end

    fft_disagreement = global_disagreement(out_direct, out_fft)
    tail_disagreement = global_disagreement(out_direct, out_tail)

    println("global disagreement (direct vs fft):  ", fft_disagreement)
    println("global disagreement (direct vs tail): ", tail_disagreement)
    fft_physical_disagreement = global_disagreement(@view(out_direct[error_indices, :]), physical_fft)
    tail_physical_disagreement = global_disagreement(@view(out_direct[error_indices, :]), physical_tail)
    println("trimmed physical-grid global disagreement (direct vs fft):  ", fft_physical_disagreement)
    println("trimmed physical-grid global disagreement (direct vs tail): ", tail_physical_disagreement)

    for idx in slice_indices
        fft_slice_disagreement = global_disagreement(@view(out_direct[:, idx]), @view(out_fft[:, idx]))
        tail_slice_disagreement = global_disagreement(@view(out_direct[:, idx]), @view(out_tail[:, idx]))
        fft_slice_physical_disagreement = global_disagreement(@view(out_direct[error_indices, idx]), @view(out_fft[error_indices, idx]))
        tail_slice_physical_disagreement = global_disagreement(@view(out_direct[error_indices, idx]), @view(out_tail[error_indices, idx]))
        fft_tail_slice_physical_disagreement = global_disagreement(@view(out_tail[error_indices, idx]), @view(out_fft[error_indices, idx]))
        println(
            "k-column ",
            idx,
            ": fft disagreement=",
            fft_slice_disagreement,
            ", tail disagreement=",
            tail_slice_disagreement,
            ", trimmed physical fft disagreement=",
            fft_slice_physical_disagreement,
            ", trimmed physical tail disagreement=",
            tail_slice_physical_disagreement,
            ", trimmed physical tail-vs-fft disagreement=",
            fft_tail_slice_physical_disagreement,
        )
    end
end

function save_results(
    filepath::AbstractString,
    ωs::AbstractVector,
    ks::AbstractVector,
    out_direct,
    out_fft::AbstractArray,
    out_tail::AbstractArray,
    kslice_indices::AbstractVector,
    tailindices::AbstractVector,
    params::NamedTuple,
)
    jldsave(
        filepath;
        ωs,
        ks,
        out_direct,
        out_fft,
        out_tail,
        kslice_indices=collect(kslice_indices),
        tailindices=collect(tailindices),
        params,
    )
end

function benchmark_main(;
    Nω::Int=8192,
    Nk::Int=32,
    dω::Float64=0.02,
    r::Float64=1.3,
    Γ::Float64=0.5,
    η::Float64=0.001,
    ϵ::Float64=0.3,
    tailorder::Int=0,
    ktargets::AbstractVector=[0.0],
    results_file::AbstractString=BENCHMARK_RESULTS_FILE,
    run_direct::Bool=false,
    direct_budget::Integer=direct_operation_count(1, Nk),
    error_trim_percent::Float64=0.1,
)
    ωs = build_frequency_grid(Nω, dω)
    total_Nω = length(ωs)
    effective_nus = padded_fft_indices(Nω)
    active_indices = effective_frequency_indices(effective_nus, total_Nω)
    ks = collect(2π .* (0:(Nk-1)) ./ Nk)
    tailindices = default_tailindices(effective_nus, total_Nω, tailorder)
    kslice_indices = nearest_k_indices(ks, ktargets)

    GR = build_retarded_grid(ωs, ks, active_indices; r=r, Γ=Γ, η=η)

    out_direct = nothing
    out_fft = zeros(ComplexF64, total_Nω, Nk)
    out_tail = zeros(ComplexF64, total_Nω, Nk)

    direct_ops = direct_operation_count(total_Nω, Nk)
    if run_direct || direct_ops <= direct_budget
        out_direct = zeros(ComplexF64, total_Nω, Nk)
        direct_convolve_rrc!(FlatBinary(trace_overlap_flat), out_direct, GR, GR, dω)
    else
        println("skipping direct convolution: estimated operation count = ", direct_ops)
        println("set `run_direct=true` in `benchmark_main` to force it")
    end

    convolve_RRc_notail!(FlatBinary(trace_overlap_flat), out_fft, copy(GR), copy(GR), dω)
    convolve_RRC_withtail!(
        FlatBinary(trace_overlap_flat),
        out_tail,
        copy(GR),
        copy(GR),
        ωs,
        tailindices,
        ϵ;
        tailorder=tailorder,
        physicalindices=active_indices,
    )

    report_errors(out_direct, out_fft, out_tail, kslice_indices, active_indices, error_trim_percent)

    params = (
        Nω=Nω,
        total_Nω=total_Nω,
        Nk=Nk,
        dω=dω,
        r=r,
        Γ=Γ,
        η=η,
        ϵ=ϵ,
        tailorder=tailorder,
        ktargets=ktargets,
        effective_nus=effective_nus,
        active_indices=active_indices,
        run_direct=run_direct,
        direct_budget=direct_budget,
        direct_ops=direct_ops,
        error_trim_percent=error_trim_percent,
    )

    save_results(
        results_file,
        ωs,
        ks,
        out_direct,
        out_fft,
        out_tail,
        kslice_indices,
        tailindices,
        params,
    )

    println("saved results to ", results_file)

    return (
        ωs=ωs,
        ks=ks,
        out_direct=out_direct,
        out_fft=out_fft,
        out_tail=out_tail,
        kslice_indices=kslice_indices,
        tailindices=tailindices,
        params=params,
    )
end

main(; kwargs...) = benchmark_main(; kwargs...)

running_in_vscode_repl() = isinteractive() && isdefined(Main, :VSCodeServer)

if abspath(PROGRAM_FILE) == (@__FILE__) || running_in_vscode_repl()
    Base.invokelatest(benchmark_main)
    nothing
end
