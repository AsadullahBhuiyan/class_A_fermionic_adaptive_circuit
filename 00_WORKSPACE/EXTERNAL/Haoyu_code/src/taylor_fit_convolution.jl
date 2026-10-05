using JLD2
using LinearAlgebra
using Base.Threads: @spawn, nthreads

include("convolution.jl")
include("quadratic_kernel.jl")

const TAYLOR_FIT_RESULTS_FILE = "taylor_fit_convolution_results.jld2"
const TAYLOR_COEFFICIENT_NAMES = ["C", "C_0", "C_1", "C_00", "C_01", "C_11"]
const MEMORY_BUDGET_BYTES = 40 * 1024^3

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

default_results_file(::Val{:fft}) = TAYLOR_FIT_RESULTS_FILE
default_results_file(::Val{:tail}) = TAYLOR_FIT_RESULTS_FILE

convolution_method_name(::Val{:fft}) = "fft"
convolution_method_name(::Val{:tail}) = "tail"

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

@inline function monitored_retarded_green_entries(ω::Float64, k::Float64, r::Float64, Γ::Float64, η::Float64)
    z = ComplexF64(ω, η)
    sk = sin(k)
    ck = cos(k)
    mass = r - ck
    Σ = Σbulk(z, k, r)
    shift = im * Γ / 2

    a = z - mass - Σ / 2 + shift
    b = -sk + Σ / 2
    d = z + mass - Σ / 2 + shift
    detinv = inv(a * d - b * b)

    return d * detinv, -b * detinv, -b * detinv, a * detinv
end

function build_retarded_pair!(
    data1::AbstractArray{ComplexF64,4},
    data2::AbstractArray{ComplexF64,4},
    ωs::AbstractVector,
    ks::AbstractVector,
    active_indices::AbstractVector;
    r::Float64=1.3,
    Γ::Float64=0.5,
    η::Float64=0.001,
)
    @inbounds for j in eachindex(ks)
        k = ks[j]
        for i in active_indices
            g11, g12, g21, g22 = monitored_retarded_green_entries(ωs[i], k, r, Γ, η)
            data1[i, j, 1, 1] = g11
            data1[i, j, 1, 2] = g12
            data1[i, j, 2, 1] = g21
            data1[i, j, 2, 2] = g22
            data2[i, j, 1, 1] = g11
            data2[i, j, 1, 2] = g12
            data2[i, j, 2, 1] = g21
            data2[i, j, 2, 2] = g22
        end
    end
    return nothing
end

trace_overlap(x, y) = x[1, 1] * conj(y[1, 1]) + x[1, 2] * conj(y[1, 2]) + x[2, 1] * conj(y[2, 1]) + x[2, 2] * conj(y[2, 2])
trace_overlap_flat(data1_flat, i1, j1, data2_flat, i2, j2) =
    data1_flat[i1, j1, 1] * conj(data2_flat[i2, j2, 1]) +
    data1_flat[i1, j1, 2] * conj(data2_flat[i2, j2, 2]) +
    data1_flat[i1, j1, 3] * conj(data2_flat[i2, j2, 3]) +
    data1_flat[i1, j1, 4] * conj(data2_flat[i2, j2, 4])

signed_k_grid(ks::AbstractVector) = [k <= π ? k : k - 2π for k in ks]
sin_k_grid(ks::AbstractVector) = sin.(ks)

format_gib(bytes::Integer) = round(bytes / 1024^3; digits=2)

function estimated_thread_working_bytes(total_Nω::Int, Nk::Int)
    # Explicit local arrays: two 2x2 payload inputs plus one scalar output grid.
    explicit_arrays = 16 * total_Nω * Nk * (4 + 4 + 1)
    # FFT planning/work buffers add noticeable overhead; keep a conservative 2x margin.
    return 2 * explicit_arrays
end

function estimated_shared_bytes(Nω::Int, Nk::Int, NΓ::Int)
    return 16 * Nω * Nk * NΓ
end

function capped_worker_count(Nω::Int, total_Nω::Int, Nk::Int, NΓ::Int; memory_budget_bytes::Int=MEMORY_BUDGET_BYTES)
    per_worker = estimated_thread_working_bytes(total_Nω, Nk)
    shared = estimated_shared_bytes(Nω, Nk, NΓ)
    available = max(0, memory_budget_bytes - shared)
    max_workers_by_memory = max(1, available ÷ max(per_worker, 1))
    worker_count = max(1, min(nthreads(), NΓ, Int(max_workers_by_memory)))
    return (
        worker_count=worker_count,
        per_worker_bytes=per_worker,
        shared_bytes=shared,
        estimated_total_bytes=shared + worker_count * per_worker,
    )
end

function run_convolution!(
    ::Val{:fft},
    out::AbstractMatrix{ComplexF64},
    gr1::AbstractArray{ComplexF64,4},
    gr2::AbstractArray{ComplexF64,4},
    dω::Float64,
    ωs::AbstractVector,
    active_indices::AbstractVector,
    tailindices::AbstractVector,
    ϵ::Float64,
    tailorder::Int,
)
    return convolve_RRc_notail!(FlatBinary(trace_overlap_flat), out, gr1, gr2, dω)
end

function run_convolution!(
    ::Val{:tail},
    out::AbstractMatrix{ComplexF64},
    gr1::AbstractArray{ComplexF64,4},
    gr2::AbstractArray{ComplexF64,4},
    dω::Float64,
    ωs::AbstractVector,
    active_indices::AbstractVector,
    tailindices::AbstractVector,
    ϵ::Float64,
    tailorder::Int,
)
    return convolve_RRC_withtail!(
        FlatBinary(trace_overlap_flat),
        out,
        gr1,
        gr2,
        ωs,
        tailindices,
        ϵ;
        tailorder=tailorder,
        physicalindices=active_indices,
    )
end

function nearest_zero_indices(values::AbstractVector, npoints::Int)
    @assert 1 <= npoints <= length(values) "requested fit window is out of bounds"
    order = sortperm(abs.(values))
    selected = order[1:npoints]
    return sort(selected; by=i -> values[i])
end

function build_taylor_design(Ωfit::AbstractVector, sin_kfit::AbstractVector, Γ::Float64)
    nrows = length(Ωfit) * length(sin_kfit)
    design = Matrix{Float64}(undef, 2 * nrows, 6)
    γhalf = Γ / 2
    inv_γhalf = inv(γhalf)
    inv_γhalf_sq = inv_γhalf^2
    row = 1

    @inbounds for sk in sin_kfit
        for Ω in Ωfit
            real_row = row
            imag_row = row + nrows

            design[real_row, 1] = 1.0
            design[real_row, 2] = 0.0
            design[real_row, 3] = 0.0
            design[real_row, 4] = -Ω^2 * inv_γhalf_sq
            design[real_row, 5] = Ω * sk * inv_γhalf_sq
            design[real_row, 6] = -sk^2 * inv_γhalf_sq

            design[imag_row, 1] = 0.0
            design[imag_row, 2] = Ω * inv_γhalf
            design[imag_row, 3] = -sk * inv_γhalf
            design[imag_row, 4] = 0.0
            design[imag_row, 5] = 0.0
            design[imag_row, 6] = 0.0

            row += 1
        end
    end

    return design
end

function evaluate_taylor_model(
    coefficients::AbstractVector,
    Ω::Float64,
    sin_k::Float64,
    Γ::Float64,
)
    γhalf = Γ / 2
    inv_γhalf = inv(γhalf)
    inv_γhalf_sq = inv_γhalf^2
    return coefficients[1] +
           im * coefficients[2] * Ω * inv_γhalf -
           im * coefficients[3] * sin_k * inv_γhalf -
           (
        coefficients[4] * Ω^2 +
        coefficients[6] * sin_k^2 -
        coefficients[5] * Ω * sin_k
    ) * inv_γhalf_sq
end

function fit_taylor_coefficients(
    conv_phys::AbstractArray{ComplexF64,3},
    ω_phys::AbstractVector,
    k_signed::AbstractVector,
    sin_k::AbstractVector;
    Γs::AbstractVector,
    NΩ_fit::Int,
    Nk_fit::Int,
)
    fit_ω_indices = nearest_zero_indices(ω_phys, NΩ_fit)
    fit_k_indices = nearest_zero_indices(k_signed, Nk_fit)
    Ωfit = ω_phys[fit_ω_indices]
    kfit = k_signed[fit_k_indices]
    sin_kfit = sin_k[fit_k_indices]

    NΓ = size(conv_phys, 3)
    @assert length(Γs) == NΓ "Γs must match the third dimension of conv_phys"
    coefficients = Matrix{Float64}(undef, 6, NΓ)
    fit_residuals = Vector{Float64}(undef, NΓ)

    for γidx in 1:NΓ
        Γ = Γs[γidx]
        design = build_taylor_design(Ωfit, sin_kfit, Γ)
        @assert length(Ωfit) * length(sin_kfit) >= size(design, 2) "fit window must contain at least 6 points"
        design_factor = qr(design)
        y = vec(@view conv_phys[fit_ω_indices, fit_k_indices, γidx])
        yscaled = Γ .* y
        yfit = vcat(real.(yscaled), imag.(yscaled))
        coefficients[:, γidx] = design_factor \ yfit

        fit_error = similar(y)
        row = 1
        @inbounds for sk in sin_kfit
            for Ω in Ωfit
                fit_error[row] = yscaled[row] - evaluate_taylor_model(@view(coefficients[:, γidx]), Ω, sk, Γ)
                row += 1
            end
        end
        fit_residuals[γidx] = norm(fit_error) / max(norm(yscaled), eps(Float64))
    end

    return (
        coefficients=coefficients,
        fit_residuals=fit_residuals,
        fit_ω_indices=fit_ω_indices,
        fit_k_indices=fit_k_indices,
        Ωfit=Ωfit,
        kfit=kfit,
        sin_kfit=sin_kfit,
    )
end

function save_results(
    filepath::AbstractString,
    Nω::Int,
    Nk::Int,
    Γs::AbstractVector,
    ωs::AbstractVector,
    ω_phys::AbstractVector,
    ks::AbstractVector,
    k_signed::AbstractVector,
    sin_k::AbstractVector,
    active_indices::AbstractVector,
    fit_ω_indices::AbstractVector,
    fit_k_indices::AbstractVector,
    conv_phys::AbstractArray{ComplexF64,3},
    coefficients::AbstractMatrix,
    fit_residuals::AbstractVector,
    params::NamedTuple,
)
    jldsave(
        filepath;
        Nω,
        Nk,
        Γs,
        ωs,
        ω_phys,
        ks,
        k_signed,
        sin_k,
        active_indices=collect(active_indices),
        fit_ω_indices=collect(fit_ω_indices),
        fit_k_indices=collect(fit_k_indices),
        conv_phys,
        coefficients,
        coefficient_names=TAYLOR_COEFFICIENT_NAMES,
        fit_residuals,
        params,
    )
end
#128,2^14
function taylor_fit_main(
    method::Val=Val(:tail);
    Nk::Int=256,
    Nω::Int=2^17,
    dω::Float64=(2π / Nk)^2,
    r::Float64=1.0,
    η::Float64=1e-4,
    Γs::AbstractVector=[0.1; 0.5; 1.0:10.0; 12.0:2:20.0],
    ϵ::Float64=0.3,
    tailorder::Int=1,
    NΩ_fit::Int=9,
    Nk_fit::Int=9,
    results_file::AbstractString=default_results_file(method),
)
    @assert Nω > 0 "Nω must be positive"
    @assert Nk > 0 "Nk must be positive"
    @assert NΩ_fit > 0 "NΩ_fit must be positive"
    @assert Nk_fit > 0 "Nk_fit must be positive"
    @assert NΩ_fit * Nk_fit >= 6 "fit window must contain at least 6 points"

    Γs = collect(Float64.(Γs))
    NΓ = length(Γs)
    @assert NΓ > 0 "Γs must be non-empty"

    ωs = build_frequency_grid(Nω, dω)
    total_Nω = length(ωs)
    effective_nus = padded_fft_indices(Nω)
    active_indices = effective_frequency_indices(effective_nus, total_Nω)
    active_order = sortperm(@view ωs[active_indices])
    active_indices = active_indices[active_order]
    ω_phys = ωs[active_indices]
    tailindices = default_tailindices(effective_nus, total_Nω, tailorder)

    ks = collect(2π .* (0:(Nk-1)) ./ Nk)
    k_signed = signed_k_grid(ks)
    sin_k = sin_k_grid(ks)

    @assert NΩ_fit <= length(ω_phys) "NΩ_fit exceeds the number of physical frequencies"
    @assert Nk_fit <= length(k_signed) "Nk_fit exceeds the number of k points"

    conv_phys = Array{ComplexF64}(undef, length(ω_phys), Nk, NΓ)
    worker_memory = capped_worker_count(Nω, total_Nω, Nk, NΓ)
    nworkers = worker_memory.worker_count
    println("computing ", NΓ, " Γ points using ", nworkers, " worker(s) out of ", nthreads(), " Julia thread(s)")
    println(
        "estimated memory: shared=",
        format_gib(worker_memory.shared_bytes),
        " GiB, per worker=",
        format_gib(worker_memory.per_worker_bytes),
        " GiB, total=",
        format_gib(worker_memory.estimated_total_bytes),
        " GiB",
    )

    @sync for worker_idx in 1:nworkers
        @spawn begin
            gr1 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
            gr2 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
            out_fft_local = zeros(ComplexF64, total_Nω, Nk)
            for γidx in worker_idx:nworkers:NΓ
                Γ = Γs[γidx]
                fill!(gr1, 0)
                fill!(gr2, 0)
                build_retarded_pair!(gr1, gr2, ωs, ks, active_indices; r=r, Γ=Γ, η=η)
                run_convolution!(method, out_fft_local, gr1, gr2, dω, ωs, active_indices, tailindices, ϵ, tailorder)
                @views conv_phys[:, :, γidx] .= out_fft_local[active_indices, :]
            end
        end
    end

    fit = fit_taylor_coefficients(
        conv_phys,
        ω_phys,
        k_signed,
        sin_k;
        Γs=Γs,
        NΩ_fit=NΩ_fit,
        Nk_fit=Nk_fit,
    )

    params = (
        Nω=Nω,
        total_Nω=total_Nω,
        Nk=Nk,
        dω=dω,
        r=r,
        η=η,
        ϵ=ϵ,
        tailorder=tailorder,
        NΓ=NΓ,
        Γs=Γs,
        NΩ_fit=NΩ_fit,
        Nk_fit=Nk_fit,
        effective_nus=effective_nus,
        convolution_method=convolution_method_name(method),
        tailindices=tailindices,
        nworkers=nworkers,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
        estimated_shared_bytes=worker_memory.shared_bytes,
        estimated_per_worker_bytes=worker_memory.per_worker_bytes,
        estimated_total_bytes=worker_memory.estimated_total_bytes,
    )

    save_results(
        results_file,
        Nω,
        Nk,
        Γs,
        ωs,
        ω_phys,
        ks,
        k_signed,
        sin_k,
        active_indices,
        fit.fit_ω_indices,
        fit.fit_k_indices,
        conv_phys,
        fit.coefficients,
        fit.fit_residuals,
        params,
    )

    println("saved results to ", results_file)

    return (
        Γs=Γs,
        ωs=ωs,
        ω_phys=ω_phys,
        ks=ks,
        k_signed=k_signed,
        sin_k=sin_k,
        active_indices=active_indices,
        fit_ω_indices=fit.fit_ω_indices,
        fit_k_indices=fit.fit_k_indices,
        conv_phys=conv_phys,
        coefficients=fit.coefficients,
        coefficient_names=TAYLOR_COEFFICIENT_NAMES,
        fit_residuals=fit.fit_residuals,
        params=params,
    )
end

main(method::Val; kwargs...) = taylor_fit_main(method; kwargs...)
main(; kwargs...) = taylor_fit_main(; kwargs...)

running_in_vscode_repl() = isinteractive() && isdefined(Main, :VSCodeServer)

if abspath(PROGRAM_FILE) == (@__FILE__) || running_in_vscode_repl()
    Base.invokelatest(taylor_fit_main)
    nothing
end
