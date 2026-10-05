using JLD2
using Printf

include("taylor_fit_convolution.jl")

function touching_window_mixed_curvature(
    scan_dir::AbstractString,
    Γ::Float64;
    pcut::Float64=pi / 2,
    ωcut::Float64=3.0,
    cutoff_power::Int=8,
)
    meta = load(joinpath(scan_dir, "metadata.jld2"))
    ωs = meta["ωs"]
    ks = meta["ks"]
    active_indices = meta["active_indices"]
    tailindices = meta["tailindices"]
    dω = meta["dω"]
    η = meta["η"]
    ϵ = meta["ϵ"]
    tailorder = meta["tailorder"]

    total_Nω = length(ωs)
    Nk = length(ks)
    gr1 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    gr2 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    out = zeros(ComplexF64, total_Nω, Nk)

    build_retarded_pair!(gr1, gr2, ωs, ks, active_indices; r=1.0, Γ=Γ, η=η)

    @inbounds for j in eachindex(ks), i in active_indices
        q = ks[j]
        dq = min(abs(q - π / 2), abs(q - 3π / 2))
        dωedge = min(abs(ωs[i] - 1), abs(ωs[i] + 1))
        weight = exp(-(dq / pcut)^cutoff_power) * exp(-(dωedge / ωcut)^cutoff_power)
        for a in 1:2, b in 1:2
            gr1[i, j, a, b] *= weight
            gr2[i, j, a, b] *= weight
        end
    end

    run_convolution!(Val(:fft), out, gr1, gr2, dω, ωs, active_indices, tailindices, ϵ, tailorder)

    Ω_indices = meta["ω_save_indices"]
    k_indices = meta["k_save_indices"]
    Ω_values = meta["Ω_values"]
    k_values = meta["k_values"]
    i0 = findfirst(==(0.0), Ω_values)
    j0 = findfirst(==(0.0), k_values)
    hΩ = Ω_values[i0 + 1] - Ω_values[i0]
    hk = abs(k_values[j0 + 1] - k_values[j0])
    small = out[active_indices[Ω_indices], k_indices]

    return (
        curvature=(
            real(Γ * small[i0 + 1, j0 + 1]) -
            real(Γ * small[i0 + 1, j0 - 1]) -
            real(Γ * small[i0 - 1, j0 + 1]) +
            real(Γ * small[i0 - 1, j0 - 1])
        ) / (4 * hΩ * hk),
        hΩ=hΩ,
        hk=hk,
    )
end

function main(;
    scan_dirs::AbstractVector{<:AbstractString}=[
        "sparse_convolution_scan_mixedcheck2_Nw16384_Nk512",
        "sparse_convolution_scan_mixedcheck_Nw16384_Nk512",
    ],
    Γs::AbstractVector{<:Real}=[20.0, 40.0, 80.0],
    pcut::Float64=pi / 2,
    ωcut::Float64=3.0,
    cutoff_power::Int=8,
    output_file::AbstractString="touching_window_mixed_analysis.jld2",
)
    rows = NamedTuple[]
    for scan_dir in scan_dirs
        for Γ in Γs
            isfile(joinpath(scan_dir, "metadata.jld2")) || continue
            result = touching_window_mixed_curvature(
                scan_dir,
                Float64(Γ);
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
            )
            println(@sprintf("%s Γ=%.1f dΩk=%.12f", scan_dir, Γ, result.curvature))
            push!(rows, (
                scan_dir=String(scan_dir),
                Γ=Float64(Γ),
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
                dΩk=result.curvature,
                hΩ=result.hΩ,
                hk=result.hk,
            ))
        end
    end
    jldsave(output_file; rows)
    println("saved ", output_file)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
