using JLD2
using Printf

include("taylor_fit_convolution.jl")

function save_touching_window_gamma(
    output_dir::AbstractString,
    Γ::Float64,
    k_values::AbstractVector,
    kline::AbstractVector{ComplexF64},
    k_positive::AbstractVector,
    kline_positive::AbstractVector{ComplexF64},
)
    filepath = joinpath(output_dir, "gamma_$(replace(@sprintf("%.3f", Γ), "." => "p")).jld2")
    jldsave(
        filepath;
        Γ,
        k_values,
        kline,
        k_positive,
        kline_positive,
    )
    return filepath
end

function collect_smallk_touching_window_fft(;
    input_dir::AbstractString,
    pcut::Float64=pi / 2,
    ωcut::Float64=3.0,
    cutoff_power::Int=8,
    output_dir::Union{Nothing, AbstractString}=nothing,
)
    meta = load(joinpath(input_dir, "metadata.jld2"))
    ωs = meta["ωs"]
    ks = meta["ks"]
    active_indices = meta["active_indices"]
    tailindices = meta["tailindices"]
    dω = meta["dω"]
    η = meta["η"]
    ϵ = meta["ϵ"]
    tailorder = meta["tailorder"]
    Γs = collect(Float64.(meta["Γs"]))
    k_signed = meta["k_signed"]
    positive_indices = meta["positive_indices"]
    k_positive = meta["k_positive"]
    ω0 = meta["ω_zero_index"]

    total_Nω = length(ωs)
    Nk = length(ks)
    resolved_output_dir = isnothing(output_dir) ? "$(input_dir)_touching_window" : String(output_dir)
    mkpath(resolved_output_dir)

    metadata = (
        input_dir=String(input_dir),
        pcut=pcut,
        ωcut=ωcut,
        cutoff_power=cutoff_power,
        Γs=Γs,
        ωband=meta["ωband"],
        dω=dω,
        Nω=meta["Nω"],
        Nk=meta["Nk"],
        k_signed=k_signed,
        k_positive=k_positive,
        positive_indices=positive_indices,
        ω_zero_index=ω0,
    )
    jldsave(joinpath(resolved_output_dir, "metadata.jld2"); metadata...)

    gr1 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    gr2 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    out_fft_local = zeros(ComplexF64, total_Nω, Nk)

    saved_files = String[]
    for Γ in Γs
        println("computing touching-window small-k FFT line for Γ=", Γ, " with pcut=", pcut, ", ωcut=", ωcut)
        fill!(gr1, 0)
        fill!(gr2, 0)
        fill!(out_fft_local, 0)
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
        run_convolution!(Val(:fft), out_fft_local, gr1, gr2, dω, ωs, active_indices, tailindices, ϵ, tailorder)
        kline = copy(@view out_fft_local[active_indices[ω0], :])
        kline_positive = copy(@view kline[positive_indices])
        filepath = save_touching_window_gamma(
            resolved_output_dir,
            Γ,
            k_signed,
            kline,
            k_positive,
            kline_positive,
        )
        println("saved ", filepath)
        push!(saved_files, filepath)
    end

    jldsave(joinpath(resolved_output_dir, "index.jld2"); output_dir=resolved_output_dir, Γs, saved_files)
    return (
        output_dir=resolved_output_dir,
        Γs=Γs,
        saved_files=saved_files,
    )
end

main(; kwargs...) = collect_smallk_touching_window_fft(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    input_dir = isempty(ARGS) ? error("pass input directory") : ARGS[1]
    collect_smallk_touching_window_fft(input_dir=input_dir)
end
