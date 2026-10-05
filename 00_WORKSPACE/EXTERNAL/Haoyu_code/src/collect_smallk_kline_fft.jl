using JLD2
using Printf

include("taylor_fit_convolution.jl")

function smallk_output_dir(Nω::Int, Nk::Int, ωmax::Float64)
    tag = replace(@sprintf("%.3f", ωmax), "." => "p")
    return "smallk_kline_fft_Nw$(Nω)_Nk$(Nk)_wmax$(tag)"
end

function save_smallk_metadata(output_dir::AbstractString, metadata::NamedTuple)
    jldsave(joinpath(output_dir, "metadata.jld2"); metadata...)
    return nothing
end

function save_smallk_gamma(
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

function collect_smallk_kline_fft(;
    method::Val=Val(:fft),
    Nk::Int=2048,
    Nω::Int=12288,
    ωmax_target::Float64=320.0,
    dω::Union{Nothing, Float64}=nothing,
    r::Float64=1.0,
    η::Float64=1e-4,
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    ϵ::Float64=0.3,
    tailorder::Int=1,
    kmax_save::Float64=0.2,
    output_dir::Union{Nothing, AbstractString}=nothing,
)
    Γs = collect(Float64.(Γs))
    resolved_dω = isnothing(dω) ? (2 * ωmax_target / Nω) : Float64(dω)
    ωs = build_frequency_grid(Nω, resolved_dω)
    total_Nω = length(ωs)
    effective_nus = padded_fft_indices(Nω)
    active_indices = effective_frequency_indices(effective_nus, total_Nω)
    active_order = sortperm(@view ωs[active_indices])
    active_indices = active_indices[active_order]
    ω_phys = ωs[active_indices]
    ω0 = findfirst(==(0.0), ω_phys)
    @assert !isnothing(ω0) "ω=0 must lie on the physical grid"
    tailindices = default_tailindices(effective_nus, total_Nω, tailorder)

    ks = collect(2π .* (0:(Nk - 1)) ./ Nk)
    k_signed = signed_k_grid(ks)
    positive_indices = findall(k -> 0 < k <= kmax_save, k_signed)
    k_positive = k_signed[positive_indices]

    ωband = maximum(abs.(ω_phys))
    @assert ωband > maximum(Γs) "physical frequency bandwidth must exceed the largest Γ"

    resolved_output_dir = isnothing(output_dir) ? smallk_output_dir(Nω, Nk, ωband) : String(output_dir)
    mkpath(resolved_output_dir)

    metadata = (
        method=convolution_method_name(method),
        Nω=Nω,
        total_Nω=total_Nω,
        Nk=Nk,
        dω=resolved_dω,
        r=r,
        η=η,
        ϵ=ϵ,
        tailorder=tailorder,
        Γs=Γs,
        ωs=ωs,
        ω_phys=ω_phys,
        ω_zero_index=ω0,
        ωband=ωband,
        ks=ks,
        k_signed=k_signed,
        positive_indices=collect(positive_indices),
        k_positive=k_positive,
        kmax_save=kmax_save,
        active_indices=collect(active_indices),
        tailindices=collect(tailindices),
    )
    save_smallk_metadata(resolved_output_dir, metadata)

    gr1 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    gr2 = zeros(ComplexF64, total_Nω, Nk, 2, 2)
    out_fft_local = zeros(ComplexF64, total_Nω, Nk)

    saved_files = String[]
    for Γ in Γs
        println("computing small-k FFT line for Γ=", Γ, " with ωband=", ωband, ", dω=", resolved_dω)
        fill!(gr1, 0)
        fill!(gr2, 0)
        fill!(out_fft_local, 0)
        build_retarded_pair!(gr1, gr2, ωs, ks, active_indices; r=r, Γ=Γ, η=η)
        run_convolution!(method, out_fft_local, gr1, gr2, resolved_dω, ωs, active_indices, tailindices, ϵ, tailorder)
        kline = copy(@view out_fft_local[active_indices[ω0], :])
        kline_positive = copy(@view kline[positive_indices])
        filepath = save_smallk_gamma(
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

    jldsave(joinpath(resolved_output_dir, "index.jld2"); output_dir=resolved_output_dir, Γs, saved_files, ωband=ωband, dω=resolved_dω)
    return (
        output_dir=resolved_output_dir,
        Γs=Γs,
        ωband=ωband,
        dω=resolved_dω,
        saved_files=saved_files,
    )
end

main(; kwargs...) = collect_smallk_kline_fft(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
