using JLD2
using LinearAlgebra
using Printf

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function load_cubic_metadata(output_dir::AbstractString)
    return load(joinpath(output_dir, "metadata.jld2"))
end

function load_cubic_gamma(output_dir::AbstractString, Γ::Real)
    return load(joinpath(output_dir, "gamma_$(gamma_token(Γ)).jld2"))
end

function metadata_ωband(meta)
    if haskey(meta, "ωband")
        return Float64(meta["ωband"])
    end
    if haskey(meta, "ω_phys")
        return maximum(abs.(Vector{Float64}(meta["ω_phys"])))
    end
    error("metadata does not contain ωband or ω_phys")
end

function metadata_dω(meta)
    if haskey(meta, "dω")
        return Float64(meta["dω"])
    end
    error("metadata does not contain dω")
end

function central_index(n::Int)
    @assert isodd(n) "central finite differences require an odd external window"
    return cld(n, 2)
end

function central_derivative(values::AbstractVector{ComplexF64}, h::Float64)
    c = central_index(length(values))
    return (values[c + 1] - values[c - 1]) / (2h)
end

function central_second_derivative(values::AbstractVector{ComplexF64}, h::Float64)
    c = central_index(length(values))
    return (values[c + 1] - 2 * values[c] + values[c - 1]) / (h^2)
end

@inline function third_leg_index(center::Int, i1::Int, i2::Int)
    return 3 * center - i1 - i2
end

@inline function inbounds_leg(triangle, iΩ::Int, ik::Int)
    return 1 <= iΩ <= size(triangle, 1) && 1 <= ik <= size(triangle, 2)
end

function symmetrized_value(
    triangle::Array{ComplexF64,4},
    cΩ::Int,
    ck::Int,
    iΩ1::Int,
    ik1::Int,
    iΩ2::Int,
    ik2::Int,
)
    iΩ3 = third_leg_index(cΩ, iΩ1, iΩ2)
    ik3 = third_leg_index(ck, ik1, ik2)
    if !inbounds_leg(triangle, iΩ3, ik3)
        return nothing
    end

    legs = ((iΩ1, ik1), (iΩ2, ik2), (iΩ3, ik3))
    perms = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))
    acc = 0.0 + 0.0im
    @inbounds for (a, b) in perms
        Ωa, ka = legs[a]
        Ωb, kb = legs[b]
        if !inbounds_leg(triangle, Ωa, ka) || !inbounds_leg(triangle, Ωb, kb)
            return nothing
        end
        acc += triangle[Ωa, ka, Ωb, kb]
    end
    return acc / 6
end

function central_mixed_derivative(
    samples::NTuple{4,ComplexF64},
    hx::Float64,
    hy::Float64,
)
    fpp, fpm, fmp, fmm = samples
    return (fpp - fpm - fmp + fmm) / (4hx * hy)
end

function symmetrized_second_derivatives(
    triangle::Array{ComplexF64,4},
    Ω_values::AbstractVector{Float64},
    k_values::AbstractVector{Float64},
)
    cΩ = central_index(length(Ω_values))
    ck = central_index(length(k_values))
    ΔΩ = Ω_values[cΩ + 1] - Ω_values[cΩ]
    Δk = k_values[ck + 1] - k_values[ck]

    f00 = symmetrized_value(triangle, cΩ, ck, cΩ, ck, cΩ, ck)
    @assert !isnothing(f00)

    Ωp = symmetrized_value(triangle, cΩ, ck, cΩ + 1, ck, cΩ, ck)
    Ωm = symmetrized_value(triangle, cΩ, ck, cΩ - 1, ck, cΩ, ck)
    Ω2p = symmetrized_value(triangle, cΩ, ck, cΩ, ck, cΩ + 1, ck)
    Ω2m = symmetrized_value(triangle, cΩ, ck, cΩ, ck, cΩ - 1, ck)
    kp = symmetrized_value(triangle, cΩ, ck, cΩ, ck + 1, cΩ, ck)
    km = symmetrized_value(triangle, cΩ, ck, cΩ, ck - 1, cΩ, ck)
    k2p = symmetrized_value(triangle, cΩ, ck, cΩ, ck, cΩ, ck + 1)
    k2m = symmetrized_value(triangle, cΩ, ck, cΩ, ck, cΩ, ck - 1)

    @assert !any(isnothing, (Ωp, Ωm, Ω2p, Ω2m, kp, km, k2p, k2m))

    ΩΩ_pp = symmetrized_value(triangle, cΩ, ck, cΩ + 1, ck, cΩ + 1, ck)
    ΩΩ_pm = symmetrized_value(triangle, cΩ, ck, cΩ + 1, ck, cΩ - 1, ck)
    ΩΩ_mp = symmetrized_value(triangle, cΩ, ck, cΩ - 1, ck, cΩ + 1, ck)
    ΩΩ_mm = symmetrized_value(triangle, cΩ, ck, cΩ - 1, ck, cΩ - 1, ck)

    Ωk11_pp = symmetrized_value(triangle, cΩ, ck, cΩ + 1, ck + 1, cΩ, ck)
    Ωk11_pm = symmetrized_value(triangle, cΩ, ck, cΩ + 1, ck - 1, cΩ, ck)
    Ωk11_mp = symmetrized_value(triangle, cΩ, ck, cΩ - 1, ck + 1, cΩ, ck)
    Ωk11_mm = symmetrized_value(triangle, cΩ, ck, cΩ - 1, ck - 1, cΩ, ck)

    Ωk12_pp = symmetrized_value(triangle, cΩ, ck, cΩ + 1, ck, cΩ, ck + 1)
    Ωk12_pm = symmetrized_value(triangle, cΩ, ck, cΩ + 1, ck, cΩ, ck - 1)
    Ωk12_mp = symmetrized_value(triangle, cΩ, ck, cΩ - 1, ck, cΩ, ck + 1)
    Ωk12_mm = symmetrized_value(triangle, cΩ, ck, cΩ - 1, ck, cΩ, ck - 1)

    kk_pp = symmetrized_value(triangle, cΩ, ck, cΩ, ck + 1, cΩ, ck + 1)
    kk_pm = symmetrized_value(triangle, cΩ, ck, cΩ, ck + 1, cΩ, ck - 1)
    kk_mp = symmetrized_value(triangle, cΩ, ck, cΩ, ck - 1, cΩ, ck + 1)
    kk_mm = symmetrized_value(triangle, cΩ, ck, cΩ, ck - 1, cΩ, ck - 1)

    result = (
        center=f00,
        d2Ω1=(Ωp - 2 * f00 + Ωm) / (ΔΩ^2),
        d2Ω2=(Ω2p - 2 * f00 + Ω2m) / (ΔΩ^2),
        d2k1=(kp - 2 * f00 + km) / (Δk^2),
        d2k2=(k2p - 2 * f00 + k2m) / (Δk^2),
        dΩ1dΩ2=isnothing(ΩΩ_pp) || isnothing(ΩΩ_pm) || isnothing(ΩΩ_mp) || isnothing(ΩΩ_mm) ?
            nothing :
            central_mixed_derivative((ΩΩ_pp, ΩΩ_pm, ΩΩ_mp, ΩΩ_mm), ΔΩ, ΔΩ),
        dΩ1dk1=isnothing(Ωk11_pp) || isnothing(Ωk11_pm) || isnothing(Ωk11_mp) || isnothing(Ωk11_mm) ?
            nothing :
            central_mixed_derivative((Ωk11_pp, Ωk11_pm, Ωk11_mp, Ωk11_mm), ΔΩ, Δk),
        dΩ1dk2=isnothing(Ωk12_pp) || isnothing(Ωk12_pm) || isnothing(Ωk12_mp) || isnothing(Ωk12_mm) ?
            nothing :
            central_mixed_derivative((Ωk12_pp, Ωk12_pm, Ωk12_mp, Ωk12_mm), ΔΩ, Δk),
        dk1dk2=isnothing(kk_pp) || isnothing(kk_pm) || isnothing(kk_mp) || isnothing(kk_mm) ?
            nothing :
            central_mixed_derivative((kk_pp, kk_pm, kk_mp, kk_mm), Δk, Δk),
    )
    return result
end

function extract_summary(output_dir::AbstractString, Γ::Real; scale_power::Int=3)
    meta = load_cubic_metadata(output_dir)
    data = load_cubic_gamma(output_dir, Γ)
    Ω_values = Vector{Float64}(data["Ω_values"])
    k_values = Vector{Float64}(data["k_values"])
    triangle = data["triangle"]

    cΩ = central_index(length(Ω_values))
    ck = central_index(length(k_values))
    ΔΩ = Ω_values[cΩ + 1] - Ω_values[cΩ]
    Δk = k_values[ck + 1] - k_values[ck]

    Ω1_line = ComplexF64[triangle[i, ck, cΩ, ck] for i in eachindex(Ω_values)]
    Ω2_line = ComplexF64[triangle[cΩ, ck, i, ck] for i in eachindex(Ω_values)]
    k1_line = ComplexF64[triangle[cΩ, i, cΩ, ck] for i in eachindex(k_values)]
    k2_line = ComplexF64[triangle[cΩ, ck, cΩ, i] for i in eachindex(k_values)]

    scale = Γ^scale_power
    return (
        output_dir=output_dir,
        Γ=Γ,
        distribution_mode=meta["distribution_mode"],
        ωband=metadata_ωband(meta),
        dω=metadata_dω(meta),
        Ω_values=Ω_values,
        k_values=k_values,
        center=triangle[cΩ, ck, cΩ, ck],
        maxabs=maximum(abs.(triangle)),
        scaled_center=scale * triangle[cΩ, ck, cΩ, ck],
        scaled_maxabs=scale * maximum(abs.(triangle)),
        scaled_dΩ1=scale * central_derivative(Ω1_line, ΔΩ),
        scaled_dΩ2=scale * central_derivative(Ω2_line, ΔΩ),
        scaled_dk1=scale * central_derivative(k1_line, Δk),
        scaled_dk2=scale * central_derivative(k2_line, Δk),
        scaled_d2Ω1=scale * central_second_derivative(Ω1_line, ΔΩ),
        scaled_d2Ω2=scale * central_second_derivative(Ω2_line, ΔΩ),
        scaled_d2k1=scale * central_second_derivative(k1_line, Δk),
        scaled_d2k2=scale * central_second_derivative(k2_line, Δk),
    )
end

function extract_symmetrized_summary(output_dir::AbstractString, Γ::Real; scale_power::Int=4)
    meta = load_cubic_metadata(output_dir)
    data = load_cubic_gamma(output_dir, Γ)
    Ω_values = Vector{Float64}(data["Ω_values"])
    k_values = Vector{Float64}(data["k_values"])
    triangle = data["triangle"]
    derivs = symmetrized_second_derivatives(triangle, Ω_values, k_values)
    scale = Γ^scale_power

    return (
        output_dir=output_dir,
        Γ=Γ,
        distribution_mode=meta["distribution_mode"],
        ωband=metadata_ωband(meta),
        dω=metadata_dω(meta),
        center=derivs.center,
        scaled_center=scale * derivs.center,
        scaled_d2Ω1=scale * derivs.d2Ω1,
        scaled_d2Ω2=scale * derivs.d2Ω2,
        scaled_d2k1=scale * derivs.d2k1,
        scaled_d2k2=scale * derivs.d2k2,
        scaled_dΩ1dΩ2=isnothing(derivs.dΩ1dΩ2) ? nothing : scale * derivs.dΩ1dΩ2,
        scaled_dΩ1dk1=isnothing(derivs.dΩ1dk1) ? nothing : scale * derivs.dΩ1dk1,
        scaled_dΩ1dk2=isnothing(derivs.dΩ1dk2) ? nothing : scale * derivs.dΩ1dk2,
        scaled_dk1dk2=isnothing(derivs.dk1dk2) ? nothing : scale * derivs.dk1dk2,
    )
end

function log_slope(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    @assert length(x) == length(y) >= 2
    return (log(y[end]) - log(y[1])) / (log(x[end]) - log(x[1]))
end

function summarize_output_dir(output_dir::AbstractString; Γs::Union{Nothing, AbstractVector}=nothing, scale_power::Int=3)
    meta = load_cubic_metadata(output_dir)
    Γlist = isnothing(Γs) ? Vector{Float64}(meta["Γs"]) : collect(Float64.(Γs))
    summaries = [extract_summary(output_dir, Γ; scale_power=scale_power) for Γ in Γlist]

    println("output_dir = ", output_dir)
    println("distribution = ", meta["distribution_mode"], ", ωband = ", metadata_ωband(meta), ", dω = ", metadata_dω(meta))
    for s in summaries
        println(
            "Γ=", s.Γ,
            " center=", s.center,
            " maxabs=", s.maxabs,
            " Γ^", scale_power, "*dΩ1=", s.scaled_dΩ1,
            " Γ^", scale_power, "*dΩ2=", s.scaled_dΩ2,
            " Γ^", scale_power, "*dk1=", s.scaled_dk1,
            " Γ^", scale_power, "*dk2=", s.scaled_dk2,
        )
    end

    maxabs_values = [s.maxabs for s in summaries]
    println("rough log-slope of maxabs vs Γ = ", log_slope(Γlist, maxabs_values))
    return summaries
end

function compare_bandwidths(output_dirs::AbstractVector{<:AbstractString}, Γ::Real; scale_power::Int=3)
    summaries = [extract_summary(dir, Γ; scale_power=scale_power) for dir in output_dirs]
    sort!(summaries; by=s -> s.ωband)
    println("Γ = ", Γ, ", scale power = ", scale_power)
    for s in summaries
        println(
            "ωband=", s.ωband,
            " Γ^", scale_power, "*dΩ1=", s.scaled_dΩ1,
            " Γ^", scale_power, "*dΩ2=", s.scaled_dΩ2,
            " Γ^", scale_power, "*dk1=", s.scaled_dk1,
            " Γ^", scale_power, "*dk2=", s.scaled_dk2,
            " scaled maxabs=", s.scaled_maxabs,
        )
    end
    return summaries
end

function summarize_symmetrized_output_dir(output_dir::AbstractString; Γs::Union{Nothing, AbstractVector}=nothing, scale_power::Int=4)
    meta = load_cubic_metadata(output_dir)
    Γlist = isnothing(Γs) ? Vector{Float64}(meta["Γs"]) : collect(Float64.(Γs))
    summaries = [extract_symmetrized_summary(output_dir, Γ; scale_power=scale_power) for Γ in Γlist]

    println("output_dir = ", output_dir)
    println("symmetrized distribution = ", meta["distribution_mode"], ", ωband = ", metadata_ωband(meta), ", dω = ", metadata_dω(meta))
    for s in summaries
        println(
            "Γ=", s.Γ,
            " Γ^", scale_power, "*d2Ω1=", s.scaled_d2Ω1,
            " Γ^", scale_power, "*d2Ω2=", s.scaled_d2Ω2,
            " Γ^", scale_power, "*dΩ1dΩ2=", s.scaled_dΩ1dΩ2,
            " Γ^", scale_power, "*d2k1=", s.scaled_d2k1,
            " Γ^", scale_power, "*d2k2=", s.scaled_d2k2,
            " Γ^", scale_power, "*dΩ1dk1=", s.scaled_dΩ1dk1,
            " Γ^", scale_power, "*dΩ1dk2=", s.scaled_dΩ1dk2,
            " Γ^", scale_power, "*dk1dk2=", s.scaled_dk1dk2,
        )
    end
    return summaries
end

function compare_symmetrized_bandwidths(output_dirs::AbstractVector{<:AbstractString}, Γ::Real; scale_power::Int=4)
    summaries = [extract_symmetrized_summary(dir, Γ; scale_power=scale_power) for dir in output_dirs]
    sort!(summaries; by=s -> s.ωband)
    println("symmetrized Γ = ", Γ, ", scale power = ", scale_power)
    for s in summaries
        println(
            "ωband=", s.ωband,
            " Γ^", scale_power, "*d2Ω1=", s.scaled_d2Ω1,
            " Γ^", scale_power, "*d2Ω2=", s.scaled_d2Ω2,
            " Γ^", scale_power, "*dΩ1dΩ2=", s.scaled_dΩ1dΩ2,
            " Γ^", scale_power, "*d2k1=", s.scaled_d2k1,
            " Γ^", scale_power, "*d2k2=", s.scaled_d2k2,
            " Γ^", scale_power, "*dΩ1dk1=", s.scaled_dΩ1dk1,
            " Γ^", scale_power, "*dΩ1dk2=", s.scaled_dΩ1dk2,
            " Γ^", scale_power, "*dk1dk2=", s.scaled_dk1dk2,
        )
    end
    return summaries
end

function main()
    dyson_sign_dir = "cubic_triangle_scan_dyson_sign_Nw12800_Nk256_wmax320p000"
    if isfile(joinpath(dyson_sign_dir, "metadata.jld2"))
        summarize_output_dir(dyson_sign_dir)
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
