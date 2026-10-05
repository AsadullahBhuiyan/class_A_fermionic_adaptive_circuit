using JLD2
using Printf

include("touching_model.jl")

const TOUCHING_WZW_SCALING_DIR_PREFIX = "touching_wzw_scaling_scan"

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

distribution_tag(::Val{:sign}) = "sign"
distribution_tag(::Val{:unity}) = "unity"
distribution_tag(::Val{:dyson_sign}) = "dyson_sign"
distribution_tag(::Val{:dyson_unity}) = "dyson_unity"

@inline function distribution_value(::Val{:sign}, ω::Float64)
    return ω > 0 ? 1.0 : (ω < 0 ? -1.0 : 0.0)
end

@inline distribution_value(::Val{:unity}, ω::Float64) = 1.0
@inline distribution_value(::Val{:dyson_sign}, ω::Float64) = distribution_value(Val(:sign), ω)
@inline distribution_value(::Val{:dyson_unity}, ω::Float64) = 1.0

@inline function touching_keldysh_entries(
    ::Val{:sign},
    ω::Float64,
    Γ::Float64,
    g::NTuple{4,ComplexF64},
    ga::NTuple{4,ComplexF64},
)
    fω = distribution_value(Val(:sign), ω)
    return (
        fω * (g[1] - ga[1]),
        fω * (g[2] - ga[2]),
        fω * (g[3] - ga[3]),
        fω * (g[4] - ga[4]),
    )
end

@inline function touching_keldysh_entries(
    ::Val{:unity},
    ω::Float64,
    Γ::Float64,
    g::NTuple{4,ComplexF64},
    ga::NTuple{4,ComplexF64},
)
    return (
        g[1] - ga[1],
        g[2] - ga[2],
        g[3] - ga[3],
        g[4] - ga[4],
    )
end

@inline function touching_keldysh_entries(
    mode::Union{Val{:dyson_sign},Val{:dyson_unity}},
    ω::Float64,
    Γ::Float64,
    g::NTuple{4,ComplexF64},
    ga::NTuple{4,ComplexF64},
)
    fω = distribution_value(mode, ω)
    p11 = g[1] * ga[1] + g[2] * ga[3]
    p12 = g[1] * ga[2] + g[2] * ga[4]
    p21 = g[3] * ga[1] + g[4] * ga[3]
    p22 = g[3] * ga[2] + g[4] * ga[4]
    return (
        fω * ((g[1] - ga[1]) + im * Γ * p11),
        fω * ((g[2] - ga[2]) + im * Γ * p12),
        fω * ((g[3] - ga[3]) + im * Γ * p21),
        fω * ((g[4] - ga[4]) + im * Γ * p22),
    )
end

function scaling_output_dir(Γs::AbstractVector, umax::Float64, ymax::Float64, du::Float64, dy::Float64, distribution_mode::Symbol)
    gtag = join(gamma_token.(Γs), "_")
    return "$(TOUCHING_WZW_SCALING_DIR_PREFIX)_$(String(distribution_mode))_umax$(gamma_token(umax))_ymax$(gamma_token(ymax))_du$(gamma_token(du))_dy$(gamma_token(dy))_Γ$(gtag)"
end

@inline function trace_prod_2x2(
    a11::ComplexF64,
    a12::ComplexF64,
    a21::ComplexF64,
    a22::ComplexF64,
    b11::ComplexF64,
    b12::ComplexF64,
    b21::ComplexF64,
    b22::ComplexF64,
    c11::ComplexF64,
    c12::ComplexF64,
    c21::ComplexF64,
    c22::ComplexF64,
)
    m11 = a11 * b11 + a12 * b21
    m12 = a11 * b12 + a12 * b22
    m21 = a21 * b11 + a22 * b21
    m22 = a21 * b12 + a22 * b22
    return m11 * c11 + m12 * c21 + m21 * c12 + m22 * c22
end

@inline function trace_product(
    a::NTuple{4,ComplexF64},
    b::NTuple{4,ComplexF64},
    c::NTuple{4,ComplexF64},
)
    return trace_prod_2x2(
        a[1], a[2], a[3], a[4],
        b[1], b[2], b[3], b[4],
        c[1], c[2], c[3], c[4],
    )
end

function eval_touching_green(
    sector::Symbol,
    Γ::Float64,
    ueff::Float64,
    δphys::Float64,
    η::Float64;
    distribution_mode::Symbol,
    pcut::Float64,
    ωcut::Float64,
    cutoff_power::Int,
)
    ueff <= 0 && return (
        (0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im),
        (0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im),
        (0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im),
    )

    mode = Val(distribution_mode)
    p = ueff / Γ

    if sector == :upper
        ω = 1 + δphys
        k = π / 2 - p
        g = upper_touching_local_xbasis_entries(
            ω,
            k,
            Γ,
            η;
            pcut=pcut,
            ωcut=ωcut,
            cutoff_power=cutoff_power,
        )
    else
        ω = -1 + δphys
        k = -π / 2 + p
        g = lower_touching_local_xbasis_entries(
            ω,
            k,
            Γ,
            η;
            pcut=pcut,
            ωcut=ωcut,
            cutoff_power=cutoff_power,
        )
    end

    ga = (conj(g[1]), conj(g[2]), conj(g[3]), conj(g[4]))
    gk = touching_keldysh_entries(mode, ω, Γ, g, ga)
    return g, ga, gk
end

function triangle_sector_scaled(
    sector::Symbol,
    Γ::Float64,
    ν1::Float64,
    κ1::Float64,
    ν2::Float64,
    κ2::Float64,
    u_values::AbstractVector,
    y_values::AbstractVector,
    du::Float64,
    dy::Float64,
    η::Float64;
    distribution_mode::Symbol,
    pcut::Float64,
    ωcut::Float64,
    cutoff_power::Int,
)
    κsign = sector == :upper ? 1.0 : -1.0
    prefactor = du * dy / (4π^2 * Γ^6)
    acc = 0.0 + 0.0im

    @inbounds for u in u_values
        u2 = u * u
        for y in y_values
            δ0 = u2 * y / (2 * Γ^2)
            δ1 = (u2 * y - 2 * ν1) / (2 * Γ^2)
            δ2 = (u2 * y - 2 * (ν1 + ν2)) / (2 * Γ^2)

            g0R, g0A, g0K = eval_touching_green(
                sector,
                Γ,
                u,
                δ0,
                η;
                distribution_mode=distribution_mode,
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
            )
            u1 = u + κsign * κ1
            g1R, g1A, g1K = eval_touching_green(
                sector,
                Γ,
                u1,
                δ1,
                η;
                distribution_mode=distribution_mode,
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
            )
            u2shift = u + κsign * (κ1 + κ2)
            g2R, g2A, g2K = eval_touching_green(
                sector,
                Γ,
                u2shift,
                δ2,
                η;
                distribution_mode=distribution_mode,
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
            )

            acc += u2 * trace_product(g0R, g1A, g2K)
            acc += u2 * trace_product(g0K, g1R, g2A)
            acc += u2 * trace_product(g0A, g1K, g2R)
            acc -= u2 * trace_product(g0K, g1K, g2K)
        end
    end

    return prefactor * acc
end

function touching_triangle_scaled(
    Γ::Float64,
    ν1::Float64,
    κ1::Float64,
    ν2::Float64,
    κ2::Float64,
    u_values::AbstractVector,
    y_values::AbstractVector,
    du::Float64,
    dy::Float64,
    η::Float64;
    distribution_mode::Symbol,
    pcut::Float64,
    ωcut::Float64,
    cutoff_power::Int,
)
    upper = triangle_sector_scaled(
        :upper,
        Γ,
        ν1,
        κ1,
        ν2,
        κ2,
        u_values,
        y_values,
        du,
        dy,
        η;
        distribution_mode=distribution_mode,
        pcut=pcut,
        ωcut=ωcut,
        cutoff_power=cutoff_power,
    )
    lower = triangle_sector_scaled(
        :lower,
        Γ,
        ν1,
        κ1,
        ν2,
        κ2,
        u_values,
        y_values,
        du,
        dy,
        η;
        distribution_mode=distribution_mode,
        pcut=pcut,
        ωcut=ωcut,
        cutoff_power=cutoff_power,
    )
    return upper + lower
end

function compute_scaling_grid(
    Γ::Float64,
    ν_values::AbstractVector,
    κ_values::AbstractVector,
    u_values::AbstractVector,
    y_values::AbstractVector,
    du::Float64,
    dy::Float64,
    η::Float64;
    distribution_mode::Symbol,
    pcut::Float64,
    ωcut::Float64,
    cutoff_power::Int,
)
    anti_grid = Array{ComplexF64}(undef, length(ν_values), length(κ_values))
    sym_grid = similar(anti_grid)
    scaled_local_grid = similar(anti_grid)
    raw12_grid = similar(anti_grid)
    raw21_grid = similar(anti_grid)

    @inbounds for j in eachindex(κ_values)
        κ = κ_values[j]
        for i in eachindex(ν_values)
            ν = ν_values[i]
            t12 = touching_triangle_scaled(
                Γ,
                ν,
                0.0,
                0.0,
                κ,
                u_values,
                y_values,
                du,
                dy,
                η;
                distribution_mode=distribution_mode,
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
            )
            t21 = touching_triangle_scaled(
                Γ,
                0.0,
                κ,
                ν,
                0.0,
                u_values,
                y_values,
                du,
                dy,
                η;
                distribution_mode=distribution_mode,
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
            )
            anti = 0.5 * (t12 - t21)
            sym = 0.5 * (t12 + t21)

            raw12_grid[i, j] = t12
            raw21_grid[i, j] = t21
            anti_grid[i, j] = anti
            sym_grid[i, j] = sym
            scaled_local_grid[i, j] = iszero(ν) || iszero(κ) ? 0.0 + 0.0im : anti * Γ^7 / (ν * κ)
        end
    end

    return (
        raw12_grid=raw12_grid,
        raw21_grid=raw21_grid,
        anti_grid=anti_grid,
        sym_grid=sym_grid,
        scaled_local_grid=scaled_local_grid,
    )
end

function save_scaling_metadata(output_dir::AbstractString, metadata::NamedTuple)
    jldsave(joinpath(output_dir, "metadata.jld2"); metadata...)
    return nothing
end

function save_scaling_gamma(output_dir::AbstractString, Γ::Float64, payload::NamedTuple)
    filepath = joinpath(output_dir, "gamma_$(gamma_token(Γ)).jld2")
    jldsave(filepath; Γ, payload...)
    return filepath
end

function search_touching_wzw_scaling_scan(;
    Γs::AbstractVector=[20.0, 40.0, 80.0, 160.0],
    ν_values::AbstractVector=[0.5, 1.0, 2.0, 4.0],
    κ_values::AbstractVector=[0.5, 1.0, 2.0, 4.0],
    du::Float64=0.1,
    umax::Float64=120.0,
    dy::Float64=0.1,
    ymax::Float64=20.0,
    η::Float64=1e-4,
    distribution_mode::Symbol=:dyson_sign,
    pcut::Float64=π / 2,
    ωcut::Float64=3.0,
    cutoff_power::Int=TOUCHING_DEFAULT_CUTOFF_POWER,
    output_dir::Union{Nothing, AbstractString}=nothing,
)
    @assert distribution_mode in (:sign, :unity, :dyson_sign, :dyson_unity) "distribution_mode must be :sign, :unity, :dyson_sign, or :dyson_unity"
    Γs = collect(Float64.(Γs))
    ν_values = collect(Float64.(ν_values))
    κ_values = collect(Float64.(κ_values))
    u_values = collect(du / 2:du:umax)
    y_values = collect(-ymax:dy:ymax)
    resolved_output_dir = isnothing(output_dir) ?
        scaling_output_dir(Γs, umax, ymax, du, dy, distribution_mode) :
        String(output_dir)
    mkpath(resolved_output_dir)

    metadata = (
        Γs=Γs,
        ν_values=ν_values,
        κ_values=κ_values,
        du=du,
        umax=umax,
        dy=dy,
        ymax=ymax,
        η=η,
        distribution_mode=String(distribution_mode),
        pcut=pcut,
        ωcut=ωcut,
        cutoff_power=cutoff_power,
        u_values=u_values,
        y_values=y_values,
    )
    save_scaling_metadata(resolved_output_dir, metadata)

    saved_files = String[]
    for Γ in Γs
        println("computing touching WZW scaling scan for Γ=", Γ)
        payload = compute_scaling_grid(
            Γ,
            ν_values,
            κ_values,
            u_values,
            y_values,
            du,
            dy,
            η;
            distribution_mode=distribution_mode,
            pcut=pcut,
            ωcut=ωcut,
            cutoff_power=cutoff_power,
        )
        filepath = save_scaling_gamma(resolved_output_dir, Γ, payload)
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

main(; kwargs...) = search_touching_wzw_scaling_scan(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
