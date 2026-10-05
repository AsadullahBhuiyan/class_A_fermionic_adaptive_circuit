using JLD2
using Printf

const EDGE_CUBIC_DIR_PREFIX = "edge_cubic_scan"

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

@inline edge_energy(q::Float64) = sin(q)
@inline edge_velocity(q::Float64) = cos(q)
@inline edge_weight(q::Float64) = max(cos(q) * (2 - cos(q)), 0.0)

@inline function edge_gR(u::Float64, Γ::Float64)
    return (2u) / (Γ * (im * u - 2))
end

@inline edge_gA(u::Float64, Γ::Float64) = conj(edge_gR(u, Γ))

@inline function distribution_sign(ω::Float64)
    return ω > 0 ? 1.0 : (ω < 0 ? -1.0 : 0.0)
end

@inline function edge_gK(u::Float64, Γ::Float64, ω::Float64)
    fω = distribution_sign(ω)
    gR = edge_gR(u, Γ)
    gA = conj(gR)
    return fω * (gR - gA)
end

function edge_triangle_value(
    Ω1::Float64,
    k1::Float64,
    Ω2::Float64,
    k2::Float64,
    Γ::Float64;
    Nq::Int=4000,
    du::Float64=0.1,
    umax::Float64=120.0,
)
    dq = π / Nq
    acc = 0.0 + 0.0im

    @inbounds for jq in 1:Nq
        q = -π / 2 + (jq - 0.5) * dq
        Zq = edge_weight(q)
        Zq == 0.0 && continue
        vq = edge_velocity(q)
        ωedge = edge_energy(q)

        for u in (du / 2):du:umax
            for sgn in (-1.0, 1.0)
                uu = sgn * u
                ω0 = ωedge + Zq * uu / Γ
                δ1 = Γ * (vq * k1 - Ω1) / Zq
                δ2 = Γ * (vq * (k1 + k2) - (Ω1 + Ω2)) / Zq
                u1 = uu + δ1
                u2 = uu + δ2

                g0R = edge_gR(uu, Γ)
                g1R = edge_gR(u1, Γ)
                g2R = edge_gR(u2, Γ)
                g0A = conj(g0R)
                g1A = conj(g1R)
                g2A = conj(g2R)
                g0K = distribution_sign(ω0) * (g0R - g0A)
                g1K = distribution_sign(ω0 - Ω1) * (g1R - g1A)
                g2K = distribution_sign(ω0 - Ω1 - Ω2) * (g2R - g2A)

                acc += Zq * (
                    g0R * g1A * g2K +
                    g0K * g1R * g2A +
                    g0A * g1K * g2R -
                    g0K * g1K * g2K
                )
            end
        end
    end

    return dq * du * acc / (2π)^2
end

function edge_triangle_scaled_scan(
    ;
    Γs::AbstractVector=[20.0, 40.0, 80.0, 160.0],
    ν_values::AbstractVector=[0.5, 1.0, 2.0],
    κ_values::AbstractVector=[0.5, 1.0, 2.0],
    Nq::Int=4000,
    du::Float64=0.1,
    umax::Float64=120.0,
    output_dir::Union{Nothing,AbstractString}=nothing,
)
    Γs = collect(Float64.(Γs))
    ν_values = collect(Float64.(ν_values))
    κ_values = collect(Float64.(κ_values))
    resolved_output_dir = isnothing(output_dir) ?
        "$(EDGE_CUBIC_DIR_PREFIX)_Nq$(Nq)_du$(gamma_token(du))_umax$(gamma_token(umax))" :
        String(output_dir)
    mkpath(resolved_output_dir)

    metadata = (
        Γs=Γs,
        ν_values=ν_values,
        κ_values=κ_values,
        Nq=Nq,
        du=du,
        umax=umax,
    )
    jldsave(joinpath(resolved_output_dir, "metadata.jld2"); metadata...)

    saved_files = String[]
    for Γ in Γs
        println("computing edge cubic scan for Γ=", Γ)
        raw12_grid = Array{ComplexF64}(undef, length(ν_values), length(κ_values))
        raw21_grid = similar(raw12_grid)
        anti_grid = similar(raw12_grid)
        sym_grid = similar(raw12_grid)
        scaled_local_grid = similar(raw12_grid)
        scaled_nonanalytic_grid = similar(raw12_grid)

        @inbounds for j in eachindex(κ_values)
            κ = κ_values[j]
            for i in eachindex(ν_values)
                ν = ν_values[i]
                Ω = ν / Γ^2
                k = κ / Γ
                t12 = edge_triangle_value(Ω, 0.0, 0.0, k, Γ; Nq=Nq, du=du, umax=umax)
                t21 = edge_triangle_value(0.0, k, Ω, 0.0, Γ; Nq=Nq, du=du, umax=umax)
                anti = 0.5 * (t12 - t21)
                sym = 0.5 * (t12 + t21)
                raw12_grid[i, j] = t12
                raw21_grid[i, j] = t21
                anti_grid[i, j] = anti
                sym_grid[i, j] = sym
                scaled_local_grid[i, j] = anti * Γ^7 / (ν * κ)
                scaled_nonanalytic_grid[i, j] = anti * Γ^7 / κ
            end
        end

        filepath = joinpath(resolved_output_dir, "gamma_$(gamma_token(Γ)).jld2")
        jldsave(
            filepath;
            Γ,
            raw12_grid,
            raw21_grid,
            anti_grid,
            sym_grid,
            scaled_local_grid,
            scaled_nonanalytic_grid,
        )
        push!(saved_files, filepath)
        println("saved ", filepath)
    end

    jldsave(joinpath(resolved_output_dir, "index.jld2"); output_dir=resolved_output_dir, Γs, saved_files)
    return (
        output_dir=resolved_output_dir,
        Γs=Γs,
        saved_files=saved_files,
    )
end

main(; kwargs...) = edge_triangle_scaled_scan(; kwargs...)

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
