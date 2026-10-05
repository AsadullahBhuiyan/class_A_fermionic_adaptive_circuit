using Printf

include("collect_cubic_triangle_data.jl")

function static_mmm_linear_coefficient(
    Γ::Float64;
    Nω::Int=12_800,
    Nk::Int=256,
    dω::Float64=0.05,
)
    ωs = physical_frequency_grid(Nω, dω)
    qs = collect(2π .* (0:(Nk - 1)) ./ Nk)
    Ω_shifts = centered_shifts(5)
    Ω_values = dω .* Ω_shifts
    zero_shift = findfirst(iszero, Ω_shifts)
    prefactor = dω / (2π * Nk)

    values = zeros(ComplexF64, length(Ω_shifts))
    sector_values = zeros(ComplexF64, length(Ω_shifts), 4)

    @inbounds for jq in eachindex(qs)
        q = qs[jq]
        z0 = ComplexF64(0.0, 1e-8)
        Σ0 = Σbulk(z0, q, 1.0)
        m = 1 - cos(q)
        b = -m
        for iω in eachindex(ωs)
            ω = ωs[iω]
            a0 = ComplexF64(ω - sin(q), Γ / 2)
            d0 = ComplexF64(real(ω + sin(q) - Σ0), imag(ω + sin(q) - Σ0) + Γ / 2)
            det0 = inv(a0 * d0 - b * b)
            g0R = a0 * det0
            g0A = conj(g0R)
            g0K = distribution_value(Val(:dyson_sign), ω) * (g0R - g0A + im * Γ * g0R * g0A)
            for (sidx, n) in pairs(Ω_shifts)
                iωs = iω - n
                if 1 <= iωs <= Nω
                    ωsft = ωs[iωs]
                    a1 = ComplexF64(ωsft - sin(q), Γ / 2)
                    d1raw = ωsft + sin(q) - Σ0
                    d1 = ComplexF64(real(d1raw), imag(d1raw) + Γ / 2)
                    det1 = inv(a1 * d1 - b * b)
                    g1R = a1 * det1
                    g1A = conj(g1R)
                    g1K = distribution_value(Val(:dyson_sign), ωsft) * (g1R - g1A + im * Γ * g1R * g1A)

                    sector_values[sidx, 1] += g0R * g1A * g1K
                    sector_values[sidx, 2] += g0K * g1R * g1A
                    sector_values[sidx, 3] += g0A * g1K * g1R
                    sector_values[sidx, 4] -= g0K * g1K * g1K
                end
            end
        end
    end

    sector_values .*= prefactor
    values .= vec(sum(sector_values; dims=2))
    ΔΩ = Ω_values[zero_shift + 1] - Ω_values[zero_shift]
    scaled_derivative = Γ^3 * (values[zero_shift + 1] - values[zero_shift - 1]) / (2 * ΔΩ)
    scaled_sector_derivative = Γ^3 .* (sector_values[zero_shift + 1, :] .- sector_values[zero_shift - 1, :]) ./ (2 * ΔΩ)

    return (
        Γ=Γ,
        scaled_derivative=scaled_derivative,
        scaled_sector_derivative=scaled_sector_derivative,
    )
end

function main()
    for Γ in (20.0, 40.0, 80.0)
        result = static_mmm_linear_coefficient(Γ)
        println("Γ = ", Γ)
        println("  static --- total = ", result.scaled_derivative)
        println("  sectors = ", result.scaled_sector_derivative)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
