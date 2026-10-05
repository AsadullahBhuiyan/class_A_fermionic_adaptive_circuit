using Printf

include("collect_cubic_triangle_data.jl")

function bare_ppp_linear_coefficient(
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
        sq = sin(q)
        for iω in eachindex(ωs)
            ω = ωs[iω]
            g0R = inv(ComplexF64(ω - sq, Γ / 2))
            g0A = conj(g0R)
            g0K = distribution_value(Val(:dyson_sign), ω) * (g0R - g0A + im * Γ * g0R * g0A)
            for (sidx, n) in pairs(Ω_shifts)
                iωs = iω - n
                if 1 <= iωs <= Nω
                    ωsft = ωs[iωs]
                    g1R = inv(ComplexF64(ωsft - sq, Γ / 2))
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
        result = bare_ppp_linear_coefficient(Γ)
        println("Γ = ", Γ)
        println("  bare +++ total = ", result.scaled_derivative)
        println("  sectors = ", result.scaled_sector_derivative)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
