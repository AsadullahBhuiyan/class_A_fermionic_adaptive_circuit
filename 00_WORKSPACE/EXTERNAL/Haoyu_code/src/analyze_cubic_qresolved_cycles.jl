using JLD2
using Printf

include("decompose_cubic_linear_omega_xbasis.jl")

function q_to_signed(q::Float64)
    return q <= π ? q : q - 2π
end

function qresolved_cycle_derivative(
    Γ::Float64;
    cycle_idx::Int,
    Nω::Int=12_800,
    Nk::Int=256,
    dω::Float64=0.05,
    r::Float64=1.0,
    η::Float64=1e-4,
    distribution_mode::Symbol=:dyson_sign,
)
    ωs = physical_frequency_grid(Nω, dω)
    ks = collect(2π .* (0:(Nk - 1)) ./ Nk)
    Ω_shifts = centered_shifts(5)
    Ω_values = dω .* Ω_shifts
    zero_shift = findfirst(iszero, Ω_shifts)
    prefactor = dω / (2π * Nk)

    GR, GA, GK = build_green_components_xbasis(
        ωs,
        ks;
        r=r,
        Γ=Γ,
        η=η,
        distribution_mode=distribution_mode,
    )

    qresolved = zeros(ComplexF64, Nk, length(Ω_shifts))
    @inbounds for jq in 1:Nk
        for iω in 1:Nω
            for (sidx, n) in pairs(Ω_shifts)
                iωs = iω - n
                if 1 <= iωs <= Nω
                    qresolved[jq, sidx] += cycle_term(GR, iω, jq, GA, iωs, jq, GK, iωs, jq, cycle_idx)
                    qresolved[jq, sidx] += cycle_term(GK, iω, jq, GR, iωs, jq, GA, iωs, jq, cycle_idx)
                    qresolved[jq, sidx] += cycle_term(GA, iω, jq, GK, iωs, jq, GR, iωs, jq, cycle_idx)
                    qresolved[jq, sidx] -= cycle_term(GK, iω, jq, GK, iωs, jq, GK, iωs, jq, cycle_idx)
                end
            end
        end
    end

    qresolved .*= prefactor
    ΔΩ = Ω_values[zero_shift + 1] - Ω_values[zero_shift]
    derivative = Γ^3 .* (qresolved[:, zero_shift + 1] .- qresolved[:, zero_shift - 1]) ./ (2 * ΔΩ)
    signed_q = q_to_signed.(ks)
    edge_mask = abs.(signed_q) .< (π / 2)

    return (
        Γ=Γ,
        cycle_idx=cycle_idx,
        cycle_label=CYCLE_LABELS[cycle_idx],
        signed_q=signed_q,
        derivative=derivative,
        in_edge=sum(derivative[edge_mask]),
        out_edge=sum(derivative[.!edge_mask]),
        total=sum(derivative),
    )
end

function main(;
    Γ::Float64=80.0,
    Nω::Int=12_800,
    Nk::Int=256,
    dω::Float64=0.05,
)
    for cycle_idx in (1, 8)
        result = qresolved_cycle_derivative(
            Γ;
            cycle_idx=cycle_idx,
            Nω=Nω,
            Nk=Nk,
            dω=dω,
        )
        println("cycle = ", result.cycle_label)
        println("  in_edge  = ", result.in_edge)
        println("  out_edge = ", result.out_edge)
        println("  total    = ", result.total)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
