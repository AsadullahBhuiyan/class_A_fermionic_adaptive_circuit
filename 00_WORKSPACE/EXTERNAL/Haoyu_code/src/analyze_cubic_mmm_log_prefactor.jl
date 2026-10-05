using JLD2
using Printf
using Statistics

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function fit_fixed_log_prefactor(Ω::AbstractVector{Float64}, values::AbstractVector{ComplexF64}, Γ::Float64; coeff_log::Float64)
    y = imag.(values)
    x = Ω
    B = -mean((Γ^4 .* y ./ x) .+ coeff_log .* log.(1.0 ./ x))
    fitted = .-(x ./ Γ^4) .* (coeff_log .* log.(1.0 ./ x) .+ B)
    rel_rms = sqrt(mean(abs2, y .- fitted)) / max(maximum(abs.(y)), eps())
    return (B=B, fitted=fitted, rel_rms=rel_rms)
end

function analyze_log_prefactor(;
    data_dir::AbstractString="data/cubic_mmm_dynamic_line_dyson_sign_Nw25600_Nk256_dw0p025",
    Γs::AbstractVector=[40.0, 80.0, 160.0],
    coeff_log::Float64=1 / 8,
)
    for Γraw in Γs
        Γ = Float64(Γraw)
        data = load(joinpath(data_dir, "gamma_$(gamma_token(Γ)).jld2"))
        Ω_values = Float64.(data["Ω_values"])
        sel = findall(>(0.0), Ω_values)
        Ω_positive = Ω_values[sel]
        exact_values = ComplexF64.(data["exact_values"][sel])
        reduced_values = ComplexF64.(data["reduced_values"][sel])

        exact_fit = fit_fixed_log_prefactor(Ω_positive, exact_values, Γ; coeff_log=coeff_log)
        reduced_fit = fit_fixed_log_prefactor(Ω_positive, reduced_values, Γ; coeff_log=coeff_log)

        println("Γ = ", Γ)
        println("  exact  : coeff_log = ", coeff_log, ", B = ", exact_fit.B, ", rel_rms = ", exact_fit.rel_rms)
        println("  reduced: coeff_log = ", coeff_log, ", B = ", reduced_fit.B, ", rel_rms = ", reduced_fit.rel_rms)
    end
end

function main()
    analyze_log_prefactor()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
