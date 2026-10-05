using JLD2
using LinearAlgebra
using Printf
using Plots

include("compare_touching_model.jl")

function load_scan_curvature_points(scan_dirs::AbstractVector{<:AbstractString})
    Γs = Float64[]
    d2Ω = Float64[]
    d2k = Float64[]
    dΩk = Float64[]

    for outdir in scan_dirs
        meta = load(joinpath(outdir, "metadata.jld2"))
        Ω_values = meta["Ω_values"]
        k_values = meta["k_values"]
        Ω0 = findfirst(==(0.0), Ω_values)
        k0 = findfirst(==(0.0), k_values)
        hΩ = Ω_values[Ω0 + 1] - Ω_values[Ω0]
        hk = abs(k_values[k0 + 1] - k_values[k0])

        for Γ in meta["Γs"]
            numeric = load(joinpath(outdir, "gamma_$(gamma_token(Γ)).jld2"))["conv_small"]
            ωline = Γ .* numeric[:, k0]
            kline = Γ .* numeric[Ω0, :]
            push!(Γs, Γ)
            push!(d2Ω, (real(ωline[Ω0 + 1]) - 2real(ωline[Ω0]) + real(ωline[Ω0 - 1])) / hΩ^2)
            push!(d2k, (real(kline[k0 + 1]) - 2real(kline[k0]) + real(kline[k0 - 1])) / hk^2)
            push!(dΩk, (
                real(Γ * numeric[Ω0 + 1, k0 + 1]) -
                real(Γ * numeric[Ω0 + 1, k0 - 1]) -
                real(Γ * numeric[Ω0 - 1, k0 + 1]) +
                real(Γ * numeric[Ω0 - 1, k0 - 1])
            ) / (4 * hΩ * hk))
        end
    end

    order = sortperm(Γs)
    return Γs[order], d2Ω[order], d2k[order], dΩk[order]
end

function load_single_scan_curvatures(scan_dir::AbstractString)
    meta = load(joinpath(scan_dir, "metadata.jld2"))
    Ω_values = meta["Ω_values"]
    k_values = meta["k_values"]
    Ω0 = findfirst(==(0.0), Ω_values)
    k0 = findfirst(==(0.0), k_values)
    hΩ = Ω_values[Ω0 + 1] - Ω_values[Ω0]
    hk = abs(k_values[k0 + 1] - k_values[k0])

    Γs = Float64[]
    d2Ω = Float64[]
    d2k = Float64[]
    dΩk = Float64[]

    for Γ in meta["Γs"]
        numeric = load(joinpath(scan_dir, "gamma_$(gamma_token(Γ)).jld2"))["conv_small"]
        ωline = Γ .* numeric[:, k0]
        kline = Γ .* numeric[Ω0, :]
        push!(Γs, Γ)
        push!(d2Ω, (real(ωline[Ω0 + 1]) - 2real(ωline[Ω0]) + real(ωline[Ω0 - 1])) / hΩ^2)
        push!(d2k, (real(kline[k0 + 1]) - 2real(kline[k0]) + real(kline[k0 - 1])) / hk^2)
        push!(dΩk, (
            real(Γ * numeric[Ω0 + 1, k0 + 1]) -
            real(Γ * numeric[Ω0 + 1, k0 - 1]) -
            real(Γ * numeric[Ω0 - 1, k0 + 1]) +
            real(Γ * numeric[Ω0 - 1, k0 - 1])
        ) / (4 * hΩ * hk))
    end

    return (Γs=Γs, d2Ω=d2Ω, d2k=d2k, dΩk=dΩk, hΩ=hΩ, hk=hk)
end

function fit_log_law(Γs::AbstractVector{<:Real}, values::AbstractVector{<:Real}; Γmin::Float64=8.0)
    sel = findall(>=(Γmin), Γs)
    A = hcat(ones(length(sel)), log.(Float64.(Γs[sel])))
    coeffs = A \ Float64.(values[sel])
    fitted = coeffs[1] .+ coeffs[2] .* log.(Float64.(Γs))
    return (coeffs=coeffs, fitted=fitted, sel=sel)
end

function fit_inverse_sqrt_law(Γs::AbstractVector{<:Real}, values::AbstractVector{<:Real}; Γmin::Float64=8.0)
    sel = findall(>=(Γmin), Γs)
    A = hcat(ones(length(sel)), Float64.(Γs[sel]).^(-0.5))
    coeffs = A \ Float64.(values[sel])
    fitted = coeffs[1] .+ coeffs[2] .* Float64.(Γs).^(-0.5)
    return (coeffs=coeffs, fitted=fitted, sel=sel)
end

function fit_constant_plus_inverse_law(Γs::AbstractVector{<:Real}, values::AbstractVector{<:Real}; Γmin::Float64=8.0)
    sel = findall(>=(Γmin), Γs)
    A = hcat(ones(length(sel)), Float64.(Γs[sel]).^(-1))
    coeffs = A \ Float64.(values[sel])
    fitted = coeffs[1] .+ coeffs[2] .* Float64.(Γs).^(-1)
    return (coeffs=coeffs, fitted=fitted, sel=sel)
end

function main(;
    scan_dirs::AbstractVector{<:AbstractString}=[
        "sparse_convolution_scan_Nw65536_Nk128",
        "sparse_convolution_scan_largeGamma_Nw65536_Nk128",
        "sparse_convolution_scan_largeGamma2_Nw65536_Nk128",
    ],
    output_pdf::AbstractString="large_gamma_curvature_analysis.pdf",
    output_jld2::AbstractString="large_gamma_curvature_analysis.jld2",
    Γmin_fit::Float64=8.0,
    refined_mixed_dirs::AbstractVector{<:AbstractString}=[
        "sparse_convolution_scan_mixedcheck2_Nw16384_Nk512",
        "sparse_convolution_scan_mixedcheck_Nw16384_Nk512",
    ],
    Γmin_mixed_fit::Float64=20.0,
)
    Γs, d2Ω, d2k, dΩk = load_scan_curvature_points(scan_dirs)
    refined = [load_single_scan_curvatures(dir) for dir in refined_mixed_dirs if isdir(dir)]
    Γs_mixed = reduce(vcat, [item.Γs for item in refined]; init=Float64[])
    dΩk_mixed = reduce(vcat, [item.dΩk for item in refined]; init=Float64[])
    mixed_order = sortperm(Γs_mixed)
    Γs_mixed = Γs_mixed[mixed_order]
    dΩk_mixed = dΩk_mixed[mixed_order]

    fitΩ = fit_log_law(Γs, d2Ω; Γmin=Γmin_fit)
    fitk = fit_log_law(Γs, d2k; Γmin=Γmin_fit)
    fitΩk = fit_constant_plus_inverse_law(Γs_mixed, dΩk_mixed; Γmin=Γmin_mixed_fit)

    plt1 = plot(
        Γs,
        d2Ω;
        xscale=:log10,
        marker=:circle,
        linewidth=2.5,
        color=:blue,
        label="data",
        xlabel=L"\Gamma",
        ylabel=L"\partial_\Omega^2\,\mathrm{Re}[\Gamma \Pi](0,0)",
        title=L"(\Omega,0)\ \mathrm{curvature}",
    )
    plot!(plt1, Γs, fitΩ.fitted; color=:red, linestyle=:dash, linewidth=2.5, label=@sprintf("%.3f %+.3f log Γ", fitΩ.coeffs[1], fitΩ.coeffs[2]))

    plt2 = plot(
        Γs,
        d2k;
        xscale=:log10,
        marker=:circle,
        linewidth=2.5,
        color=:blue,
        label="data",
        xlabel=L"\Gamma",
        ylabel=L"\partial_k^2\,\mathrm{Re}[\Gamma \Pi](0,0)",
        title=L"(0,k)\ \mathrm{curvature}",
    )
    plot!(plt2, Γs, fitk.fitted; color=:red, linestyle=:dash, linewidth=2.5, label=@sprintf("%.3f %+.3f log Γ", fitk.coeffs[1], fitk.coeffs[2]))

    plt3 = plot(
        Γs_mixed,
        dΩk_mixed;
        xscale=:log10,
        marker=:circle,
        linewidth=2.5,
        color=:blue,
        label="refined data",
        xlabel=L"\Gamma",
        ylabel=L"\partial_\Omega \partial_k\,\mathrm{Re}[\Gamma \Pi](0,0)",
        title=L"\Omega k\ \mathrm{curvature}",
    )
    plot!(plt3, Γs, dΩk; color=:gray, marker=:x, linestyle=:dot, linewidth=1.5, label="coarse Nk=128")
    plot!(plt3, Γs_mixed, fitΩk.fitted; color=:red, linestyle=:dash, linewidth=2.5, label=@sprintf("%.3f %+.3f / Γ", fitΩk.coeffs[1], fitΩk.coeffs[2]))

    plt = plot(plt1, plt2, plt3; layout=(3, 1), size=(900, 1350))
    savefig(plt, output_pdf)

    jldsave(
        output_jld2;
        scan_dirs=collect(String.(scan_dirs)),
        Γs,
        d2Ω,
        d2k,
        dΩk,
        Γs_mixed,
        dΩk_mixed,
        fit_Γmin=Γmin_fit,
        fit_Γmin_mixed=Γmin_mixed_fit,
        fitΩ_coeffs=fitΩ.coeffs,
        fitk_coeffs=fitk.coeffs,
        fitΩk_coeffs=fitΩk.coeffs,
        fitΩ_fitted=fitΩ.fitted,
        fitk_fitted=fitk.fitted,
        fitΩk_fitted=fitΩk.fitted,
        refined_mixed_dirs=collect(String.(refined_mixed_dirs)),
        output_pdf,
    )

    println("saved ", output_pdf)
    println("saved ", output_jld2)
    println("Ω curvature fit = ", fitΩ.coeffs)
    println("k curvature fit = ", fitk.coeffs)
    println("Ωk curvature fit = ", fitΩk.coeffs)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
