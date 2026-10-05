using JLD2
using Printf
using Plots
using LaTeXStrings

function gamma_token(Γ::Real)
    replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function load_ycurve(input_dir::AbstractString, Γ::Real)
    data = load(joinpath(input_dir, "gamma_$(gamma_token(Γ)).jld2"))
    ks = data["k_positive"]
    k0 = data["kline"][findfirst(==(0.0), data["k_values"])]
    y = real.(Γ .* data["kline_positive"] .- Γ .* k0) ./ (ks .^ 2)
    return ks, y
end

function analyze_smallk_touching_plateau(;
    full_dir::AbstractString=raw"smallk_kline_fft_hiBW_Nw12288_Nk2048",
    touch_dir::AbstractString=raw"smallk_kline_fft_hiBW_Nw12288_Nk2048_touching_window",
    full_dir_hiBW2::AbstractString=raw"smallk_kline_fft_hiBW2_Nw12288_Nk2048",
    touch_dir_hiBW2::AbstractString=raw"smallk_kline_fft_hiBW2_Nw12288_Nk2048_touching_window",
    output_pdf::AbstractString="smallk_touching_plateau_analysis.pdf",
    output_jld2::AbstractString="smallk_touching_plateau_analysis.jld2",
    nplot::Int=24,
)
    Γs = [20.0, 40.0, 80.0]
    plateau_full = Float64[]
    plateau_touch = Float64[]

    plt1 = plot(
        xlabel=L"k",
        ylabel=L"\mathrm{Re}[\Gamma\Pi(0,k)-\Gamma\Pi(0,0)]/k^2",
        title="Small-k Plateau At Γ = 80",
        legend=:bottomright,
    )

    ks_full_80, y_full_80 = load_ycurve(full_dir_hiBW2, 80.0)
    ks_touch_80, y_touch_80 = load_ycurve(touch_dir_hiBW2, 80.0)
    plot!(plt1, ks_full_80[1:nplot], y_full_80[1:nplot]; linewidth=2.5, label="full, ωmax = 640")
    plot!(plt1, ks_touch_80[1:nplot], y_touch_80[1:nplot]; linewidth=2.5, label="touching window, ωmax = 640")
    hline!(plt1, [-1 / (4π)]; linestyle=:dash, linewidth=2, label=L"-1/(4\pi)")

    plt2 = plot(
        xlabel=L"1/\Gamma",
        ylabel=L"Y_\Gamma(k_1)",
        title=L"First-point plateau: \; Y_\Gamma(k)=\mathrm{Re}[\Gamma\Pi(0,k)-\Gamma\Pi(0,0)]/k^2",
        legend=:bottomright,
    )
    invΓ = 1.0 ./ Γs
    for Γ in Γs
        ks_full, y_full = load_ycurve(full_dir, Γ)
        ks_touch, y_touch = load_ycurve(touch_dir, Γ)
        push!(plateau_full, y_full[1])
        push!(plateau_touch, y_touch[1])
    end
    scatter!(plt2, invΓ, plateau_full; markersize=6, label="full, ωmax = 320")
    plot!(plt2, invΓ, plateau_full; linewidth=2, label="")
    scatter!(plt2, invΓ, plateau_touch; markersize=6, label="touching window, ωmax = 320")
    plot!(plt2, invΓ, plateau_touch; linewidth=2, label="")
    scatter!(plt2, [1 / 80], [y_touch_80[1]]; markersize=7, markerstrokewidth=0, label="touching window, ωmax = 640")
    hline!(plt2, [-1 / (4π)]; linestyle=:dash, linewidth=2, label=L"-1/(4\pi)")

    plt = plot(plt1, plt2; layout=(2, 1), size=(900, 800))
    savefig(plt, output_pdf)

    jldsave(
        output_jld2;
        full_dir,
        touch_dir,
        full_dir_hiBW2,
        touch_dir_hiBW2,
        Γs,
        invΓ,
        plateau_full,
        plateau_touch,
        plateau_touch_hiBW2=y_touch_80[1],
        k_plot_full_80=ks_full_80[1:nplot],
        y_plot_full_80=y_full_80[1:nplot],
        k_plot_touch_80=ks_touch_80[1:nplot],
        y_plot_touch_80=y_touch_80[1:nplot],
        output_pdf,
    )

    println("saved ", output_pdf)
    println("saved ", output_jld2)
    for (Γ, yf, yt) in zip(Γs, plateau_full, plateau_touch)
        println((; Γ, y1_full=yf, y1_touch=yt, diff=yf - yt, Γdiff=Γ * (yf - yt)))
    end
    println((; Γ=80.0, y1_touch_hiBW2=y_touch_80[1]))
end

main(; kwargs...) = analyze_smallk_touching_plateau(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    analyze_smallk_touching_plateau()
end
