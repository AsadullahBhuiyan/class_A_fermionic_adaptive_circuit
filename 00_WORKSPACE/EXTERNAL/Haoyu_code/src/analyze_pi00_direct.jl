using JLD2
using Plots
using Printf
using Base.Threads: @threads, maxthreadid, threadid

include("taylor_fit_convolution.jl")

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function midpoint_frequency_grid_from_spacing(ωmax::Float64, dω_target::Float64)
    Nω = ceil(Int, 2 * ωmax / dω_target)
    isodd(Nω) && (Nω += 1)
    dω = 2 * ωmax / Nω
    ωs = Vector{Float64}(undef, Nω)
    @inbounds for i in 1:Nω
        ωs[i] = -ωmax + (i - 0.5) * dω
    end
    return ωs, dω
end

function uniform_momentum_grid(Nq::Int)
    dq = 2π / Nq
    qs = Vector{Float64}(undef, Nq)
    @inbounds for j in 1:Nq
        qs[j] = (j - 1) * dq
    end
    return qs, dq
end

@inline function rotate_to_sigma_x_entries(
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
)
    gpp = 0.5 * (g11 + g12 + g21 + g22)
    gpm = 0.5 * (g11 - g12 + g21 - g22)
    gmp = 0.5 * (g11 + g12 - g21 - g22)
    gmm = 0.5 * (g11 - g12 - g21 + g22)
    return gpp, gpm, gmp, gmm
end

@inline tail_correction_pi00(ωmax::Float64) = 2 / (π * ωmax)
@inline tail_correction_pi00_pp_or_mm(ωmax::Float64) = 1 / (π * ωmax)

function pi00_components_for_gamma(
    Γ::Float64,
    ωs::Vector{Float64},
    dω::Float64,
    qs::Vector{Float64};
    r::Float64=1.0,
    η::Float64=1e-4,
)
    nt = maxthreadid()
    partial_total = zeros(Float64, nt)
    partial_pp = zeros(Float64, nt)
    partial_mixed = zeros(Float64, nt)
    partial_mm = zeros(Float64, nt)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        total_local = 0.0
        pp_local = 0.0
        mixed_local = 0.0
        mm_local = 0.0

        @inbounds for ω in ωs
            g11, g12, g21, g22 = monitored_retarded_green_entries(ω, q, r, Γ, η)
            total_local += abs2(g11) + abs2(g12) + abs2(g21) + abs2(g22)

            gpp, gpm, gmp, gmm = rotate_to_sigma_x_entries(g11, g12, g21, g22)
            pp_local += abs2(gpp)
            mixed_local += abs2(gpm) + abs2(gmp)
            mm_local += abs2(gmm)
        end

        partial_total[tid] += total_local
        partial_pp[tid] += pp_local
        partial_mixed[tid] += mixed_local
        partial_mm[tid] += mm_local
    end

    prefactor = dω / (2π * length(qs))
    Π00 = prefactor * sum(partial_total)
    Πpp00 = prefactor * sum(partial_pp)
    Πmixed00 = prefactor * sum(partial_mixed)
    Πmm00 = prefactor * sum(partial_mm)
    return (Π00=Π00, Πpp00=Πpp00, Πmixed00=Πmixed00, Πmm00=Πmm00)
end

function load_existing_pi00_points(dir::AbstractString)
    meta = load(joinpath(dir, "metadata.jld2"))
    Γs = Float64.(meta["Γs"])
    ωmax = Float64(meta["ωmax"])
    rows = NamedTuple[]
    for Γ in Γs
        data = load(joinpath(dir, "gamma_$(gamma_token(Γ)).jld2"))
        Π00 = Float64(real(data["Π00"]))
        push!(rows, (
            dir=dir,
            Γ=Γ,
            ωmax=ωmax,
            Π00=Π00,
            ΓΠ00=Γ * Π00,
            corrected_ΓΠ00=Γ * (Π00 + tail_correction_pi00(ωmax)),
        ))
    end
    return rows
end

function plot_pi00(results::AbstractVector, overlap_rows::AbstractVector, output_pdf::AbstractString)
    Γvals = Float64[getfield(row, :Γ) for row in results]
    full_raw = [row.ΓΠ00 for row in results]
    full_corr = [row.corrected_ΓΠ00 for row in results]
    pp_corr = [row.corrected_ΓΠpp00 for row in results]
    mixed_corr = [row.corrected_ΓΠmixed00 for row in results]
    mm_corr = [row.corrected_ΓΠmm00 for row in results]

    plt1 = plot(
        xlabel="Γ",
        ylabel="Γ Π(0,0)",
        title="Raw and tail-corrected Γ Π(0,0)",
        xscale=:log10,
        legend=:bottomleft,
    )
    plot!(plt1, Γvals, full_raw; label="raw direct", marker=:circle, linewidth=2.0, color=:black)
    plot!(plt1, Γvals, full_corr; label="tail-corrected", marker=:square, linewidth=2.0, color=:red)
    hline!(plt1, [2.0]; label="2", linestyle=:dash, color=:gray)

    plt2 = plot(
        xlabel="Γ",
        ylabel="tail-corrected contribution",
        title="Rotated-basis decomposition of Γ Π(0,0)",
        xscale=:log10,
        legend=:right,
    )
    plot!(plt2, Γvals, pp_corr; label="exact ++", marker=:circle, linewidth=2.0, color=:blue)
    plot!(plt2, Γvals, mixed_corr; label="off-diagonal", marker=:diamond, linewidth=2.0, color=:green)
    plot!(plt2, Γvals, mm_corr; label="exact --", marker=:utriangle, linewidth=2.0, color=:orange)
    plot!(plt2, Γvals, full_corr; label="total", marker=:square, linewidth=2.0, color=:black, linestyle=:dash)

    overlap_groups = Dict{Float64, Vector{NamedTuple}}()
    for row in overlap_rows
        push!(get!(overlap_groups, row.Γ, NamedTuple[]), row)
    end
    plt3 = plot(
        xlabel="1 / ωmax",
        ylabel="Γ Π(0,0)",
        title="Bandwidth dependence from existing direct datasets",
        legend=:bottomleft,
    )
    colors = Dict(20.0 => :blue, 40.0 => :green, 80.0 => :red)
    for Γ in sort(collect(keys(overlap_groups)))
        rows = sort(overlap_groups[Γ]; by=row -> row.ωmax)
        xs = [1 / row.ωmax for row in rows]
        ys = [row.ΓΠ00 for row in rows]
        ycorr = [row.corrected_ΓΠ00 for row in rows]
        plot!(plt3, xs, ys; label="raw, Γ=$(Int(round(Γ)))", marker=:circle, linewidth=2.0, color=get(colors, Γ, :black))
        plot!(plt3, xs, ycorr; label="corrected, Γ=$(Int(round(Γ)))", marker=:square, linewidth=1.8, linestyle=:dash, color=get(colors, Γ, :black))
    end
    hline!(plt3, [2.0]; label="2", linestyle=:dot, color=:gray)

    plt = plot(plt1, plt2, plt3; layout=(3, 1), size=(950, 1200))
    savefig(plt, output_pdf)
end

function main(;
    Γs::AbstractVector=[10.0, 20.0, 40.0, 80.0, 160.0],
    Nq::Int=2048,
    dω_target::Float64=2 * 640.0 / 32768.0,
    bandwidth_factor::Float64=8.0,
    min_ωmax::Float64=320.0,
    r::Float64=1.0,
    η::Float64=1e-4,
    overlap_dirs::AbstractVector=[
        "data/smallk_direct_lines_Nw16384_Nq4096_wmax320",
        "data/smallomega_direct_lines_Nw32768_Nq2048_wmax640",
    ],
    output_path::AbstractString="data/pi00_direct_analysis.jld2",
    output_plot::AbstractString="note/pi00_direct_analysis.pdf",
)
    qs, dq = uniform_momentum_grid(Nq)
    results = NamedTuple[]

    for Γraw in Γs
        Γ = Float64(Γraw)
        ωmax = max(min_ωmax, bandwidth_factor * Γ)
        ωs, dω = midpoint_frequency_grid_from_spacing(ωmax, dω_target)
        comp = pi00_components_for_gamma(Γ, ωs, dω, qs; r=r, η=η)
        tail = tail_correction_pi00(ωmax)
        tail_pm = 0.0
        tail_pp_mm = tail_correction_pi00_pp_or_mm(ωmax)

        push!(results, (
            Γ=Γ,
            ωmax=ωmax,
            Nω=length(ωs),
            Nq=Nq,
            dω=dω,
            dq=dq,
            Π00=comp.Π00,
            Πpp00=comp.Πpp00,
            Πmixed00=comp.Πmixed00,
            Πmm00=comp.Πmm00,
            rotated_total=comp.Πpp00 + comp.Πmixed00 + comp.Πmm00,
            ΓΠ00=Γ * comp.Π00,
            ΓΠpp00=Γ * comp.Πpp00,
            ΓΠmixed00=Γ * comp.Πmixed00,
            ΓΠmm00=Γ * comp.Πmm00,
            corrected_ΓΠ00=Γ * (comp.Π00 + tail),
            corrected_ΓΠpp00=Γ * (comp.Πpp00 + tail_pp_mm),
            corrected_ΓΠmixed00=Γ * (comp.Πmixed00 + tail_pm),
            corrected_ΓΠmm00=Γ * (comp.Πmm00 + tail_pp_mm),
        ))
    end

    overlap_rows = NamedTuple[]
    for dir in overlap_dirs
        append!(overlap_rows, load_existing_pi00_points(dir))
    end

    plot_pi00(results, overlap_rows, output_plot)
    jldsave(
        output_path;
        Γs=Float64.(Γs),
        Nq,
        dω_target,
        bandwidth_factor,
        min_ωmax,
        r,
        η,
        results,
        overlap_rows,
        output_plot,
    )

    println("saved ", output_path)
    println("saved ", output_plot)
    for row in results
        println(
            @sprintf(
                "Γ=%.1f ωmax=%.1f Nω=%d raw=%.8f corr=%.8f pp=%.8f mixed=%.8f mm=%.8f total(rot)=%.8f",
                row.Γ,
                row.ωmax,
                row.Nω,
                row.ΓΠ00,
                row.corrected_ΓΠ00,
                row.corrected_ΓΠpp00,
                row.corrected_ΓΠmixed00,
                row.corrected_ΓΠmm00,
                row.corrected_ΓΠpp00 + row.corrected_ΓΠmixed00 + row.corrected_ΓΠmm00,
            ),
        )
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
