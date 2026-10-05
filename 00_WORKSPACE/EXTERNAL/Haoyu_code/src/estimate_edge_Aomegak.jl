using JLD2
using Plots
using Printf
using Statistics
using Base.Threads: @threads, maxthreadid, threadid

include("taylor_fit_convolution.jl")

function midpoint_frequency_grid(Nω::Int, ωmax::Float64)
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

@inline function overlap_entries(
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
    h11::ComplexF64,
    h12::ComplexF64,
    h21::ComplexF64,
    h22::ComplexF64,
)
    return g11 * conj(h11) + g12 * conj(h12) + g21 * conj(h21) + g22 * conj(h22)
end

function component_direct_mixed_grid_for_gamma(
    Γ::Float64,
    ωs::Vector{Float64},
    dω::Float64,
    qs::Vector{Float64},
    Ω_values::Vector{Float64},
    k_values::Vector{Float64};
    r::Float64=1.0,
    η::Float64=1e-4,
)
    nt = maxthreadid()
    nΩ = length(Ω_values)
    nk = length(k_values)

    partial_full = zeros(ComplexF64, nt, nΩ, nk)
    partial_pp = zeros(ComplexF64, nt, nΩ, nk)
    partial_pm = zeros(ComplexF64, nt, nΩ, nk)
    partial_mp = zeros(ComplexF64, nt, nΩ, nk)
    partial_mm = zeros(ComplexF64, nt, nΩ, nk)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        @inbounds for ω in ωs
            g11, g12, g21, g22 = monitored_retarded_green_entries(ω, q, r, Γ, η)
            gpp, gpm, gmp, gmm = rotate_to_sigma_x_entries(g11, g12, g21, g22)

            for iΩ in eachindex(Ω_values)
                Ω = Ω_values[iΩ]
                for ik in eachindex(k_values)
                    k = k_values[ik]
                    h11, h12, h21, h22 = monitored_retarded_green_entries(ω - Ω, q - k, r, Γ, η)
                    hpp, hpm, hmp, hmm = rotate_to_sigma_x_entries(h11, h12, h21, h22)

                    partial_full[tid, iΩ, ik] += overlap_entries(g11, g12, g21, g22, h11, h12, h21, h22)
                    partial_pp[tid, iΩ, ik] += gpp * conj(hpp)
                    partial_pm[tid, iΩ, ik] += gpm * conj(hpm)
                    partial_mp[tid, iΩ, ik] += gmp * conj(hmp)
                    partial_mm[tid, iΩ, ik] += gmm * conj(hmm)
                end
            end
        end
    end

    prefactor = dω / (2π * length(qs))
    return (
        full=prefactor .* dropdims(sum(partial_full; dims=1); dims=1),
        pp=prefactor .* dropdims(sum(partial_pp; dims=1); dims=1),
        pm=prefactor .* dropdims(sum(partial_pm; dims=1); dims=1),
        mp=prefactor .* dropdims(sum(partial_mp; dims=1); dims=1),
        mm=prefactor .* dropdims(sum(partial_mm; dims=1); dims=1),
    )
end

function odd_odd_projection(values::AbstractMatrix{<:Real})
    nΩ2, nk2 = size(values)
    @assert iseven(nΩ2) && iseven(nk2)
    nΩ = nΩ2 ÷ 2
    nk = nk2 ÷ 2
    out = Matrix{Float64}(undef, nΩ, nk)
    @inbounds for i in 1:nΩ
        ip = nΩ + i
        im = nΩ - i + 1
        for j in 1:nk
            jp = nk + j
            jm = nk - j + 1
            out[i, j] = 0.25 * (values[ip, jp] - values[ip, jm] - values[im, jp] + values[im, jm])
        end
    end
    return out
end

function fit_bilinear(Ω_pos::Vector{Float64}, k_pos::Vector{Float64}, values::Matrix{Float64})
    x = Float64[]
    y = Float64[]
    @inbounds for i in eachindex(Ω_pos), j in eachindex(k_pos)
        push!(x, Ω_pos[i] * k_pos[j])
        push!(y, values[i, j])
    end
    coeff = sum(x .* y) / sum(abs2, x)
    fitted = reshape(coeff .* x, length(Ω_pos), length(k_pos))
    rel_rms = sqrt(mean(abs2, vec(values .- fitted))) / max(maximum(abs.(values)), eps())
    return (coeff=coeff, fitted=fitted, rel_rms=rel_rms)
end

function signed_values_from_positive(pos::Vector{Float64})
    return vcat(-reverse(pos), pos)
end

function plot_mixed_ratio(results::AbstractVector, output_pdf::AbstractString)
    panels = Plots.Plot[]
    for row in results
        xvals = vec([sqrt(Ω * k) for Ω in row.Ω_pos, k in row.k_pos])
        order = sortperm(xvals)

        function flattened_ratio(mat)
            vals = vec(mat)
            den = vec([Ω * k for Ω in row.Ω_pos, k in row.k_pos])
            return vals[order] ./ den[order]
        end

        x = xvals[order]
        plt = plot(
            x,
            flattened_ratio(row.full_oddodd);
            xscale=:log10,
            label="full direct",
            color=:black,
            marker=:circle,
            markersize=4,
            markerstrokewidth=0,
            linewidth=2.0,
            xlabel="sqrt(Ω k)",
            ylabel="odd-odd projection / (Ω k)",
            title="Γ = $(round(row.Γ; digits=1))",
        )
        plot!(plt, x, flattened_ratio(row.pp_oddodd); label="exact ++", color=:blue, linewidth=2.0, marker=:diamond, markersize=3, markerstrokewidth=0)
        plot!(plt, x, flattened_ratio(row.mixed_oddodd); label="off-diagonal", color=:green, linewidth=2.0, marker=:utriangle, markersize=3, markerstrokewidth=0)
        plot!(plt, x, flattened_ratio(row.mm_oddodd); label="exact --", color=:red, linewidth=2.0, marker=:square, markersize=3, markerstrokewidth=0)
        plot!(plt, x, flattened_ratio(row.total_oddodd); label="rotated total", color=:orange, linewidth=2.0, linestyle=:dash)
        push!(panels, plt)
    end
    plt = plot(panels...; layout=(length(results), 1), size=(1000, 340 * length(results)))
    savefig(plt, output_pdf)
end

function main(;
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    Ω_pos::AbstractVector=[5e-4, 1e-3, 1.5e-3],
    k_pos::AbstractVector=[5e-4, 1e-3, 2e-3],
    Nω::Int=16384,
    Nq::Int=4096,
    ωmax::Float64=640.0,
    r::Float64=1.0,
    η::Float64=1e-4,
    output_path::AbstractString="data/edge_Aomegak_estimate.jld2",
    output_plot::AbstractString="note/edge_Aomegak_model_comparison.pdf",
)
    Ω_pos_v = Float64.(collect(Ω_pos))
    k_pos_v = Float64.(collect(k_pos))
    Ω_values = signed_values_from_positive(Ω_pos_v)
    k_values = signed_values_from_positive(k_pos_v)
    ωs, dω = midpoint_frequency_grid(Nω, ωmax)
    qs, _ = uniform_momentum_grid(Nq)

    results = NamedTuple[]
    for Γraw in Γs
        Γ = Float64(Γraw)
        comp = component_direct_mixed_grid_for_gamma(Γ, ωs, dω, qs, Ω_values, k_values; r=r, η=η)

        full_vals = real.(Γ .* comp.full)
        pp_vals = real.(Γ .* comp.pp)
        pm_vals = real.(Γ .* comp.pm)
        mp_vals = real.(Γ .* comp.mp)
        mm_vals = real.(Γ .* comp.mm)
        mixed_vals = pm_vals .+ mp_vals
        total_vals = pp_vals .+ mixed_vals .+ mm_vals

        full_oddodd = odd_odd_projection(full_vals)
        pp_oddodd = odd_odd_projection(pp_vals)
        pm_oddodd = odd_odd_projection(pm_vals)
        mp_oddodd = odd_odd_projection(mp_vals)
        mixed_oddodd = odd_odd_projection(mixed_vals)
        mm_oddodd = odd_odd_projection(mm_vals)
        total_oddodd = odd_odd_projection(total_vals)

        fit_full = fit_bilinear(Ω_pos_v, k_pos_v, full_oddodd)
        fit_pp = fit_bilinear(Ω_pos_v, k_pos_v, pp_oddodd)
        fit_pm = fit_bilinear(Ω_pos_v, k_pos_v, pm_oddodd)
        fit_mp = fit_bilinear(Ω_pos_v, k_pos_v, mp_oddodd)
        fit_mixed = fit_bilinear(Ω_pos_v, k_pos_v, mixed_oddodd)
        fit_mm = fit_bilinear(Ω_pos_v, k_pos_v, mm_oddodd)
        fit_total = fit_bilinear(Ω_pos_v, k_pos_v, total_oddodd)

        push!(results, (
            Γ=Γ,
            Ω_pos=Ω_pos_v,
            k_pos=k_pos_v,
            Ω_values=Ω_values,
            k_values=k_values,
            full_values=full_vals,
            pp_values=pp_vals,
            pm_values=pm_vals,
            mp_values=mp_vals,
            mixed_values=mixed_vals,
            mm_values=mm_vals,
            total_values=total_vals,
            full_oddodd=full_oddodd,
            pp_oddodd=pp_oddodd,
            pm_oddodd=pm_oddodd,
            mp_oddodd=mp_oddodd,
            mixed_oddodd=mixed_oddodd,
            mm_oddodd=mm_oddodd,
            total_oddodd=total_oddodd,
            full_coeff=fit_full.coeff,
            pp_coeff=fit_pp.coeff,
            pm_coeff=fit_pm.coeff,
            mp_coeff=fit_mp.coeff,
            mixed_coeff=fit_mixed.coeff,
            mm_coeff=fit_mm.coeff,
            total_coeff=fit_total.coeff,
            full_rel_rms=fit_full.rel_rms,
            pp_rel_rms=fit_pp.rel_rms,
            pm_rel_rms=fit_pm.rel_rms,
            mp_rel_rms=fit_mp.rel_rms,
            mixed_rel_rms=fit_mixed.rel_rms,
            mm_rel_rms=fit_mm.rel_rms,
            total_rel_rms=fit_total.rel_rms,
        ))
    end

    plot_mixed_ratio(results, output_plot)
    jldsave(
        output_path;
        Γs=Float64.(collect(Γs)),
        Ω_pos=Ω_pos_v,
        k_pos=k_pos_v,
        Nω,
        Nq,
        ωmax,
        r,
        η,
        output_plot,
        results,
    )

    println("saved ", output_path)
    println("saved ", output_plot)
    for row in results
        println(
            @sprintf(
                "Γ=%.1f full=%.8f exact_pp=%.8f exact_pm=%.8f exact_mp=%.8f mixed=%.8f exact_mm=%.8f rotated_total=%.8f",
                row.Γ,
                row.full_coeff,
                row.pp_coeff,
                row.pm_coeff,
                row.mp_coeff,
                row.mixed_coeff,
                row.mm_coeff,
                row.total_coeff,
            ),
        )
        println(
            @sprintf(
                "Γ=%.1f rel_rms: full=%.4e pp=%.4e mixed=%.4e mm=%.4e total=%.4e",
                row.Γ,
                row.full_rel_rms,
                row.pp_rel_rms,
                row.mixed_rel_rms,
                row.mm_rel_rms,
                row.total_rel_rms,
            ),
        )
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
