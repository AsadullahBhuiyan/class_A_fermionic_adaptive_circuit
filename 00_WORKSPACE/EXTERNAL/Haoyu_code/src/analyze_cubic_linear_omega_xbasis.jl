using JLD2
using Printf

include("analyze_cubic_triangle_large_gamma.jl")

function load_xbasis_breakdown(output_dir::AbstractString, Γ::Real)
    return load(joinpath(output_dir, "gamma_$(gamma_token(Γ)).jld2"))
end

function format_complex(z::ComplexF64)
    return @sprintf("%.15e %+.15ei", real(z), imag(z))
end

function top_sector_cycle_terms(
    values::Array{ComplexF64,2},
    sector_labels::Vector{String},
    cycle_labels::Vector{String};
    topn::Int=12,
)
    ranked = NamedTuple[]
    for isector in eachindex(sector_labels)
        for icycle in eachindex(cycle_labels)
            push!(ranked, (
                sector=sector_labels[isector],
                cycle=cycle_labels[icycle],
                value=values[isector, icycle],
                absimag=abs(imag(values[isector, icycle])),
            ))
        end
    end
    sort!(ranked; by=entry -> entry.absimag, rev=true)
    return ranked[1:min(topn, length(ranked))]
end

function write_xbasis_report(
    io::IO,
    Γ::Float64,
    full_summary::NamedTuple,
    breakdown,
)
    sector_labels = Vector{String}(breakdown["sector_labels"])
    cycle_labels = Vector{String}(breakdown["cycle_labels"])
    scaled_dΩ1 = Array{ComplexF64}(breakdown["scaled_dΩ1"])
    scaled_dΩ2 = Array{ComplexF64}(breakdown["scaled_dΩ2"])
    sector_sum_dΩ1 = Vector{ComplexF64}(breakdown["sector_sum_dΩ1"])
    sector_sum_dΩ2 = Vector{ComplexF64}(breakdown["sector_sum_dΩ2"])
    cycle_sum_dΩ1 = Vector{ComplexF64}(breakdown["cycle_sum_dΩ1"])
    cycle_sum_dΩ2 = Vector{ComplexF64}(breakdown["cycle_sum_dΩ2"])
    total_dΩ1 = ComplexF64(breakdown["total_dΩ1"])
    total_dΩ2 = ComplexF64(breakdown["total_dΩ2"])

    println(io, "Γ = ", Γ)
    println(io, "xbasis total Γ^3 dΩ1 = ", format_complex(total_dΩ1))
    println(io, "full-trace total Γ^3 dΩ1 = ", format_complex(full_summary.scaled_dΩ1))
    println(io, "difference dΩ1 = ", format_complex(total_dΩ1 - full_summary.scaled_dΩ1))
    println(io, "xbasis total Γ^3 dΩ2 = ", format_complex(total_dΩ2))
    println(io, "full-trace total Γ^3 dΩ2 = ", format_complex(full_summary.scaled_dΩ2))
    println(io, "difference dΩ2 = ", format_complex(total_dΩ2 - full_summary.scaled_dΩ2))
    println(io)

    println(io, "Sector sums for Γ^3 dΩ1")
    for isector in eachindex(sector_labels)
        println(io, @sprintf(
            "  %-3s  %s",
            sector_labels[isector],
            format_complex(sector_sum_dΩ1[isector]),
        ))
    end
    println(io)

    println(io, "Cycle sums for Γ^3 dΩ1")
    for icycle in eachindex(cycle_labels)
        println(io, @sprintf(
            "  %-3s  %s",
            cycle_labels[icycle],
            format_complex(cycle_sum_dΩ1[icycle]),
        ))
    end
    println(io)

    println(io, "Sector-cycle table for Γ^3 dΩ1")
    for isector in eachindex(sector_labels)
        for icycle in eachindex(cycle_labels)
            println(io, @sprintf(
                "  %-3s %-3s  %s",
                sector_labels[isector],
                cycle_labels[icycle],
                format_complex(scaled_dΩ1[isector, icycle]),
            ))
        end
    end
    println(io)

    println(io, "Sector sums for Γ^3 dΩ2")
    for isector in eachindex(sector_labels)
        println(io, @sprintf(
            "  %-3s  %s",
            sector_labels[isector],
            format_complex(sector_sum_dΩ2[isector]),
        ))
    end
    println(io)

    println(io, "Cycle sums for Γ^3 dΩ2")
    for icycle in eachindex(cycle_labels)
        println(io, @sprintf(
            "  %-3s  %s",
            cycle_labels[icycle],
            format_complex(cycle_sum_dΩ2[icycle]),
        ))
    end
    println(io)

    println(io, "Top |Im| sector-cycle terms for Γ^3 dΩ1")
    for entry in top_sector_cycle_terms(scaled_dΩ1, sector_labels, cycle_labels)
        println(io, @sprintf(
            "  %-3s %-3s  %s",
            entry.sector,
            entry.cycle,
            format_complex(entry.value),
        ))
    end
end

function analyze_xbasis_breakdown(
    xbasis_dir::AbstractString;
    full_dir::AbstractString="data/cubic_triangle_scan_dyson_sign_Nw12800_Nk256_wmax320p000",
    Γs::AbstractVector{<:Real}=[20.0, 40.0, 80.0],
    report_dir::Union{Nothing,AbstractString}=nothing,
)
    resolved_report_dir = isnothing(report_dir) ? xbasis_dir : String(report_dir)
    mkpath(resolved_report_dir)

    for Γ in Float64.(Γs)
        breakdown = load_xbasis_breakdown(xbasis_dir, Γ)
        full_summary = extract_summary(full_dir, Γ)
        report_path = joinpath(resolved_report_dir, "report_gamma_$(gamma_token(Γ)).txt")
        open(report_path, "w") do io
            write_xbasis_report(io, Γ, full_summary, breakdown)
        end
        println("wrote ", report_path)
    end
end

main(; kwargs...) = analyze_xbasis_breakdown("data/cubic_linear_omega_xbasis_dyson_sign_Nw12800_Nk256_dw0p050"; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
