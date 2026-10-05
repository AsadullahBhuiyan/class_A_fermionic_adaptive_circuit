using Dates
using JLD2

include("InterfaceChern.jl")
using .InterfaceChern

function as_vector(values)
    return collect(values)
end

function main(;
    output_root::AbstractString="data/interface_chern",
    scan_tag::Union{Nothing,AbstractString}=nothing,
    r_ti_values=(1.0,),
    r_triv_values=(3.0,),
    M_values=(8,),
    Nkx_values=(121,),
    monitored_values=(false,),
    GammaM_values=(0.5,),
    t0::Real=1.0,
    omega_dense::Real=4.0,
    omega_max::Real=12.0,
    domega_dense::Real=0.02,
    domega_coarse::Real=0.10,
    eta::Real=NaN,
    y_mon_lo::Integer=0,
    y_mon_hi::Integer=0,
    bulk_reset_enabled::Bool=false,
    Gamma_bulk_top::Real=0.0,
    Gamma_bulk_bottom::Real=0.0,
    bulk_reset_region_policy::Symbol=:auto_outside_monitor,
    y_reset_top_lo::Union{Nothing,Integer}=nothing,
    y_reset_bottom_hi::Union{Nothing,Integer}=nothing,
    Nky_projector::Integer=2048,
    assume_monitor_half_filling::Bool=true,
    monitor_mixing::Real=0.5,
    monitor_tol::Real=1e-8,
    monitor_maxiter::Integer=200,
    interface_window_halfwidth::Integer=1,
    probe_rows=nothing,
    lead_tol::Real=1e-12,
    lead_maxiter::Integer=200,
)
    mkpath(output_root)
    entries = NamedTuple[]

    for r_ti in as_vector(r_ti_values)
        for r_triv in as_vector(r_triv_values)
            for M in as_vector(M_values)
                for Nkx in as_vector(Nkx_values)
                    for monitored in Bool.(as_vector(monitored_values))
                        gamma_values = monitored ? as_vector(GammaM_values) : [0.0]
                        for GammaM in gamma_values
                            result = solve_interface_green(
                                output_root=output_root,
                                r_ti=r_ti,
                                r_triv=r_triv,
                                M=M,
                                Nkx=Nkx,
                                t0=t0,
                                omega_dense=omega_dense,
                                omega_max=omega_max,
                                domega_dense=domega_dense,
                                domega_coarse=domega_coarse,
                                eta=eta,
                                monitored=monitored,
                                GammaM=GammaM,
                                bulk_reset_enabled=bulk_reset_enabled,
                                Gamma_bulk_top=Gamma_bulk_top,
                                Gamma_bulk_bottom=Gamma_bulk_bottom,
                                bulk_reset_region_policy=bulk_reset_region_policy,
                                y_reset_top_lo=y_reset_top_lo,
                                y_reset_bottom_hi=y_reset_bottom_hi,
                                Nky_projector=Nky_projector,
                                assume_monitor_half_filling=assume_monitor_half_filling,
                                y_mon_lo=y_mon_lo,
                                y_mon_hi=y_mon_hi,
                                monitor_mixing=monitor_mixing,
                                monitor_tol=monitor_tol,
                                monitor_maxiter=monitor_maxiter,
                                interface_window_halfwidth=interface_window_halfwidth,
                                probe_rows=probe_rows,
                                lead_tol=lead_tol,
                                lead_maxiter=lead_maxiter,
                            )
                            push!(entries, (
                                case_tag=result.case_tag,
                                case_dir=result.case_dir,
                                green_file=result.green_file,
                                metadata_file=result.metadata_file,
                                monitored=monitored,
                                GammaM=Float64(GammaM),
                                M=Int(M),
                                Nkx=Int(Nkx),
                                r_ti=Float64(r_ti),
                                r_triv=Float64(r_triv),
                                bulk_reset_enabled=Bool(bulk_reset_enabled),
                                Gamma_bulk_top=Float64(Gamma_bulk_top),
                                Gamma_bulk_bottom=Float64(Gamma_bulk_bottom),
                            ))
                        end
                    end
                end
            end
        end
    end

    index_file = nothing
    if length(entries) > 1 || !isnothing(scan_tag)
        scan_name = isnothing(scan_tag) ? InterfaceChern.default_scan_tag() : String(scan_tag)
        scan_dir = joinpath(output_root, scan_name)
        mkpath(scan_dir)
        index_file = joinpath(scan_dir, "index.jld2")
        jldsave(index_file; scan_dir, output_root=String(output_root), entries)
        println("saved ", index_file)
    end

    for entry in entries
        println("saved ", entry.green_file)
    end

    return (; entries, index_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
