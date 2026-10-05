using JLD2
using Printf

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function load_scaling_scan(scan_dir::AbstractString)
    metadata = load(joinpath(scan_dir, "metadata.jld2"))
    Γs = Float64.(metadata["Γs"])
    ν_values = Float64.(metadata["ν_values"])
    κ_values = Float64.(metadata["κ_values"])

    payloads = Dict{Float64, NamedTuple}()
    for Γ in Γs
        data = load(joinpath(scan_dir, "gamma_$(gamma_token(Γ)).jld2"))
        scaled_local = data["scaled_local_grid"]
        νscaled = ν_values .* scaled_local
        κscaled = permutedims(κ_values) .* scaled_local
        payloads[Γ] = (
            anti_grid=data["anti_grid"],
            sym_grid=data["sym_grid"],
            scaled_local_grid=scaled_local,
            νscaled_grid=νscaled,
            κscaled_grid=κscaled,
            max_sym=maximum(abs.(data["sym_grid"])),
        )
    end

    return (
        scan_dir=String(scan_dir),
        Γs=Γs,
        ν_values=ν_values,
        κ_values=κ_values,
        umax=Float64(metadata["umax"]),
        ymax=Float64(metadata["ymax"]),
        du=Float64(metadata["du"]),
        dy=Float64(metadata["dy"]),
        payloads=payloads,
    )
end

function save_scaling_scan_analysis(scan_dir::AbstractString; output_path::Union{Nothing,AbstractString}=nothing)
    analysis = load_scaling_scan(scan_dir)
    resolved_output = isnothing(output_path) ?
        joinpath(scan_dir, "analysis.jld2") :
        String(output_path)
    jldsave(
        resolved_output;
        scan_dir=analysis.scan_dir,
        Γs=analysis.Γs,
        ν_values=analysis.ν_values,
        κ_values=analysis.κ_values,
        umax=analysis.umax,
        ymax=analysis.ymax,
        du=analysis.du,
        dy=analysis.dy,
        payloads=analysis.payloads,
    )
    return resolved_output
end

function print_scaling_scan_summary(scan_dir::AbstractString)
    analysis = load_scaling_scan(scan_dir)
    println("scan_dir = ", analysis.scan_dir)
    println(
        "umax = ", analysis.umax,
        ", ymax = ", analysis.ymax,
        ", du = ", analysis.du,
        ", dy = ", analysis.dy,
    )
    println("ν_values = ", analysis.ν_values)
    println("κ_values = ", analysis.κ_values)

    for Γ in analysis.Γs
        payload = analysis.payloads[Γ]
        println()
        println("Γ = ", Γ)
        println("max|sym| = ", payload.max_sym)
        println("scaled_local_grid =")
        show(stdout, "text/plain", payload.scaled_local_grid)
        println()
        println("ν * scaled_local_grid =")
        show(stdout, "text/plain", payload.νscaled_grid)
        println()
    end

    return nothing
end

function main(args=ARGS)
    if isempty(args)
        error("usage: julia --project src/analyze_touching_wzw_scaling_scan.jl <scan_dir>")
    end
    scan_dir = args[1]
    print_scaling_scan_summary(scan_dir)
    output_path = save_scaling_scan_analysis(scan_dir)
    println()
    println("saved ", output_path)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
