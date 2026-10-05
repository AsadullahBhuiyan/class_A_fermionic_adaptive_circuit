include("InterfaceChern.jl")
using .InterfaceChern

function main(;
    green_file::Union{Nothing,AbstractString}=nothing,
    marker_file::Union{Nothing,AbstractString}=nothing,
    index_file::Union{Nothing,AbstractString}=nothing,
    output_prefix::Union{Nothing,AbstractString}=nothing,
    comparison_output::Union{Nothing,AbstractString}=nothing,
)
    outputs = NamedTuple[]

    if !isnothing(green_file) && !isnothing(marker_file)
        result = plot_interface_results(
            green_file=String(green_file),
            marker_file=String(marker_file),
            output_prefix=output_prefix,
        )
        push!(outputs, result)
        println("saved ", result.marker_pdf)
        println("saved ", result.spectral_pdf)
    end

    if !isnothing(index_file)
        result = InterfaceChern.plot_scan_marker_comparison(String(index_file); output_pdf=comparison_output)
        println("saved ", result)
    end

    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
