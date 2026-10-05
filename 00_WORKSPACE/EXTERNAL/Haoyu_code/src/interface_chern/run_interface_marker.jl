using JLD2

include("InterfaceChern.jl")
using .InterfaceChern

function main(;
    green_file::Union{Nothing,AbstractString}=nothing,
    index_file::Union{Nothing,AbstractString}=nothing,
    output_file::Union{Nothing,AbstractString}=nothing,
    deep_rows_ti=nothing,
    deep_rows_triv=nothing,
    derivative_scheme::Symbol=:centered2,
    window_M::Union{Nothing,Integer}=nothing,
)
    results = NamedTuple[]

    if !isnothing(green_file)
        result = compute_row_chern_marker(
            green_file=String(green_file),
            output_file=output_file,
            deep_rows_ti=deep_rows_ti,
            deep_rows_triv=deep_rows_triv,
            derivative_scheme=derivative_scheme,
            window_M=window_M,
        )
        push!(results, result)
        println("saved ", result.marker_file)
    end

    if !isnothing(index_file)
        scan_data = load(String(index_file))
        entries = scan_data["entries"]
        marker_entries = NamedTuple[]
        for entry in entries
            result = compute_row_chern_marker(
                green_file=String(InterfaceChern.fieldget(entry, :green_file)),
                deep_rows_ti=deep_rows_ti,
                deep_rows_triv=deep_rows_triv,
                derivative_scheme=derivative_scheme,
                window_M=window_M,
            )
            push!(results, result)
            push!(marker_entries, (
                case_tag=result.case_tag,
                marker_file=result.marker_file,
                marker_kind=String(result.marker_kind),
                c_ti_avg=result.c_ti_avg,
                c_triv_avg=result.c_triv_avg,
            ))
            println("saved ", result.marker_file)
        end
        marker_index_file = joinpath(dirname(String(index_file)), "marker_index.jld2")
        jldsave(marker_index_file; index_file=String(index_file), marker_entries)
        println("saved ", marker_index_file)
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
