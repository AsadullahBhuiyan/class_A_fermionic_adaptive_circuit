function default_marker_output_path(
    green_file::AbstractString;
    window_M::Union{Nothing,Integer}=nothing,
)
    if isnothing(window_M)
        return joinpath(dirname(green_file), "marker.jld2")
    end
    return joinpath(dirname(green_file), "marker_windowM$(Int(window_M)).jld2")
end

function centered_window_data(
    Ckx::Array{ComplexF64,3},
    row_ids::AbstractVector{<:Integer},
    M_parent::Integer,
    window_M::Integer,
)
    Mp = Int(M_parent)
    Mw = Int(window_M)
    1 <= Mw <= Mp || error("window_M must satisfy 1 <= window_M <= parent M")

    if Mw == Mp
        return Ckx, Int.(row_ids), Mw
    end

    target_rows = select_rows_in_bounds((-Mw):(Mw - 1), Mp)
    row_to_offset = Dict(Int(y) => idx for (idx, y) in pairs(row_ids))
    if !all(haskey(row_to_offset, y) for y in target_rows)
        error("requested centered marker window is not contained in the saved row set")
    end

    matrix_indices = Int[]
    for y in target_rows
        block = row_block_range_from_offset(row_to_offset[y])
        append!(matrix_indices, block)
    end

    return Ckx[matrix_indices, matrix_indices, :], target_rows, Mw
end

function row_marker_profile(
    Ckx::Array{ComplexF64,3},
    row_ids::AbstractVector{<:Integer},
    M::Integer;
    dk::Real,
    derivative_scheme::Symbol=:spectral,
)
    dC = if derivative_scheme == :centered2
        centered_periodic_derivative(Ckx, dk)
    elseif derivative_scheme == :spectral
        spectral_periodic_derivative(Ckx)
    else
        error("unsupported derivative_scheme $(derivative_scheme)")
    end
    Y = row_position_operator(M)
    Ly = length(row_ids)
    Nkx = size(Ckx, 3)
    c_rows = zeros(Float64, Ly)

    for ik in 1:Nkx
        C = Ckx[:, :, ik]
        dCk = dC[:, :, ik]
        comm_y = Y * C - C * Y
        kernel = C * (dCk * comm_y - comm_y * dCk)
        for (offset, _) in enumerate(row_ids)
            block = row_block_range_from_offset(offset)
            c_rows[offset] += real(2π * tr(kernel[block, block])) / Nkx
        end
    end

    return c_rows
end

function compute_row_chern_marker(;
    green_file::AbstractString,
    output_file::Union{Nothing,AbstractString}=nothing,
    deep_rows_ti::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    deep_rows_triv::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    derivative_scheme::Symbol=:spectral,
    window_M::Union{Nothing,Integer}=nothing,
)
    data = load(green_file)
    metadata = data["metadata"]
    parent_Ckx = ComplexF64.(data["Ckx"])
    parent_row_ids = Int.(data["row_indices"])
    dk = Float64(fieldget(metadata, :dk))
    M_parent = Int(fieldget(metadata, :M))
    monitored = Bool(fieldget(metadata, :monitored))
    effective_window_M = isnothing(window_M) ? M_parent : Int(window_M)

    Ckx, row_ids, M = centered_window_data(parent_Ckx, parent_row_ids, M_parent, effective_window_M)

    c_rows = row_marker_profile(Ckx, row_ids, M; dk=dk, derivative_scheme=derivative_scheme)
    defaults = default_deep_rows(M)
    ti_rows = isnothing(deep_rows_ti) ? defaults.ti_rows : select_rows_in_bounds(deep_rows_ti, M)
    triv_rows = isnothing(deep_rows_triv) ? defaults.triv_rows : select_rows_in_bounds(deep_rows_triv, M)

    row_to_position = Dict(y => idx for (idx, y) in pairs(row_ids))
    ti_indices = [row_to_position[y] for y in ti_rows]
    triv_indices = [row_to_position[y] for y in triv_rows]
    c_ti_avg = mean(c_rows[ti_indices])
    c_triv_avg = mean(c_rows[triv_indices])
    marker_kind = monitored ? :monitored_diagnostic : :equilibrium

    resolved_output = isnothing(output_file) ?
        default_marker_output_path(green_file; window_M=isnothing(window_M) ? nothing : effective_window_M) :
        String(output_file)
    mkpath(dirname(resolved_output))

    jldsave(
        resolved_output;
        green_file=String(green_file),
        case_tag=String(fieldget(metadata, :case_tag)),
        parent_M=M_parent,
        marker_window_M=M,
        row_indices=row_ids,
        c_rows=c_rows,
        c_ti_avg=Float64(c_ti_avg),
        c_triv_avg=Float64(c_triv_avg),
        deep_rows_ti=ti_rows,
        deep_rows_triv=triv_rows,
        marker_kind=String(marker_kind),
        derivative_scheme=String(derivative_scheme),
        monitored=monitored,
    )

    return (
        marker_file=resolved_output,
        case_tag=String(fieldget(metadata, :case_tag)),
        marker_kind=marker_kind,
        parent_M=M_parent,
        marker_window_M=M,
        c_ti_avg=Float64(c_ti_avg),
        c_triv_avg=Float64(c_triv_avg),
    )
end
