const SIGMA_X = ComplexF64[0 1; 1 0]
const SIGMA_Y = ComplexF64[0 -im; im 0]
const SIGMA_Z = ComplexF64[1 0; 0 -1]
const ORBITAL_I = Matrix{ComplexF64}(I, 2, 2)

function float_token(x::Real; digits::Int=3)
    token = @sprintf("%.*f", digits, float(x))
    token = replace(token, "-" => "m")
    return replace(token, "." => "p")
end

function case_tag(;
    r_ti::Real,
    r_triv::Real,
    M::Integer,
    Nkx::Integer,
    monitored::Bool,
    GammaM::Real,
    y_mon_lo::Integer,
    y_mon_hi::Integer,
    bulk_reset_enabled::Bool=false,
    Gamma_bulk_top::Real=0.0,
    Gamma_bulk_bottom::Real=0.0,
    bulk_reset_region_policy::AbstractString="auto_outside_monitor",
    y_reset_top_lo::Union{Nothing,Integer}=nothing,
    y_reset_bottom_hi::Union{Nothing,Integer}=nothing,
)
    base = string(
        monitored ? "mon" : "eq",
        "_rti", float_token(r_ti),
        "_rtriv", float_token(r_triv),
        "_M", Int(M),
        "_Nkx", Int(Nkx),
    )
    if monitored
        base *= string(
            "_Gm", float_token(GammaM),
            "_y", Int(y_mon_lo),
            "to",
            Int(y_mon_hi),
        )
    end
    if bulk_reset_enabled
        top_token = isnothing(y_reset_top_lo) ? "none" : string("ge", Int(y_reset_top_lo))
        bottom_token = isnothing(y_reset_bottom_hi) ? "none" : string("le", Int(y_reset_bottom_hi))
        base *= string(
            "_br", replace(String(bulk_reset_region_policy), ":" => ""),
            "_Gt", float_token(Gamma_bulk_top),
            "_Gb", float_token(Gamma_bulk_bottom),
            "_rt", top_token,
            "_rb", bottom_token,
        )
    end
    return base
end

row_values(M::Integer) = collect(-Int(M):(Int(M) - 1))

row_to_offset(y::Integer, M::Integer) = Int(y) + Int(M) + 1

function row_block_range_from_offset(offset::Integer)
    start = 2 * (Int(offset) - 1) + 1
    return start:(start + 1)
end

row_block_range(y::Integer, M::Integer) = row_block_range_from_offset(row_to_offset(y, M))

function row_position_in_ids(y::Integer, row_ids::AbstractVector{<:Integer})
    pos = findfirst(==(Int(y)), row_ids)
    isnothing(pos) && error("row $(Int(y)) is not present in the provided row id list")
    return pos
end

function matrix_indices_for_rows(rows::AbstractVector{<:Integer}, row_ids::AbstractVector{<:Integer})
    indices = Int[]
    for y in rows
        block = row_block_range_from_offset(row_position_in_ids(y, row_ids))
        append!(indices, block)
    end
    return indices
end

function select_rows_in_bounds(rows, M::Integer)
    ymin = -Int(M)
    ymax = Int(M) - 1
    selected = Int[]
    for y in rows
        yi = Int(y)
        if ymin <= yi <= ymax
            push!(selected, yi)
        end
    end
    sort!(unique!(selected))
    return selected
end

function default_probe_rows(M::Integer)
    Mm = Int(M)
    candidates = [
        -Mm,
        -max(1, Mm ÷ 2),
        -1,
        0,
        min(1, Mm - 1),
        max(0, Mm ÷ 2),
        Mm - 1,
    ]
    return select_rows_in_bounds(candidates, Mm)
end

function default_interface_rows(M::Integer, halfwidth::Integer)
    hw = max(0, Int(halfwidth))
    return select_rows_in_bounds((-hw):(hw - 1), Int(M))
end

function default_deep_rows(M::Integer)
    Mm = Int(M)
    cutoff = max(1, Mm ÷ 2)
    triv_rows = select_rows_in_bounds((-Mm):(-cutoff), Mm)
    ti_rows = select_rows_in_bounds(cutoff:(Mm - 1), Mm)
    if isempty(triv_rows)
        triv_rows = select_rows_in_bounds([-Mm], Mm)
    end
    if isempty(ti_rows)
        ti_rows = select_rows_in_bounds([Mm - 1], Mm)
    end
    return (; triv_rows, ti_rows)
end

function row_position_operator(M::Integer)
    ys = row_values(M)
    dim = 4 * Int(M)
    Y = zeros(ComplexF64, dim, dim)
    for (offset, y) in enumerate(ys)
        block = row_block_range_from_offset(offset)
        Y[block, block] .= ComplexF64(y) .* ORBITAL_I
    end
    return Y
end

function monitored_rows(M::Integer, y_mon_lo::Integer, y_mon_hi::Integer)
    lo = min(Int(y_mon_lo), Int(y_mon_hi))
    hi = max(Int(y_mon_lo), Int(y_mon_hi))
    return select_rows_in_bounds(lo:hi, Int(M))
end

function default_scan_tag()
    return @sprintf(
        "scan_%04d%02d%02d_%02d%02d%02d",
        Dates.year(Dates.now()),
        Dates.month(Dates.now()),
        Dates.day(Dates.now()),
        Dates.hour(Dates.now()),
        Dates.minute(Dates.now()),
        Dates.second(Dates.now()),
    )
end

fieldget(obj::NamedTuple, name::Symbol) = getproperty(obj, name)
fieldget(obj::AbstractDict, name::Symbol) = haskey(obj, name) ? obj[name] : obj[String(name)]
