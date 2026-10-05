function bulk_projector_symbol(kx::Real, ky::Real, r::Real; t0::Real=1.0)
    scale = Float64(t0)
    dx = scale * sin(Float64(kx))
    dy = scale * sin(Float64(ky))
    dz = scale * (Float64(r) - cos(Float64(kx)) - cos(Float64(ky)))
    energy = sqrt(dx^2 + dy^2 + dz^2)
    energy > 0 || error("bulk projector is undefined at a gap closing")

    projector = 0.5 .* (ORBITAL_I .- (dx .* SIGMA_X .+ dy .* SIGMA_Y .+ dz .* SIGMA_Z) ./ energy)
    complement = ORBITAL_I .- projector
    return projector, complement
end

function bulk_projector_kernel(
    kx::Real,
    dy::Integer,
    r::Real;
    t0::Real=1.0,
    Nky_projector::Integer=2048,
)
    nky = Int(Nky_projector)
    nky > 0 || throw(ArgumentError("Nky_projector must be positive"))

    kernel = zeros(ComplexF64, 2, 2)
    kys = range(-π, π; length=nky + 1)[1:end-1]
    for ky in kys
        projector, _ = bulk_projector_symbol(kx, ky, r; t0=t0)
        kernel .+= cis(Float64(ky) * Int(dy)) .* projector
    end
    return kernel ./ nky
end

function restricted_bulk_projector(
    kx::Real,
    rows::AbstractVector{<:Integer},
    r::Real;
    t0::Real=1.0,
    Nky_projector::Integer=2048,
)
    ordered_rows = sort(unique(Int.(rows)))
    nrows = length(ordered_rows)
    matrix = zeros(ComplexF64, 2 * nrows, 2 * nrows)
    kernel_cache = Dict{Int, Matrix{ComplexF64}}()

    for (irow, y) in pairs(ordered_rows)
        block_i = row_block_range_from_offset(irow)
        for (jrow, yp) in pairs(ordered_rows)
            dy = y - yp
            block_j = row_block_range_from_offset(jrow)
            kernel = get!(kernel_cache, dy) do
                bulk_projector_kernel(kx, dy, r; t0=t0, Nky_projector=Nky_projector)
            end
            matrix[block_i, block_j] .= kernel
        end
    end

    return matrix
end

function resolve_bulk_reset_regions(
    M::Integer;
    bulk_reset_enabled::Bool=false,
    bulk_reset_region_policy::Symbol=:auto_outside_monitor,
    monitored::Bool=false,
    y_mon_lo::Integer=0,
    y_mon_hi::Integer=0,
    y_reset_top_lo::Union{Nothing,Integer}=nothing,
    y_reset_bottom_hi::Union{Nothing,Integer}=nothing,
)
    if !bulk_reset_enabled
        return (
            reset_top_rows=Int[],
            reset_bottom_rows=Int[],
            y_reset_top_lo=nothing,
            y_reset_bottom_hi=nothing,
            bulk_reset_region_policy=bulk_reset_region_policy,
        )
    end

    bulk_reset_region_policy == :auto_outside_monitor ||
        throw(ArgumentError("unsupported bulk reset region policy $(bulk_reset_region_policy)"))

    mon_lo = min(Int(y_mon_lo), Int(y_mon_hi))
    mon_hi = max(Int(y_mon_lo), Int(y_mon_hi))

    top_lo = isnothing(y_reset_top_lo) ? (monitored ? mon_hi + 1 : 0) : Int(y_reset_top_lo)
    bottom_hi = isnothing(y_reset_bottom_hi) ? (monitored ? mon_lo - 1 : -1) : Int(y_reset_bottom_hi)

    reset_top_rows = select_rows_in_bounds(top_lo:(Int(M) - 1), M)
    reset_bottom_rows = select_rows_in_bounds((-Int(M)):bottom_hi, M)

    return (
        reset_top_rows=reset_top_rows,
        reset_bottom_rows=reset_bottom_rows,
        y_reset_top_lo=isempty(reset_top_rows) ? nothing : first(reset_top_rows),
        y_reset_bottom_hi=isempty(reset_bottom_rows) ? nothing : last(reset_bottom_rows),
        bulk_reset_region_policy=bulk_reset_region_policy,
    )
end

function fill_reset_retarded!(
    sigma_reset_r::Array{ComplexF64,3},
    matrix_indices::AbstractVector{<:Integer},
    gamma::Real,
)
    isempty(matrix_indices) && return nothing
    gamma_value = Float64(gamma)
    gamma_value == 0 && return nothing

    block = -(0.5im * gamma_value) .* Matrix{ComplexF64}(I, length(matrix_indices), length(matrix_indices))
    for ik in axes(sigma_reset_r, 3)
        sigma_reset_r[matrix_indices, matrix_indices, ik] .+= block
    end
    return nothing
end

function fill_reset_keldysh!(
    sigma_reset_k::Array{ComplexF64,3},
    matrix_indices::AbstractVector{<:Integer},
    rows::AbstractVector{<:Integer},
    kxs::AbstractVector{<:Real},
    r::Real,
    gamma::Real;
    t0::Real=1.0,
    Nky_projector::Integer=2048,
)
    isempty(matrix_indices) && return nothing
    gamma_value = Float64(gamma)
    gamma_value == 0 && return nothing

    sector_identity = Matrix{ComplexF64}(I, length(matrix_indices), length(matrix_indices))
    for (ik, kx) in pairs(kxs)
        restricted_projector = restricted_bulk_projector(
            kx,
            rows,
            r;
            t0=t0,
            Nky_projector=Nky_projector,
        )
        sigma_reset_k[matrix_indices, matrix_indices, ik] .+= (-1im * gamma_value) .* (sector_identity .- 2.0 .* restricted_projector)
    end
    return nothing
end

function build_bulk_reset_self_energies(
    kxs::AbstractVector{<:Real},
    row_ids::AbstractVector{<:Integer};
    r_ti::Real=1.0,
    r_triv::Real=3.0,
    Gamma_bulk_top::Real=0.0,
    Gamma_bulk_bottom::Real=0.0,
    reset_top_rows::AbstractVector{<:Integer}=Int[],
    reset_bottom_rows::AbstractVector{<:Integer}=Int[],
    t0::Real=1.0,
    Nky_projector::Integer=2048,
)
    dim = 2 * length(row_ids)
    Nkx = length(kxs)
    sigma_reset_r = zeros(ComplexF64, dim, dim, Nkx)
    sigma_reset_k = zeros(ComplexF64, dim, dim, Nkx)

    top_rows = sort(unique(Int.(reset_top_rows)))
    bottom_rows = sort(unique(Int.(reset_bottom_rows)))
    top_indices = matrix_indices_for_rows(top_rows, row_ids)
    bottom_indices = matrix_indices_for_rows(bottom_rows, row_ids)

    fill_reset_retarded!(sigma_reset_r, top_indices, Gamma_bulk_top)
    fill_reset_retarded!(sigma_reset_r, bottom_indices, Gamma_bulk_bottom)
    fill_reset_keldysh!(
        sigma_reset_k,
        top_indices,
        top_rows,
        kxs,
        r_ti,
        Gamma_bulk_top;
        t0=t0,
        Nky_projector=Nky_projector,
    )
    fill_reset_keldysh!(
        sigma_reset_k,
        bottom_indices,
        bottom_rows,
        kxs,
        r_triv,
        Gamma_bulk_bottom;
        t0=t0,
        Nky_projector=Nky_projector,
    )

    return (
        sigma_reset_r=sigma_reset_r,
        sigma_reset_k=sigma_reset_k,
        reset_top_rows=top_rows,
        reset_bottom_rows=bottom_rows,
    )
end

function default_reset_comparison_rows(
    reset_rows::AbstractVector{<:Integer},
    M::Integer;
    side::Symbol,
    max_rows::Integer=12,
)
    rows = sort(unique(Int.(reset_rows)))
    isempty(rows) && return Int[]

    defaults = default_deep_rows(M)
    candidates = intersect(rows, side == :top ? defaults.ti_rows : defaults.triv_rows)
    if isempty(candidates)
        candidates = rows
    end

    keep = min(length(candidates), max(1, Int(max_rows)))
    return side == :top ? candidates[(end - keep + 1):end] : candidates[1:keep]
end

function bulk_reset_side_diagnostics(
    Ckx::Array{ComplexF64,3},
    kxs::AbstractVector{<:Real},
    row_ids::AbstractVector{<:Integer},
    rows::AbstractVector{<:Integer},
    r::Real;
    M::Integer,
    side::Symbol,
    t0::Real=1.0,
    Nky_projector::Integer=2048,
)
    deep_rows = default_reset_comparison_rows(rows, M; side=side)
    if isempty(rows)
        return (
            rows=Int[],
            deep_rows=Int[],
            block_error_mean=0.0,
            block_error_max=0.0,
            onsite_error_mean=0.0,
            onsite_error_max=0.0,
            nn_error_mean=0.0,
            nn_error_max=0.0,
        )
    end

    deep_indices = matrix_indices_for_rows(deep_rows, row_ids)
    row_positions = [row_position_in_ids(y, row_ids) for y in deep_rows]
    block_errors = Float64[]
    onsite_errors = Float64[]
    nn_errors = Float64[]

    for (ik, kx) in pairs(kxs)
        projector_block = restricted_bulk_projector(
            kx,
            deep_rows,
            r;
            t0=t0,
            Nky_projector=Nky_projector,
        )
        push!(block_errors, opnorm(Ckx[deep_indices, deep_indices, ik] - projector_block, Inf))

        for (local_idx, row_pos) in pairs(row_positions)
            full_block = row_block_range_from_offset(row_pos)
            local_block = row_block_range_from_offset(local_idx)
            push!(onsite_errors, opnorm(Ckx[full_block, full_block, ik] - projector_block[local_block, local_block], Inf))
        end

        for local_idx in 1:(length(deep_rows) - 1)
            deep_rows[local_idx + 1] == deep_rows[local_idx] + 1 || continue
            left_full = row_block_range_from_offset(row_positions[local_idx])
            right_full = row_block_range_from_offset(row_positions[local_idx + 1])
            left_local = row_block_range_from_offset(local_idx)
            right_local = row_block_range_from_offset(local_idx + 1)
            push!(
                nn_errors,
                opnorm(
                    Ckx[left_full, right_full, ik] - projector_block[left_local, right_local],
                    Inf,
                ),
            )
        end
    end

    return (
        rows=deepcopy(sort(unique(Int.(rows)))),
        deep_rows=deep_rows,
        block_error_mean=mean(block_errors),
        block_error_max=maximum(block_errors),
        onsite_error_mean=mean(onsite_errors),
        onsite_error_max=maximum(onsite_errors),
        nn_error_mean=isempty(nn_errors) ? 0.0 : mean(nn_errors),
        nn_error_max=isempty(nn_errors) ? 0.0 : maximum(nn_errors),
    )
end

function bulk_reset_diagnostics(
    Ckx::Array{ComplexF64,3},
    kxs::AbstractVector{<:Real},
    row_ids::AbstractVector{<:Integer};
    M::Integer,
    bulk_reset_enabled::Bool=false,
    reset_top_rows::AbstractVector{<:Integer}=Int[],
    reset_bottom_rows::AbstractVector{<:Integer}=Int[],
    r_ti::Real=1.0,
    r_triv::Real=3.0,
    t0::Real=1.0,
    Nky_projector::Integer=2048,
)
    top = bulk_reset_side_diagnostics(
        Ckx,
        kxs,
        row_ids,
        reset_top_rows,
        r_ti;
        M=M,
        side=:top,
        t0=t0,
        Nky_projector=Nky_projector,
    )
    bottom = bulk_reset_side_diagnostics(
        Ckx,
        kxs,
        row_ids,
        reset_bottom_rows,
        r_triv;
        M=M,
        side=:bottom,
        t0=t0,
        Nky_projector=Nky_projector,
    )

    return (
        enabled=Bool(bulk_reset_enabled),
        top=top,
        bottom=bottom,
    )
end
