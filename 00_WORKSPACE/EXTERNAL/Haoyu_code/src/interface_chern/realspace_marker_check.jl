using JLD2
using LinearAlgebra
using Plots
using Statistics

include("InterfaceChern.jl")
using .InterfaceChern

function build_realspace_correlator(
    Ckx::Array{ComplexF64,3},
    kxs::AbstractVector{<:Real},
)
    dim_cell = size(Ckx, 1)
    Lx = size(Ckx, 3)
    length(kxs) == Lx || error("kx grid length must match size(Ckx, 3)")

    kernel_dx = Array{ComplexF64,3}(undef, dim_cell, dim_cell, Lx)
    for dx in 0:(Lx - 1)
        block = zeros(ComplexF64, dim_cell, dim_cell)
        for (ik, kx) in pairs(kxs)
            block .+= cis(Float64(kx) * dx) .* Ckx[:, :, ik]
        end
        kernel_dx[:, :, dx + 1] .= block ./ Lx
    end

    full_dim = Lx * dim_cell
    Cfull = zeros(ComplexF64, full_dim, full_dim)
    for x in 0:(Lx - 1)
        block_x = ((x * dim_cell) + 1):((x + 1) * dim_cell)
        for xp in 0:(Lx - 1)
            block_xp = ((xp * dim_cell) + 1):((xp + 1) * dim_cell)
            dx = mod(x - xp, Lx)
            Cfull[block_x, block_xp] .= kernel_dx[:, :, dx + 1]
        end
    end

    return 0.5 .* (Cfull .+ adjoint(Cfull))
end

function realspace_marker_map(
    Cfull::AbstractMatrix{ComplexF64},
    row_ids::AbstractVector{<:Integer},
    Lx::Integer,
)
    Ly = length(row_ids)
    dim_cell = 2 * Ly
    size(Cfull, 1) == Lx * dim_cell || error("Cfull has incompatible dimension")

    xcoords = collect(0:(Int(Lx) - 1))
    Xvec = zeros(Float64, size(Cfull, 1))
    Yvec = zeros(Float64, size(Cfull, 1))

    for (ix, x) in pairs(xcoords)
        cell_offset = (ix - 1) * dim_cell
        for (irow, y) in pairs(row_ids)
            block = (cell_offset + 2 * (irow - 1) + 1):(cell_offset + 2 * irow)
            Xvec[block] .= Float64(x)
            Yvec[block] .= Float64(y)
        end
    end

    PXP = (Cfull .* transpose(Xvec)) * Cfull
    PYP = (Cfull .* transpose(Yvec)) * Cfull
    comm = PXP * PYP - PYP * PXP
    local_basis = real.((-2π * 1im) .* diag(comm))

    cmap = zeros(Float64, Lx, Ly)
    for ix in 1:Int(Lx)
        cell_offset = (ix - 1) * dim_cell
        for irow in 1:Ly
            block = (cell_offset + 2 * (irow - 1) + 1):(cell_offset + 2 * irow)
            cmap[ix, irow] = sum(local_basis[block])
        end
    end

    return xcoords, Int.(row_ids), cmap
end

function plot_realspace_marker_map(
    xcoords::AbstractVector{<:Integer},
    row_ids::AbstractVector{<:Integer},
    cmap::AbstractMatrix{<:Real};
    output_pdf::AbstractString,
    title_label::AbstractString,
)
    plt = heatmap(
        xcoords,
        row_ids,
        permutedims(cmap, (2, 1));
        xlabel="x",
        ylabel="row y",
        colorbar_title="c(x,y)",
        title=title_label,
    )
    savefig(plt, output_pdf)
    return output_pdf
end

function plot_realspace_marker_slices(
    xcoords::AbstractVector{<:Integer},
    row_ids::AbstractVector{<:Integer},
    cmap::AbstractMatrix{<:Real};
    output_pdf::AbstractString,
    x_values::AbstractVector{<:Integer},
)
    plt = plot(
        xlabel="row y",
        ylabel="c(x,y)",
        title="Real-space marker slices",
    )
    x_to_idx = Dict(x => idx for (idx, x) in pairs(xcoords))
    for x in x_values
        haskey(x_to_idx, x) || continue
        plot!(
            plt,
            row_ids,
            cmap[x_to_idx[x], :];
            linewidth=2.5,
            marker=:circle,
            markersize=4,
            markerstrokewidth=0,
            label="x=$(x)",
        )
    end
    savefig(plt, output_pdf)
    return output_pdf
end

function main(;
    green_file::AbstractString,
    output_prefix::Union{Nothing,AbstractString}=nothing,
)
    green_data = load(String(green_file))
    Ckx = ComplexF64.(green_data["Ckx"])
    row_ids = Int.(green_data["row_indices"])
    kxs = Float64.(green_data["kx_grid"])
    Lx = length(kxs)

    all(isapprox(exp(-1im * kx * Lx), 1.0 + 0.0im; atol=1e-8) for kx in kxs) ||
        error("kx grid is not compatible with a periodic real-space ring; use even Nkx or a periodic grid")

    prefix = isnothing(output_prefix) ? joinpath(dirname(String(green_file)), "realspace_marker") : String(output_prefix)
    mkpath(dirname(prefix))

    Cfull = build_realspace_correlator(Ckx, kxs)
    xcoords, ycoords, cmap = realspace_marker_map(Cfull, row_ids, Lx)

    center_x = Int(Lx) ÷ 2
    map_pdf = prefix * "_map.pdf"
    slices_pdf = prefix * "_slices.pdf"
    plot_realspace_marker_map(xcoords, ycoords, cmap; output_pdf=map_pdf, title_label="Real-space marker map")
    plot_realspace_marker_slices(
        xcoords,
        ycoords,
        cmap;
        output_pdf=slices_pdf,
        x_values=[0, center_x],
    )

    data_file = prefix * ".jld2"
    jldsave(
        data_file;
        green_file=String(green_file),
        xcoords=xcoords,
        row_ids=ycoords,
        marker_map=cmap,
        center_x=center_x,
        map_pdf=map_pdf,
        slices_pdf=slices_pdf,
    )

    return (
        data_file=data_file,
        map_pdf=map_pdf,
        slices_pdf=slices_pdf,
        center_x=center_x,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    error("run via include(...) and main(...)")
end
