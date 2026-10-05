using Plots
using Plots.PlotMeasures: mm
using Statistics

function default_plot_prefix(case_tag::AbstractString)
    return joinpath("note", "interface_chern_" * String(case_tag))
end

function plot_marker_profile(
    marker_data;
    output_pdf::AbstractString,
    title_label::AbstractString,
    show_avg_lines::Bool=true,
    x_limits::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    y_limits::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    color=:black,
)
    row_ids = Int.(marker_data["row_indices"])
    c_rows = Float64.(marker_data["c_rows"])
    c_ti_avg = Float64(marker_data["c_ti_avg"])
    c_triv_avg = Float64(marker_data["c_triv_avg"])
    marker_kind = String(marker_data["marker_kind"])

    plt = plot(
        row_ids,
        c_rows;
        linewidth=2.5,
        marker=:circle,
        markersize=4,
        markerstrokewidth=0,
        color=color,
        xlabel="row y",
        ylabel=marker_kind == "equilibrium" ? "c(y)" : "c_rho(y)",
        title=title_label,
        label=marker_kind == "equilibrium" ? "equilibrium marker" : "monitored diagnostic",
    )
    if show_avg_lines
        hline!(plt, [c_ti_avg]; color=:blue, linestyle=:dash, linewidth=2, label="deep TI avg")
        hline!(plt, [c_triv_avg]; color=:red, linestyle=:dash, linewidth=2, label="deep trivial avg")
    end
    if !isnothing(x_limits)
        xlims!(plt, x_limits)
    end
    if !isnothing(y_limits)
        ylims!(plt, y_limits)
    end
    savefig(plt, output_pdf)
    return output_pdf
end

function plot_marker_sweep_panels(
    summary_file::AbstractString;
    output_pdf::AbstractString,
    ncols::Integer=2,
    x_limits::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
)
    summary = load(summary_file)
    case_rows = summary["case_rows"]
    ncases = length(case_rows)
    ncases > 0 || error("no cases found in sweep summary")

    marker_payloads = NamedTuple[]
    ymin = Inf
    ymax = -Inf
    for case_row in case_rows
        marker_data = load(String(fieldget(case_row, :marker_file)))
        c_rows = Float64.(marker_data["c_rows"])
        ymin = min(ymin, minimum(c_rows))
        ymax = max(ymax, maximum(c_rows))
        push!(marker_payloads, (
            label=String(fieldget(case_row, :label)),
            marker_data=marker_data,
        ))
    end
    spread = ymax - ymin
    ypad = spread > 0 ? 0.05 * spread : 1.0e-3
    shared_y_limits = (ymin - ypad, ymax + ypad)

    colors = palette(:viridis, ncases)
    panels = Plots.Plot[]
    for (idx, payload) in enumerate(marker_payloads)
        row_ids = Int.(payload.marker_data["row_indices"])
        c_rows = Float64.(payload.marker_data["c_rows"])
        if isnothing(x_limits)
            xs = row_ids
            ys = c_rows
        else
            mask = (first(x_limits) .<= row_ids) .& (row_ids .<= last(x_limits))
            xs = row_ids[mask]
            ys = c_rows[mask]
        end
        push!(
            panels,
            plot(
                xs,
                ys;
                linewidth=2.5,
                marker=:circle,
                markersize=3,
                markerstrokewidth=0,
                color=colors[idx],
                xlabel="row y",
                ylabel="marker / diagnostic",
                title=payload.label,
                label=false,
                ylims=shared_y_limits,
            ),
        )
    end

    nrows = cld(ncases, Int(ncols))
    fig = plot(panels...; layout=(nrows, Int(ncols)), size=(500 * Int(ncols), 240 * nrows))
    savefig(fig, output_pdf)
    return output_pdf
end

function plot_spectral_maps(
    green_data;
    output_pdf::AbstractString,
    title_label::AbstractString,
)
    omega_grid = Float64.(green_data["omega_grid"])
    kx_grid = Float64.(green_data["kx_grid"])
    rho_int = Float64.(green_data["rho_int_wk"])
    rho_rowcuts = Float64.(green_data["rho_rowcuts_wk"])
    probe_rows = Int.(green_data["probe_rows"])

    panel_margins = (left_margin=5mm, right_margin=5mm, top_margin=4mm, bottom_margin=4mm)
    interface_upper = quantile(vec(rho_int), 0.995)
    row_upper = quantile(vec(rho_rowcuts), 0.995)
    neg_rows = sort(filter(<(0), probe_rows); rev=true)
    nonneg_rows = sort(filter(>=(0), probe_rows))
    row_index = Dict(y => idx for (idx, y) in pairs(probe_rows))
    nrows = max(length(neg_rows), length(nonneg_rows))

    blank_panel() = plot(;
        xlims=(0.0, 1.0),
        ylims=(0.0, 1.0),
        framestyle=:none,
        xticks=false,
        yticks=false,
        legend=false,
        panel_margins...,
    )

    interface_panel = heatmap(
        kx_grid,
        omega_grid,
        rho_int;
        xlabel="k_x",
        ylabel="omega",
        title=title_label * " interface sum",
        colorbar=true,
        colorbar_title="rho",
        clims=(0.0, max(interface_upper, maximum(rho_int) == 0 ? 1.0 : min(maximum(rho_int), interface_upper))),
        panel_margins...,
    )
    make_row_panel(y) = heatmap(
        kx_grid,
        omega_grid,
        rho_rowcuts[:, :, row_index[y]];
        xlabel="k_x",
        ylabel="omega",
        title="row y=$(y)",
        colorbar=true,
        colorbar_title="rho",
        clims=(0.0, max(row_upper, maximum(rho_rowcuts[:, :, row_index[y]]) == 0 ? 1.0 : min(maximum(rho_rowcuts[:, :, row_index[y]]), row_upper))),
        panel_margins...,
    )

    left_column = Plots.Plot[interface_panel]
    append!(left_column, [make_row_panel(y) for y in neg_rows])
    right_column = Plots.Plot[make_row_panel(y) for y in nonneg_rows]

    nrows = max(length(left_column), length(right_column))
    panels = Plots.Plot[]
    for row_idx in 1:nrows
        push!(panels, row_idx <= length(left_column) ? left_column[row_idx] : blank_panel())
        push!(panels, row_idx <= length(right_column) ? right_column[row_idx] : blank_panel())
    end

    plt = plot(panels...; layout=(nrows, 2), size=(1250, 270 * nrows))
    savefig(plt, output_pdf)
    return output_pdf
end

function plot_scan_marker_comparison(index_file::AbstractString; output_pdf::Union{Nothing,AbstractString}=nothing)
    index_data = load(index_file)
    entries = index_data["entries"]
    scan_dir = dirname(String(index_file))
    resolved_output = isnothing(output_pdf) ? joinpath(scan_dir, "marker_compare.pdf") : String(output_pdf)

    plt = plot(
        xlabel="row y",
        ylabel="marker profile",
        title="Interface marker comparison",
    )
    for entry in entries
        case_dir = String(fieldget(entry, :case_dir))
        marker_file = joinpath(case_dir, "marker.jld2")
        if !isfile(marker_file)
            continue
        end
        marker_data = load(marker_file)
        push!(
            plt,
            Int.(marker_data["row_indices"]),
            Float64.(marker_data["c_rows"]);
            linewidth=2,
            label=String(fieldget(entry, :case_tag)),
        )
    end
    savefig(plt, output_pdf)
    return output_pdf
end

function plot_interface_results(;
    green_file::AbstractString,
    marker_file::AbstractString,
    output_prefix::Union{Nothing,AbstractString}=nothing,
)
    green_data = load(green_file)
    marker_data = load(marker_file)
    case_tag = haskey(marker_data, "case_tag") ? String(marker_data["case_tag"]) : String(fieldget(green_data["metadata"], :case_tag))
    prefix = isnothing(output_prefix) ? default_plot_prefix(case_tag) : String(output_prefix)
    mkpath(dirname(prefix))

    marker_pdf = prefix * "_marker.pdf"
    spectral_pdf = prefix * "_spectral.pdf"
    plot_marker_profile(marker_data; output_pdf=marker_pdf, title_label=case_tag)
    plot_spectral_maps(green_data; output_pdf=spectral_pdf, title_label=case_tag)

    return (
        case_tag=case_tag,
        marker_pdf=marker_pdf,
        spectral_pdf=spectral_pdf,
    )
end
