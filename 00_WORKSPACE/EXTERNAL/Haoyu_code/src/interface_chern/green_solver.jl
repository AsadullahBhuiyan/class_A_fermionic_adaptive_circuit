function sigma_monitor_retarded_matrix(
    M::Integer;
    monitored::Bool=false,
    GammaM::Real=0.0,
    y_mon_lo::Integer=0,
    y_mon_hi::Integer=0,
)
    dim = 4 * Int(M)
    sigma = zeros(ComplexF64, dim, dim)
    if !monitored || GammaM == 0
        return sigma
    end
    block_value = -(0.5im * Float64(GammaM)) .* ORBITAL_I
    for y in monitored_rows(M, y_mon_lo, y_mon_hi)
        block = row_block_range(y, M)
        sigma[block, block] .= block_value
    end
    return sigma
end

function sigma_monitor_keldysh_matrix(
    M::Integer,
    sigma_m_k_entries::AbstractMatrix{<:Real};
    y_mon_lo::Integer=0,
    y_mon_hi::Integer=0,
)
    dim = 4 * Int(M)
    sigma = zeros(ComplexF64, dim, dim)
    monitored_set = Set(monitored_rows(M, y_mon_lo, y_mon_hi))
    for y in row_values(M)
        if !(y in monitored_set)
            continue
        end
        offset = row_to_offset(y, M)
        block = row_block_range_from_offset(offset)
        # The stored entries are the Hermitian density kernels Gamma * (1 - 2n).
        # In the Dyson convention G^K = G^R * Sigma^K * G^A, the actual
        # Keldysh self-energy is anti-Hermitian: Sigma^K = -i * K.
        sigma[block[1], block[1]] = -1im * Float64(sigma_m_k_entries[offset, 1])
        sigma[block[2], block[2]] = -1im * Float64(sigma_m_k_entries[offset, 2])
    end
    return sigma
end

function row_orbital_densities(Ckx::Array{ComplexF64,3}, M::Integer)
    Ly = 2 * Int(M)
    Nkx = size(Ckx, 3)
    densities = zeros(Float64, Ly, 2)
    for offset in 1:Ly
        block = row_block_range_from_offset(offset)
        for ik in 1:Nkx
            densities[offset, 1] += real(Ckx[block[1], block[1], ik]) / Nkx
            densities[offset, 2] += real(Ckx[block[2], block[2], ik]) / Nkx
        end
    end
    return densities
end

function sigma_m_k_entries_from_densities(
    densities::AbstractMatrix{<:Real},
    M::Integer;
    GammaM::Real=0.0,
    y_mon_lo::Integer=0,
    y_mon_hi::Integer=0,
)
    entries = zeros(Float64, size(densities))
    for y in monitored_rows(M, y_mon_lo, y_mon_hi)
        offset = row_to_offset(y, M)
        entries[offset, 1] = Float64(GammaM) * (1.0 - 2.0 * Float64(densities[offset, 1]))
        entries[offset, 2] = Float64(GammaM) * (1.0 - 2.0 * Float64(densities[offset, 2]))
    end
    return entries
end

function setup_case_dir(
    output_root::AbstractString;
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
    bulk_reset_region_policy::Symbol=:auto_outside_monitor,
    y_reset_top_lo::Union{Nothing,Integer}=nothing,
    y_reset_bottom_hi::Union{Nothing,Integer}=nothing,
)
    tag = case_tag(
        r_ti=r_ti,
        r_triv=r_triv,
        M=M,
        Nkx=Nkx,
        monitored=monitored,
        GammaM=GammaM,
        y_mon_lo=y_mon_lo,
        y_mon_hi=y_mon_hi,
        bulk_reset_enabled=bulk_reset_enabled,
        Gamma_bulk_top=Gamma_bulk_top,
        Gamma_bulk_bottom=Gamma_bulk_bottom,
        bulk_reset_region_policy=String(bulk_reset_region_policy),
        y_reset_top_lo=y_reset_top_lo,
        y_reset_bottom_hi=y_reset_bottom_hi,
    )
    case_dir = joinpath(output_root, tag)
    mkpath(case_dir)
    return tag, case_dir
end

function resolve_eta(
    eta::Real,
    domega_dense::Real,
    domega_coarse::Real,
)
    dense_step = abs(Float64(domega_dense))
    dense_step > 0 || throw(ArgumentError("dense frequency step must be positive"))
    eta_floor = 2.0 * dense_step
    eta_value = Float64(eta)

    if !isfinite(eta_value) || eta_value <= 0
        return eta_floor, eta_floor, false
    end

    if eta_value < eta_floor
        @warn "Requested eta is smaller than twice the dense frequency step; clamping eta upward." eta_requested=eta_value eta_minimum=eta_floor
        return eta_floor, eta_floor, true
    end

    return eta_value, eta_floor, false
end

function fill_lead_source!(
    source::AbstractMatrix{ComplexF64},
    sigma_top_r::AbstractMatrix{ComplexF64},
    sigma_bottom_r::AbstractMatrix{ComplexF64},
    omega::Real,
    M::Integer,
)
    fill!(source, 0.0 + 0.0im)
    top_block = row_block_range(M - 1, M)
    bottom_block = row_block_range(-M, M)
    sign_factor = sign(Float64(omega))
    sigma_top_k = sign_factor .* (sigma_top_r .- adjoint(sigma_top_r))
    sigma_bottom_k = sign_factor .* (sigma_bottom_r .- adjoint(sigma_bottom_r))
    source[top_block, top_block] .= sigma_top_k
    source[bottom_block, bottom_block] .= sigma_bottom_k
    return nothing
end

function point_retarded_green(
    base_matrix::AbstractMatrix{ComplexF64},
    sigma_top_r::AbstractMatrix{ComplexF64},
    sigma_bottom_r::AbstractMatrix{ComplexF64},
    omega::Real,
    M::Integer;
    eta::Real=1e-6,
)
    A = copy(base_matrix)
    z = ComplexF64(Float64(omega), Float64(eta))
    for idx in 1:size(A, 1)
        A[idx, idx] += z
    end
    top_block = row_block_range(M - 1, M)
    bottom_block = row_block_range(-M, M)
    A[top_block, top_block] .-= sigma_top_r
    A[bottom_block, bottom_block] .-= sigma_bottom_r
    return inv(A)
end

function spectral_density_for_rows(A::AbstractMatrix{ComplexF64}, rows::AbstractVector{<:Integer}, M::Integer)
    density = 0.0
    for y in rows
        block = row_block_range(y, M)
        density += real(tr(A[block, block]))
    end
    return density / (2π)
end

function run_green_pass(
    Hks::Array{ComplexF64,3},
    sigma_top::Array{ComplexF64,4},
    sigma_bottom::Array{ComplexF64,4},
    omegas::AbstractVector{<:Real},
    omega_weights::AbstractVector{<:Real},
    kxs::AbstractVector{<:Real};
    M::Integer,
    eta::Real=1e-6,
    sigma_m_r::AbstractMatrix{ComplexF64}=zeros(ComplexF64, 0, 0),
    sigma_m_k_entries::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    sigma_extra_r::Union{Nothing,Array{ComplexF64,3}}=nothing,
    sigma_extra_k::Union{Nothing,Array{ComplexF64,3}}=nothing,
    equilibrium::Bool=true,
    interface_rows::AbstractVector{<:Integer}=Int[],
    probe_rows::AbstractVector{<:Integer}=Int[],
    compute_spectra::Bool=true,
    compute_fdt::Bool=false,
)
    dim = size(Hks, 1)
    Nomega = length(omegas)
    Nkx = length(kxs)
    Ckx = Array{ComplexF64,3}(undef, dim, dim, Nkx)
    sigma_m_k_full = sigma_m_k_entries === nothing ? zeros(ComplexF64, dim, dim) :
        sigma_monitor_keldysh_matrix(M, sigma_m_k_entries; y_mon_lo=-M, y_mon_hi=M - 1)
    sigma_extra_r_full = isnothing(sigma_extra_r) ? zeros(ComplexF64, dim, dim, Nkx) : sigma_extra_r
    sigma_extra_k_full = isnothing(sigma_extra_k) ? zeros(ComplexF64, dim, dim, Nkx) : sigma_extra_k

    rho_int_wk = compute_spectra ? zeros(Float64, Nomega, Nkx) : zeros(Float64, 0, 0)
    rho_rowcuts_wk = compute_spectra ? zeros(Float64, Nomega, Nkx, length(probe_rows)) : zeros(Float64, 0, 0, 0)
    source = zeros(ComplexF64, dim, dim)
    fdt_max_error = 0.0

    for ik in 1:Nkx
        base_matrix = -Hks[:, :, ik] .- sigma_m_r .- sigma_extra_r_full[:, :, ik]
        integral = zeros(ComplexF64, dim, dim)
        for iw in 1:Nomega
            sigma_top_r = sigma_top[:, :, iw, ik]
            sigma_bottom_r = sigma_bottom[:, :, iw, ik]
            G_R = point_retarded_green(base_matrix, sigma_top_r, sigma_bottom_r, omegas[iw], M; eta=eta)
            G_A = adjoint(G_R)

            fill_lead_source!(source, sigma_top_r, sigma_bottom_r, omegas[iw], M)
            source .+= sigma_extra_k_full[:, :, ik]
            fdt_target = sign(Float64(omegas[iw])) .* (G_R .- G_A)
            G_K = if equilibrium
                # eta is only the numerical retarded prescription; enforce the
                # equilibrium Keldysh sector from the full spectral function.
                fdt_target
            else
                G_R * (source + sigma_m_k_full) * G_A
            end

            if compute_fdt
                fdt_max_error = max(fdt_max_error, norm(G_K - fdt_target, Inf))
            end

            integral .+= Float64(omega_weights[iw]) .* G_K

            if compute_spectra
                spectral = 1im .* (G_R .- G_A)
                rho_int_wk[iw, ik] = spectral_density_for_rows(spectral, interface_rows, M)
                for (iprobe, y) in pairs(probe_rows)
                    rho_rowcuts_wk[iw, ik, iprobe] = spectral_density_for_rows(spectral, [y], M)
                end
            end
        end

        Ckx[:, :, ik] .= 0.5 .* Matrix{ComplexF64}(I, dim, dim) .- 0.5im .* integral ./ (2π)
        Ckx[:, :, ik] .= 0.5 .* (Ckx[:, :, ik] .+ adjoint(Ckx[:, :, ik]))
    end

    return (; Ckx, rho_int_wk, rho_rowcuts_wk, fdt_max_error)
end

function monitored_self_consistent_pass(
    Hks::Array{ComplexF64,3},
    sigma_top::Array{ComplexF64,4},
    sigma_bottom::Array{ComplexF64,4},
    omegas::AbstractVector{<:Real},
    omega_weights::AbstractVector{<:Real},
    kxs::AbstractVector{<:Real};
    M::Integer,
    GammaM::Real=0.5,
    y_mon_lo::Integer=0,
    y_mon_hi::Integer=0,
    eta::Real=1e-6,
    sigma_extra_r::Union{Nothing,Array{ComplexF64,3}}=nothing,
    sigma_extra_k::Union{Nothing,Array{ComplexF64,3}}=nothing,
    monitor_mixing::Real=0.5,
    monitor_tol::Real=1e-8,
    monitor_maxiter::Integer=200,
)
    sigma_m_r = sigma_monitor_retarded_matrix(
        M;
        monitored=true,
        GammaM=GammaM,
        y_mon_lo=y_mon_lo,
        y_mon_hi=y_mon_hi,
    )
    sigma_entries = zeros(Float64, 2 * Int(M), 2)
    converged = false
    final_pass = nothing
    densities = zeros(Float64, 2 * Int(M), 2)
    final_delta = Inf
    iterations = 0

    for iter in 1:Int(monitor_maxiter)
        pass = run_green_pass(
            Hks,
            sigma_top,
            sigma_bottom,
            omegas,
            omega_weights,
            kxs;
            M=M,
            eta=eta,
            sigma_m_r=sigma_m_r,
            sigma_m_k_entries=sigma_entries,
            sigma_extra_r=sigma_extra_r,
            sigma_extra_k=sigma_extra_k,
            equilibrium=false,
            interface_rows=Int[],
            probe_rows=Int[],
            compute_spectra=false,
            compute_fdt=false,
        )
        densities = row_orbital_densities(pass.Ckx, M)
        sigma_target = sigma_m_k_entries_from_densities(
            densities,
            M;
            GammaM=GammaM,
            y_mon_lo=y_mon_lo,
            y_mon_hi=y_mon_hi,
        )
        delta = maximum(abs.(sigma_target .- sigma_entries))
        sigma_entries .= (1.0 - Float64(monitor_mixing)) .* sigma_entries .+ Float64(monitor_mixing) .* sigma_target
        final_pass = pass
        final_delta = delta
        iterations = iter
        if delta < Float64(monitor_tol)
            converged = true
            break
        end
    end

    final_pass = run_green_pass(
        Hks,
        sigma_top,
        sigma_bottom,
        omegas,
        omega_weights,
        kxs;
        M=M,
        eta=eta,
        sigma_m_r=sigma_m_r,
        sigma_m_k_entries=sigma_entries,
        sigma_extra_r=sigma_extra_r,
        sigma_extra_k=sigma_extra_k,
        equilibrium=false,
        interface_rows=Int[],
        probe_rows=Int[],
        compute_spectra=false,
        compute_fdt=false,
    )
    densities = row_orbital_densities(final_pass.Ckx, M)

    return (
        sigma_m_r=sigma_m_r,
        sigma_m_k_entries=sigma_entries,
        row_densities=densities,
        converged=converged,
        iterations=iterations,
        final_density_change=final_delta,
        Ckx=final_pass.Ckx,
        monitor_mode="self_consistent",
    )
end

function monitored_half_filling_pass(
    Hks::Array{ComplexF64,3},
    sigma_top::Array{ComplexF64,4},
    sigma_bottom::Array{ComplexF64,4},
    omegas::AbstractVector{<:Real},
    omega_weights::AbstractVector{<:Real},
    kxs::AbstractVector{<:Real};
    M::Integer,
    GammaM::Real=0.5,
    y_mon_lo::Integer=0,
    y_mon_hi::Integer=0,
    eta::Real=1e-6,
    sigma_extra_r::Union{Nothing,Array{ComplexF64,3}}=nothing,
    sigma_extra_k::Union{Nothing,Array{ComplexF64,3}}=nothing,
)
    sigma_m_r = sigma_monitor_retarded_matrix(
        M;
        monitored=true,
        GammaM=GammaM,
        y_mon_lo=y_mon_lo,
        y_mon_hi=y_mon_hi,
    )
    sigma_entries = zeros(Float64, 2 * Int(M), 2)
    pass = run_green_pass(
        Hks,
        sigma_top,
        sigma_bottom,
        omegas,
        omega_weights,
        kxs;
        M=M,
        eta=eta,
        sigma_m_r=sigma_m_r,
        sigma_m_k_entries=sigma_entries,
        sigma_extra_r=sigma_extra_r,
        sigma_extra_k=sigma_extra_k,
        equilibrium=false,
        interface_rows=Int[],
        probe_rows=Int[],
        compute_spectra=false,
        compute_fdt=false,
    )
    densities = row_orbital_densities(pass.Ckx, M)
    return (
        sigma_m_r=sigma_m_r,
        sigma_m_k_entries=sigma_entries,
        row_densities=densities,
        converged=true,
        iterations=1,
        final_density_change=0.0,
        Ckx=pass.Ckx,
        monitor_mode="half_filling_assumed",
    )
end

function monitored_density_deviation(
    densities::AbstractMatrix{<:Real},
    M::Integer;
    y_mon_lo::Integer=0,
    y_mon_hi::Integer=0,
)
    offsets = [row_to_offset(y, M) for y in monitored_rows(M, y_mon_lo, y_mon_hi)]
    if isempty(offsets)
        return 0.0
    end
    return maximum(abs.(densities[offsets, :] .- 0.5))
end

function solve_interface_green(;
    output_root::AbstractString="data/interface_chern",
    r_ti::Real=1.0,
    r_triv::Real=3.0,
    M::Integer=8,
    Nkx::Integer=121,
    t0::Real=1.0,
    omega_dense::Real=4.0,
    omega_max::Real=12.0,
    domega_dense::Real=0.02,
    domega_coarse::Real=0.10,
    eta::Real=NaN,
    monitored::Bool=false,
    GammaM::Real=0.5,
    assume_monitor_half_filling::Bool=true,
    y_mon_lo::Integer=0,
    y_mon_hi::Integer=0,
    bulk_reset_enabled::Bool=false,
    Gamma_bulk_top::Real=0.0,
    Gamma_bulk_bottom::Real=0.0,
    bulk_reset_region_policy::Symbol=:auto_outside_monitor,
    y_reset_top_lo::Union{Nothing,Integer}=nothing,
    y_reset_bottom_hi::Union{Nothing,Integer}=nothing,
    Nky_projector::Integer=2048,
    monitor_mixing::Real=0.5,
    monitor_tol::Real=1e-8,
    monitor_maxiter::Integer=200,
    interface_window_halfwidth::Integer=1,
    probe_rows::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    lead_tol::Real=1e-12,
    lead_maxiter::Integer=200,
)
    output_root = String(output_root)
    mkpath(output_root)
    eta_eff, eta_floor, eta_was_clamped = resolve_eta(eta, domega_dense, domega_coarse)
    row_ids = row_values(M)
    reset_regions = resolve_bulk_reset_regions(
        M;
        bulk_reset_enabled=bulk_reset_enabled,
        bulk_reset_region_policy=bulk_reset_region_policy,
        monitored=monitored,
        y_mon_lo=y_mon_lo,
        y_mon_hi=y_mon_hi,
        y_reset_top_lo=y_reset_top_lo,
        y_reset_bottom_hi=y_reset_bottom_hi,
    )

    tag, case_dir = setup_case_dir(
        output_root;
        r_ti=r_ti,
        r_triv=r_triv,
        M=M,
        Nkx=Nkx,
        monitored=monitored,
        GammaM=GammaM,
        y_mon_lo=y_mon_lo,
        y_mon_hi=y_mon_hi,
        bulk_reset_enabled=bulk_reset_enabled,
        Gamma_bulk_top=Gamma_bulk_top,
        Gamma_bulk_bottom=Gamma_bulk_bottom,
        bulk_reset_region_policy=bulk_reset_region_policy,
        y_reset_top_lo=reset_regions.y_reset_top_lo,
        y_reset_bottom_hi=reset_regions.y_reset_bottom_hi,
    )

    omegas, omega_weights = build_symmetric_frequency_grid(
        omega_dense=omega_dense,
        omega_max=omega_max,
        domega_dense=domega_dense,
        domega_coarse=domega_coarse,
    )
    kxs, dk = uniform_periodic_kx_grid(Nkx)
    interface_rows = default_interface_rows(M, interface_window_halfwidth)
    selected_probe_rows = isnothing(probe_rows) ? default_probe_rows(M) : select_rows_in_bounds(probe_rows, M)

    Hks = build_strip_hamiltonians(kxs; M=M, r_ti=r_ti, r_triv=r_triv, t0=t0)
    leads = build_lead_self_energies(
        omegas,
        kxs;
        r_ti=r_ti,
        r_triv=r_triv,
        t0=t0,
        eta=eta_eff,
        tol=lead_tol,
        maxiter=lead_maxiter,
    )
    bulk_reset_active = Bool(bulk_reset_enabled) && (Float64(Gamma_bulk_top) != 0.0 || Float64(Gamma_bulk_bottom) != 0.0)
    bulk_reset = build_bulk_reset_self_energies(
        kxs,
        row_ids;
        r_ti=r_ti,
        r_triv=r_triv,
        Gamma_bulk_top=Gamma_bulk_top,
        Gamma_bulk_bottom=Gamma_bulk_bottom,
        reset_top_rows=reset_regions.reset_top_rows,
        reset_bottom_rows=reset_regions.reset_bottom_rows,
        t0=t0,
        Nky_projector=Nky_projector,
    )

    sigma_m_r = sigma_monitor_retarded_matrix(
        M;
        monitored=monitored,
        GammaM=GammaM,
        y_mon_lo=y_mon_lo,
        y_mon_hi=y_mon_hi,
    )
    sigma_m_k_entries = zeros(Float64, 2 * Int(M), 2)
    convergence = (
        converged=true,
        iterations=0,
        final_density_change=0.0,
    )
    row_dens_monitor = zeros(Float64, 2 * Int(M), 2)
    monitor_mode = monitored ? "half_filling_assumed" : "equilibrium_unmonitored"
    monitored_density_max_deviation = 0.0

    if monitored
        monitor_result = assume_monitor_half_filling ?
            monitored_half_filling_pass(
                Hks,
                leads.sigma_top,
                leads.sigma_bottom,
                omegas,
                omega_weights,
                kxs;
                M=M,
                GammaM=GammaM,
                y_mon_lo=y_mon_lo,
                y_mon_hi=y_mon_hi,
                eta=eta_eff,
                sigma_extra_r=bulk_reset.sigma_reset_r,
                sigma_extra_k=bulk_reset.sigma_reset_k,
            ) :
            monitored_self_consistent_pass(
                Hks,
                leads.sigma_top,
                leads.sigma_bottom,
                omegas,
                omega_weights,
                kxs;
                M=M,
                GammaM=GammaM,
                y_mon_lo=y_mon_lo,
                y_mon_hi=y_mon_hi,
                eta=eta_eff,
                sigma_extra_r=bulk_reset.sigma_reset_r,
                sigma_extra_k=bulk_reset.sigma_reset_k,
                monitor_mixing=monitor_mixing,
                monitor_tol=monitor_tol,
                monitor_maxiter=monitor_maxiter,
            )
        sigma_m_r = monitor_result.sigma_m_r
        sigma_m_k_entries = monitor_result.sigma_m_k_entries
        row_dens_monitor = monitor_result.row_densities
        convergence = (
            converged=monitor_result.converged,
            iterations=monitor_result.iterations,
            final_density_change=monitor_result.final_density_change,
        )
        monitor_mode = monitor_result.monitor_mode
        monitored_density_max_deviation = monitored_density_deviation(
            row_dens_monitor,
            M;
            y_mon_lo=y_mon_lo,
            y_mon_hi=y_mon_hi,
        )
    end

    pass = run_green_pass(
        Hks,
        leads.sigma_top,
        leads.sigma_bottom,
        omegas,
        omega_weights,
        kxs;
        M=M,
        eta=eta_eff,
        sigma_m_r=sigma_m_r,
        sigma_m_k_entries=monitored ? sigma_m_k_entries : nothing,
        sigma_extra_r=bulk_reset.sigma_reset_r,
        sigma_extra_k=bulk_reset.sigma_reset_k,
        equilibrium=!(monitored || bulk_reset_active),
        interface_rows=interface_rows,
        probe_rows=selected_probe_rows,
        compute_spectra=true,
        compute_fdt=!(monitored || bulk_reset_active),
    )
    row_dens = row_orbital_densities(pass.Ckx, M)
    reset_diagnostics = bulk_reset_diagnostics(
        pass.Ckx,
        kxs,
        row_ids;
        M=M,
        bulk_reset_enabled=bulk_reset_enabled,
        reset_top_rows=bulk_reset.reset_top_rows,
        reset_bottom_rows=bulk_reset.reset_bottom_rows,
        r_ti=r_ti,
        r_triv=r_triv,
        t0=t0,
        Nky_projector=Nky_projector,
    )

    metadata = (
        case_tag=tag,
        case_dir=case_dir,
        green_file=joinpath(case_dir, "green.jld2"),
        metadata_file=joinpath(case_dir, "metadata.jld2"),
        r_ti=Float64(r_ti),
        r_triv=Float64(r_triv),
        M=Int(M),
        Ly=2 * Int(M),
        dim=4 * Int(M),
        Nkx=Int(Nkx),
        dk=Float64(dk),
        t0=Float64(t0),
        omega_dense=Float64(omega_dense),
        omega_max=Float64(omega_max),
        domega_dense=Float64(domega_dense),
        domega_coarse=Float64(domega_coarse),
        eta=Float64(eta_eff),
        eta_minimum=Float64(eta_floor),
        eta_was_clamped=Bool(eta_was_clamped),
        monitored=Bool(monitored),
        GammaM=Float64(monitored ? GammaM : 0.0),
        assume_monitor_half_filling=Bool(monitored ? assume_monitor_half_filling : false),
        monitor_mode=String(monitor_mode),
        monitored_rows=monitored_rows(M, y_mon_lo, y_mon_hi),
        y_mon_lo=Int(y_mon_lo),
        y_mon_hi=Int(y_mon_hi),
        bulk_reset_enabled=Bool(bulk_reset_enabled),
        Gamma_bulk_top=Float64(Gamma_bulk_top),
        Gamma_bulk_bottom=Float64(Gamma_bulk_bottom),
        bulk_reset_region_policy=String(reset_regions.bulk_reset_region_policy),
        y_reset_top_lo=reset_regions.y_reset_top_lo,
        y_reset_bottom_hi=reset_regions.y_reset_bottom_hi,
        reset_top_rows=bulk_reset.reset_top_rows,
        reset_bottom_rows=bulk_reset.reset_bottom_rows,
        Nky_projector=Int(Nky_projector),
        monitor_mixing=Float64(monitor_mixing),
        monitor_tol=Float64(monitor_tol),
        monitor_maxiter=Int(monitor_maxiter),
        converged=Bool(convergence.converged),
        iterations=Int(convergence.iterations),
        final_density_change=Float64(convergence.final_density_change),
        monitored_density_max_deviation=Float64(monitored_density_max_deviation),
        lead_residual_max=Float64(leads.residual_max),
        lead_all_converged=Bool(leads.all_converged),
        fdt_max_error=Float64(pass.fdt_max_error),
        bulk_reset_diagnostics=reset_diagnostics,
        interface_rows=interface_rows,
        probe_rows=selected_probe_rows,
        row_indices=row_ids,
        omega_grid=Float64.(omegas),
        omega_weights=Float64.(omega_weights),
        kx_grid=Float64.(kxs),
    )

    green_file = metadata.green_file
    metadata_file = metadata.metadata_file
    jldsave(
        green_file;
        Ckx=pass.Ckx,
        rho_int_wk=pass.rho_int_wk,
        rho_rowcuts_wk=pass.rho_rowcuts_wk,
        row_densities=row_dens,
        sigma_m_k_entries=sigma_m_k_entries,
        monitored_rows=metadata.monitored_rows,
        reset_top_rows=metadata.reset_top_rows,
        reset_bottom_rows=metadata.reset_bottom_rows,
        bulk_reset_diagnostics=reset_diagnostics,
        row_indices=row_ids,
        probe_rows=selected_probe_rows,
        interface_rows=interface_rows,
        omega_grid=Float64.(omegas),
        kx_grid=Float64.(kxs),
        metadata=metadata,
    )
    jldsave(metadata_file; metadata...)

    return (
        case_tag=tag,
        case_dir=case_dir,
        green_file=green_file,
        metadata_file=metadata_file,
        monitored=Bool(monitored),
    )
end
