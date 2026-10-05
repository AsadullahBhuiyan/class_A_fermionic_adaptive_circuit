function surface_green_decimation(
    z_minus_h::AbstractMatrix{ComplexF64},
    forward::AbstractMatrix{ComplexF64},
    backward::AbstractMatrix{ComplexF64};
    tol::Real=1e-12,
    maxiter::Integer=200,
)
    e_surface = copy(z_minus_h)
    e_bulk = copy(z_minus_h)
    alpha = copy(forward)
    beta = copy(backward)
    converged = false

    for _ in 1:Int(maxiter)
        g_bulk = inv(e_bulk)
        correction_surface = alpha * g_bulk * beta
        correction_bulk = correction_surface + beta * g_bulk * alpha
        e_surface_new = e_surface - correction_surface
        e_bulk_new = e_bulk - correction_bulk
        alpha_new = alpha * g_bulk * alpha
        beta_new = beta * g_bulk * beta

        e_surface = e_surface_new
        e_bulk = e_bulk_new
        alpha = alpha_new
        beta = beta_new

        if max(norm(alpha, Inf), norm(beta, Inf)) < Float64(tol)
            converged = true
            break
        end
    end

    g_surface = inv(e_surface)
    residual = norm(inv(g_surface) - (z_minus_h - forward * g_surface * backward), Inf)
    return g_surface, residual, converged
end

function build_lead_self_energies(
    omegas::AbstractVector{<:Real},
    kxs::AbstractVector{<:Real};
    r_ti::Real=1.0,
    r_triv::Real=3.0,
    t0::Real=1.0,
    eta::Real=1e-6,
    tol::Real=1e-12,
    maxiter::Integer=200,
)
    Nomega = length(omegas)
    Nkx = length(kxs)
    sigma_top = Array{ComplexF64,4}(undef, 2, 2, Nomega, Nkx)
    sigma_bottom = Array{ComplexF64,4}(undef, 2, 2, Nomega, Nkx)
    V = hopping_block(t0=t0)
    Vdag = adjoint(V)
    residual_max = 0.0
    all_converged = true

    for (ik, kx) in pairs(kxs)
        h_top = onsite_block(kx, r_ti; t0=t0)
        h_bottom = onsite_block(kx, r_triv; t0=t0)
        for (iw, omega) in pairs(omegas)
            z = ComplexF64(Float64(omega), Float64(eta))
            z_top = z .* ORBITAL_I .- h_top
            z_bottom = z .* ORBITAL_I .- h_bottom

            g_top, res_top, conv_top = surface_green_decimation(
                z_top,
                V,
                Vdag;
                tol=tol,
                maxiter=maxiter,
            )
            g_bottom, res_bottom, conv_bottom = surface_green_decimation(
                z_bottom,
                Vdag,
                V;
                tol=tol,
                maxiter=maxiter,
            )

            sigma_top[:, :, iw, ik] .= V * g_top * Vdag
            sigma_bottom[:, :, iw, ik] .= Vdag * g_bottom * V

            residual_max = max(residual_max, res_top, res_bottom)
            all_converged &= conv_top & conv_bottom
        end
    end

    return (; sigma_top, sigma_bottom, residual_max, all_converged)
end

