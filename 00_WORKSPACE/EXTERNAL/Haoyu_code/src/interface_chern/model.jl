function hopping_block(; t0::Real=1.0)
    return Float64(t0) .* (SIGMA_Y ./ (2im) .- SIGMA_Z ./ 2)
end

function onsite_block(kx::Real, r::Real; t0::Real=1.0)
    return Float64(t0) .* (sin(Float64(kx)) .* SIGMA_X .+ (Float64(r) - cos(Float64(kx))) .* SIGMA_Z)
end

function strip_mass_profile(M::Integer; r_ti::Real=1.0, r_triv::Real=3.0)
    profile = Float64[]
    for y in row_values(M)
        push!(profile, y < 0 ? Float64(r_triv) : Float64(r_ti))
    end
    return profile
end

function strip_hamiltonian(kx::Real; M::Integer, r_ti::Real=1.0, r_triv::Real=3.0, t0::Real=1.0)
    Ly = 2 * Int(M)
    dim = 2 * Ly
    H = zeros(ComplexF64, dim, dim)
    masses = strip_mass_profile(M; r_ti=r_ti, r_triv=r_triv)
    V = hopping_block(t0=t0)
    for offset in 1:Ly
        block = row_block_range_from_offset(offset)
        H[block, block] .= onsite_block(kx, masses[offset]; t0=t0)
    end
    for offset in 1:(Ly - 1)
        b1 = row_block_range_from_offset(offset)
        b2 = row_block_range_from_offset(offset + 1)
        H[b1, b2] .= V
        H[b2, b1] .= adjoint(V)
    end
    return H
end

function build_strip_hamiltonians(
    kxs::AbstractVector{<:Real};
    M::Integer,
    r_ti::Real=1.0,
    r_triv::Real=3.0,
    t0::Real=1.0,
)
    dim = 4 * Int(M)
    Hks = Array{ComplexF64,3}(undef, dim, dim, length(kxs))
    for (ik, kx) in pairs(kxs)
        Hks[:, :, ik] .= strip_hamiltonian(kx; M=M, r_ti=r_ti, r_triv=r_triv, t0=t0)
    end
    return Hks
end

