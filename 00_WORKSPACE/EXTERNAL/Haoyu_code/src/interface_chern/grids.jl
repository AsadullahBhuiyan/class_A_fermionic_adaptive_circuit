function append_midpoints!(xs::Vector{Float64}, ws::Vector{Float64}, a::Float64, b::Float64, step::Float64)
    if !(b > a)
        return nothing
    end
    n = max(1, ceil(Int, (b - a) / step))
    h = (b - a) / n
    for i in 1:n
        push!(xs, a + (i - 0.5) * h)
        push!(ws, h)
    end
    return nothing
end

function build_symmetric_frequency_grid(;
    omega_dense::Real=4.0,
    omega_max::Real=12.0,
    domega_dense::Real=0.02,
    domega_coarse::Real=0.10,
)
    wdense = Float64(omega_dense)
    wmax = Float64(omega_max)
    ddense = Float64(domega_dense)
    dcoarse = Float64(domega_coarse)

    @assert 0.0 < wdense <= wmax
    @assert ddense > 0.0
    @assert dcoarse > 0.0

    pos = Float64[]
    pos_w = Float64[]
    append_midpoints!(pos, pos_w, 0.0, wdense, ddense)
    if wmax > wdense
        append_midpoints!(pos, pos_w, wdense, wmax, dcoarse)
    end

    omegas = Float64[]
    weights = Float64[]
    for idx in reverse(eachindex(pos))
        push!(omegas, -pos[idx])
        push!(weights, pos_w[idx])
    end
    append!(omegas, pos)
    append!(weights, pos_w)
    return omegas, weights
end

function uniform_periodic_kx_grid(Nkx::Integer)
    n = Int(Nkx)
    @assert n >= 3
    dk = 2π / n
    kxs = Vector{Float64}(undef, n)
    for idx in 1:n
        kxs[idx] = -π + (idx - 1) * dk
    end
    return kxs, dk
end

function centered_periodic_derivative(data::Array{ComplexF64,3}, dk::Real)
    _, _, Nkx = size(data)
    derivative = similar(data)
    scale = 1.0 / (2.0 * Float64(dk))
    for ik in 1:Nkx
        ip = ik == Nkx ? 1 : (ik + 1)
        im = ik == 1 ? Nkx : (ik - 1)
        derivative[:, :, ik] .= (data[:, :, ip] .- data[:, :, im]) .* scale
    end
    return derivative
end

function spectral_periodic_derivative(data::Array{ComplexF64,3})
    _, _, Nkx = size(data)
    derivative = similar(data)
    modes = if isodd(Nkx)
        ComplexF64.(vcat(0:((Nkx - 1) ÷ 2), -((Nkx - 1) ÷ 2):-1))
    else
        ComplexF64.(vcat(0:(Nkx ÷ 2 - 1), 0, -((Nkx ÷ 2) - 1):-1))
    end
    multiplier = 1im .* reshape(modes, 1, 1, :)

    # On a finite periodic k-grid, X is represented by the exact Fourier
    # spectral derivative on trigonometric interpolants, not by a local
    # nearest-neighbor stencil.
    derivative .= ifft(multiplier .* fft(data, [3]), [3])
    return derivative
end
