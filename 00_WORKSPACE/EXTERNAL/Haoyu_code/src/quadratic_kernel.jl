using LinearAlgebra
using FFTW
using Plots

Epsqr(k, r) = (sin(k)^2 + (r - cos(k) + 1)^2)
Emsqr(k, r) = (sin(k)^2 + (r - cos(k) - 1)^2)

Σbulk(z, k, r) = (z^2 + 2r * cos(k) - r^2 - sqrt(z^2 - Epsqr(k, r)) * sqrt(z^2 - Emsqr(k, r))) / (2 * (z - sin(k)))

const σx = [0 1; 1 0] .+ zeros(ComplexF64, 2, 2)
const σy = [0 -im; im 0] .+ zeros(ComplexF64, 2, 2)
const σz = [1 0; 0 -1] .+ zeros(ComplexF64, 2, 2)
const I2 = Matrix{ComplexF64}(I, 2, 2)
const Pp = (I2 + σx) / 2
const Pm = (I2 - σx) / 2

@inline function keldysh_distribution_value(ω::Real)
    return ω > 0 ? 1.0 : (ω < 0 ? -1.0 : 0.0)
end

function Gfree(z, k, r, η=0.001)
    z = z + im * η
    Σ = Σbulk(z, k, r)
    Ginv = z .* I2 .- sin(k) .* σx .- (r - cos(k)) .* σz .- Σ .* Pm
    return inv(Ginv)
end

function Gfull(z, k, r, Γ, η=0.001)
    GRfree = Gfree(z, k, r, η)
    GAfree = conj.(GRfree)
    F0 = keldysh_distribution_value(real(z))
    GKfree = @.(F0 * (GRfree - GAfree))
    G0mat = [GRfree GKfree; zeros(ComplexF64, 2, 2) GAfree]
    return inv(inv(G0mat) .+ @.([I2 zeros(ComplexF64, 2, 2); zeros(ComplexF64, 2, 2) -I2] * (im * Γ / 2)))
end

function extractA(G)
    return -imag(G[1:2, 1:2])
end

function extractK(G)
    return G[1:2, 3:4] 
end

function extractR(G)
    return G[1:2, 1:2]
end

function main()
    r = 1.0
    Γ = 0.5
    zlist = range(-5, 5, step=0.005)
    k = -1.3
    Glist = [Gfull(z, k, r, Γ) for z in zlist]
    Adat = [tr(extractA(G)) for G in Glist]
    Kdat = [tr(extractK(G)) for G in Glist]

    plt = plot(zlist, Adat, label="k=$k", xlabel="z", ylabel="A(z)")
    pltK = plot(zlist, imag(Kdat), label="k=$k", xlabel="z", ylabel="K(z)", ylims=(-0.2, 0.2))
    display(plt)
    display(pltK)
end

# running_in_vscode_repl() = isinteractive() && isdefined(Main, :VSCodeServer)

# if abspath(PROGRAM_FILE) == (@__FILE__) || running_in_vscode_repl()
#     main()
# end
