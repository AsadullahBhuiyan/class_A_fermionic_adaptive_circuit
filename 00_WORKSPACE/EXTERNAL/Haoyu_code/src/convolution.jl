using FFTW, LinearAlgebra

function create_padding(data::AbstractArray, padded_half_length::Int; zeropoint::Int)

    padded_size = size(data)
    padded_size[1] = 2 * padded_half_length

    data_padded = zeros(eltype(data), padded_size)

    lenR = size(data, 1) - zeropoint + 1

    data_padded[1:lenR, :] .= data[zeropoint:end, :]

    if zeropoint > 1
        data_padded[(end-zeropoint+2):end, :] .= data[1:(zeropoint-1), :]
    end
    return data_padded
end

function write_to_padding!(dest::AbstractArray, data::AbstractArray; zeropoint::Int)
    fill!(dest, zero(eltype(dest)))
    lenR = size(data, 1) - zeropoint + 1

    dest[1:lenR, :] .= data[zeropoint:end, :]

    if zeropoint > 1
        dest[(end-zeropoint+2):end, :] .= data[1:(zeropoint-1), :]
    end
end

abstract type AbstractTail end

# Tail interface:
# - each concrete tail must implement `fω` and `ft`
# - `fω(t, ω)` evaluates the tail in frequency space
# - `ft(t, x)` evaluates the Fourier-transformed tail in time space

function tailparams end
function fω end
function ft end

struct RegulatedRetardedTail <: AbstractTail
    ϵ::Float64
    n::Int
    a0::ComplexF64
end

function RegulatedRetardedTail(ϵ::Real, n::Integer, a0::Number=1.0 + 0im)
    return RegulatedRetardedTail(Float64(ϵ), Int(n), ComplexF64(a0))
end

Heaviside(x) = (1 + sign(x)) / 2

Base.:*(t1::RegulatedRetardedTail, t2::RegulatedRetardedTail) = RegulatedRetardedTail(
    t1.ϵ + t2.ϵ,
    t1.n + t2.n,
    t1.a0 * t2.a0,
)

ft(t::RegulatedRetardedTail, x) = t.a0 * Heaviside(x) * exp(-t.ϵ * x) * x^(t.n)
fω(t::RegulatedRetardedTail, ω) = (im)^(t.n + 1) * factorial(t.n) * t.a0 / (ω + im * t.ϵ)^(t.n + 1)

function padded_fft_indices(N::Int)
    return vcat(collect(0:fld(N - 1, 2)), collect(-fld(N, 2):-1))
end

@inline function padded_fft_index(i::Int, N::Int)
    pivot = fld(N - 1, 2) + 1
    return i <= pivot ? i - 1 : i - 1 - N
end

default_binary!(x, y) = x * y
default_binary!(dest, x, y) = mul!(dest, x, y)

@inline payload_shape(data::AbstractArray{<:Any,N}) where {N} = _payload_shape(data, Val(N))
@inline _payload_shape(data, ::Val{2}) = ()
@inline _payload_shape(data, ::Val{N}) where {N} = ntuple(i -> size(data, i + 2), Val(N - 2))

struct FlatBinary{F}
    f::F
end

@inline reshape_payload(slice, ::Tuple{}) = slice[1]
@inline reshape_payload(slice, shape::NTuple{N,Int}) where {N} = reshape(slice, shape)

@inline function scalar_binary_payload(
    binary!::F,
    x,
    y,
    data1_payload_shape::S1,
    data2_payload_shape::S2,
) where {F, S1<:Tuple, S2<:Tuple}
    return binary!(
        reshape_payload(x, data1_payload_shape),
        reshape_payload(y, data2_payload_shape),
    )
end

@inline function scalar_binary_payload(
    binary!::F,
    data1_flat,
    i1::Int,
    j1::Int,
    data2_flat,
    i2::Int,
    j2::Int,
    data1_payload_shape::S1,
    data2_payload_shape::S2,
) where {F, S1<:Tuple, S2<:Tuple}
    return scalar_binary_payload(
        binary!,
        view(data1_flat, i1, j1, :),
        view(data2_flat, i2, j2, :),
        data1_payload_shape,
        data2_payload_shape,
    )
end

@inline function scalar_binary_payload(
    binary!::FlatBinary,
    data1_flat,
    i1::Int,
    j1::Int,
    data2_flat,
    i2::Int,
    j2::Int,
    data1_payload_shape::Tuple,
    data2_payload_shape::Tuple,
)
    return binary!.f(data1_flat, i1, j1, data2_flat, i2, j2)
end

function wrap_binary_payload(
    binary!::F,
    out_payload_shape::NTuple{N,Int},
    data1_payload_shape::S1,
    data2_payload_shape::S2,
) where {F, N, S1<:Tuple, S2<:Tuple}
    return function (dest, x, y)
        binary!(
            reshape(dest, out_payload_shape),
            reshape_payload(x, data1_payload_shape),
            reshape_payload(y, data2_payload_shape),
        )
        return dest
    end
end

function wrap_binary_payload(
    ::FlatBinary,
    out_payload_shape::NTuple{N,Int},
    data1_payload_shape::Tuple,
    data2_payload_shape::Tuple,
) where {N}
    error("FlatBinary only supports scalar-output convolutions")
end

"""
Compute convolution of two functions in 1+1D, given by data1 and data2;

The convolution is assumed to be of the form 

f^R_1(t,x) * f^A_2(-t,-x) = f^R_1(t,x) * f^R_2(t,x)^* 

Therefore, both f^R_1 and f^R_2 are assumed to be retarded functions. 

The input encodes 

data1: f^R_1(ω,k)
data2c: f^R_2(ω,k)

We assume data1 and data2 are already padded, in the form ω=dω*[0,1,2,...,-2,-1,0]. 
"""

function convolve_RRc_notail!(
    binary!::F,
    out::AbstractArray{Tout,No},
    data1::AbstractArray{T1,N1},
    data2::AbstractArray{T2,N2},
    dω::Float64,
) where {F, Tout, No, T1, N1, T2, N2}

    @assert ndims(out) >= 2 && ndims(data1) >= 2 && ndims(data2) >= 2 "all arrays must have at least two dimensions"

    Nω = size(data1, 1)
    L = size(data1, 2)
    @assert size(data2, 1) == Nω && size(data2, 2) == L "input sizes not compatible on convolution axes"
    @assert size(out, 1) == Nω && size(out, 2) == L "output size not compatible on convolution axes"

    out_payload_shape = payload_shape(out)
    data1_payload_shape = payload_shape(data1)
    data2_payload_shape = payload_shape(data2)
    out_flat = reshape(out, Nω, L, :)
    data1_flat = reshape(data1, Nω, L, :)
    data2_flat = reshape(data2, Nω, L, :)

    plan_x_out = plan_ifft!(out, 2)
    plan_t_out = plan_fft!(out, 1)
    plan_x_1 = plan_ifft!(data1, 2)
    plan_t_1 = plan_fft!(data1, 1)
    plan_x_2 = plan_ifft!(data2, 2)
    plan_t_2 = plan_fft!(data2, 1)
    forward_tfactor = dω / (2π)

    mul!(data1, plan_x_1, data1)
    mul!(data1, plan_t_1, data1)
    @. data1 *= forward_tfactor

    mul!(data2, plan_x_2, data2)
    mul!(data2, plan_t_2, data2)
    @. data2 *= forward_tfactor

    if No == 2
        @inbounds for j in axes(out_flat, 2), i in axes(out_flat, 1)
            out_flat[i, j, 1] = scalar_binary_payload(
                binary!,
                data1_flat,
                i,
                j,
                data2_flat,
                i,
                j,
                data1_payload_shape,
                data2_payload_shape,
            )
        end
    else
        binary_flat! = wrap_binary_payload(binary!, out_payload_shape, data1_payload_shape, data2_payload_shape)
        @inbounds for j in axes(out_flat, 2), i in axes(out_flat, 1)
            binary_flat!(
                view(out_flat, i, j, :),
                view(data1_flat, i, j, :),
                view(data2_flat, i, j, :),
            )
        end
    end


    ldiv!(out, plan_x_out, out)
    ldiv!(out, plan_t_out, out)
    @. out /= forward_tfactor

    return out

end

convolve_RRc_notail!(out::AbstractArray, data1::AbstractArray, data2::AbstractArray, dω::Float64) =
    convolve_RRc_notail!(default_binary!, out, data1, data2, dω)

function convolve_RRc_direct!(
    binary!::F,
    out::AbstractArray{Tout,No},
    data1::AbstractArray{T1,N1},
    data2::AbstractArray{T2,N2},
    dω::Float64,
) where {F, Tout, No, T1, N1, T2, N2}
    @assert ndims(out) >= 2 && ndims(data1) >= 2 && ndims(data2) >= 2 "all arrays must have at least two dimensions"

    Nω = size(data1, 1)
    L = size(data1, 2)
    @assert size(data2, 1) == Nω && size(data2, 2) == L "input sizes not compatible on convolution axes"
    @assert size(out, 1) == Nω && size(out, 2) == L "output size not compatible on convolution axes"

    out_payload_shape = payload_shape(out)
    data1_payload_shape = payload_shape(data1)
    data2_payload_shape = payload_shape(data2)
    out_flat = reshape(out, Nω, L, :)
    data1_flat = reshape(data1, Nω, L, :)
    data2_flat = reshape(data2, Nω, L, :)

    prefactor = dω / (2π * L)
    fill!(out, zero(eltype(out)))

    if No == 2
        @inbounds for p in 1:Nω
            for q in 1:L
                value = out_flat[p, q, 1]
                for i in 1:Nω
                    ip = mod1(i - p + 1, Nω)
                    for j in 1:L
                        jq = mod1(j - q + 1, L)
                        value += prefactor * scalar_binary_payload(
                            binary!,
                            data1_flat,
                            i,
                            j,
                            data2_flat,
                            ip,
                            jq,
                            data1_payload_shape,
                            data2_payload_shape,
                        )
                    end
                end
                out_flat[p, q, 1] = value
            end
        end
    else
        binary_flat! = wrap_binary_payload(binary!, out_payload_shape, data1_payload_shape, data2_payload_shape)
        temp = zeros(eltype(out), size(out_flat, 3))
        @inbounds for p in 1:Nω
            for q in 1:L
                dest = view(out_flat, p, q, :)
                for i in 1:Nω
                    ip = mod1(i - p + 1, Nω)
                    for j in 1:L
                        jq = mod1(j - q + 1, L)
                        binary_flat!(temp, view(data1_flat, i, j, :), view(data2_flat, ip, jq, :))
                        @. dest += prefactor * temp
                    end
                end
            end
        end
    end

    return out
end

convolve_RRc_direct!(out::AbstractArray, data1::AbstractArray, data2::AbstractArray, dω::Float64) =
    convolve_RRc_direct!(default_binary!, out, data1, data2, dω)
"""
Convolution similar to convelve_RRc_notail!, but with the tail contribution handled separately by fitting. 


"""

function subtract_fitted_tail!(
    data_flat,
    work_flat,
    ::Nothing,
)
    @inbounds for idx in axes(data_flat, 1), j in axes(data_flat, 2), p in axes(data_flat, 3)
        data_flat[idx, j, p] -= work_flat[idx, j, p]
    end
    return nothing
end

function subtract_fitted_tail!(
    data_flat,
    work_flat,
    physicalindices::AbstractVector,
)
    Nω = size(data_flat, 1)
    physical_mask = falses(Nω)
    @inbounds for idx in physicalindices
        physical_mask[idx] = true
    end
    @inbounds for idx in physicalindices, j in axes(data_flat, 2), p in axes(data_flat, 3)
        data_flat[idx, j, p] -= work_flat[idx, j, p]
    end
    @inbounds for idx in 1:Nω
        if !physical_mask[idx]
            for j in axes(data_flat, 2)
                fill!(view(data_flat, idx, j, :), 0)
            end
        end
    end
    return nothing
end

function convolve_RRC_withtail_impl!(
    binary!::F,
    out::AbstractArray{Tout,No},
    data1::AbstractArray{T1,N1},
    data2::AbstractArray{T2,N2},
    ωs::AbstractVector,
    tailindices::AbstractVector,
    ϵ::Float64,
    tailorder::Int,
    physicalindices,
) where {F, Tout, No, T1, N1, T2, N2}

    @assert ndims(out) >= 2 && ndims(data1) >= 2 && ndims(data2) >= 2 "all arrays must have at least two dimensions"
    Nω = size(data1, 1)
    L = size(data1, 2)
    @assert size(data2, 1) == Nω && size(data2, 2) == L "input sizes not compatible on convolution axes"
    @assert size(out, 1) == Nω && size(out, 2) == L "output size not compatible on convolution axes"
    @assert length(ωs) == Nω "frequency grid size not compatible"
    @assert tailorder >= 0 "tailorder must be non-negative"
    @assert length(tailindices) >= tailorder + 1 "not enough tailindices for requested tailorder"
    @assert all((1 .<= tailindices) .& (tailindices .<= Nω)) "tailindices out of bounds"
    @assert physicalindices === nothing || all((1 .<= physicalindices) .& (physicalindices .<= Nω)) "physicalindices out of bounds"
    dω = Float64(ωs[2] - ωs[1])
    Δt = 2π / (Nω * dω)
    Nt = tailorder + 1

    out_payload_shape = payload_shape(out)
    data1_payload_shape = payload_shape(data1)
    data2_payload_shape = payload_shape(data2)
    out_flat = reshape(out, Nω, L, :)
    data1_flat = reshape(data1, Nω, L, :)
    data2_flat = reshape(data2, Nω, L, :)
    Pout = size(out_flat, 3)
    P1 = size(data1_flat, 3)
    P2 = size(data2_flat, 3)

    plan_x_out = plan_ifft!(out, 2)
    plan_t_out = plan_fft!(out, 1)
    plan_x_1 = plan_ifft!(data1, 2)
    plan_t_1 = plan_fft!(data1, 1)
    plan_x_2 = plan_ifft!(data2, 2)
    plan_t_2 = plan_fft!(data2, 1)
    forward_tfactor = dω / (2π)

    mul!(data1, plan_x_1, data1)
    mul!(data2, plan_x_2, data2)

    Bfit = Matrix{ComplexF64}(undef, length(tailindices), Nt)
    Bfull = Matrix{ComplexF64}(undef, Nω, Nt)
    for n in 0:tailorder
        unit_tail = RegulatedRetardedTail(ϵ, n, 1.0 + 0im)
        @inbounds for j in eachindex(tailindices)
            Bfit[j, n + 1] = fω(unit_tail, ωs[tailindices[j]])
        end
        @inbounds for j in eachindex(ωs)
            Bfull[j, n + 1] = fω(unit_tail, ωs[j])
        end
    end

    fitfact = qr(Bfit)
    rhs1 = Matrix{ComplexF64}(undef, length(tailindices), L * P1)
    rhs2 = Matrix{ComplexF64}(undef, length(tailindices), L * P2)
    a1 = Matrix{ComplexF64}(undef, Nt, L * P1)
    a2 = Matrix{ComplexF64}(undef, Nt, L * P2)
    work1 = Matrix{ComplexF64}(undef, Nω, L * P1)
    work2 = Matrix{ComplexF64}(undef, Nω, L * P2)
    @inbounds for r in eachindex(tailindices)
        idx = tailindices[r]
        for j in 1:L
            for p in 1:P1
                rhs1[r, (j - 1) * P1 + p] = data1_flat[idx, j, p]
            end
            for p in 1:P2
                rhs2[r, (j - 1) * P2 + p] = data2_flat[idx, j, p]
            end
        end
    end

    ldiv!(a1, fitfact, rhs1)
    mul!(work1, Bfull, a1)
    work1_flat = reshape(work1, Nω, L, P1)
    subtract_fitted_tail!(data1_flat, work1_flat, physicalindices)

    ldiv!(a2, fitfact, rhs2)
    mul!(work2, Bfull, a2)
    work2_flat = reshape(work2, Nω, L, P2)
    subtract_fitted_tail!(data2_flat, work2_flat, physicalindices)

    mul!(data1, plan_t_1, data1)
    @. data1 *= forward_tfactor
    mul!(data2, plan_t_2, data2)
    @. data2 *= forward_tfactor

    Tau = Matrix{ComplexF64}(undef, Nω, Nt)
    for n in 0:tailorder
        unit_tail = RegulatedRetardedTail(ϵ, n, 1.0 + 0im)
        @inbounds for j in 1:Nω
            Tau[j, n + 1] = ft(unit_tail, Δt * (j - 1))
        end
    end

    mul!(work1, Tau, a1)
    mul!(work2, Tau, a2)
    tail1_tx = reshape(work1, Nω, L, P1)
    tail2_tx = reshape(work2, Nω, L, P2)
    fill!(out, zero(eltype(out)))

    if No == 2
        @inbounds for j in 1:L
            for i in 1:Nω
                value = out_flat[i, j, 1]
                value += scalar_binary_payload(
                    binary!,
                    data1_flat,
                    i,
                    j,
                    data2_flat,
                    i,
                    j,
                    data1_payload_shape,
                    data2_payload_shape,
                )
                value += scalar_binary_payload(
                    binary!,
                    data1_flat,
                    i,
                    j,
                    tail2_tx,
                    i,
                    j,
                    data1_payload_shape,
                    data2_payload_shape,
                )
                value += scalar_binary_payload(
                    binary!,
                    tail1_tx,
                    i,
                    j,
                    data2_flat,
                    i,
                    j,
                    data1_payload_shape,
                    data2_payload_shape,
                )
                out_flat[i, j, 1] = value
            end
        end
    else
        binary_flat! = wrap_binary_payload(binary!, out_payload_shape, data1_payload_shape, data2_payload_shape)
        temp = zeros(eltype(out), Pout)
        @inbounds for j in 1:L
            for i in 1:Nω
                dest = view(out_flat, i, j, :)
                binary_flat!(temp, view(data1_flat, i, j, :), view(data2_flat, i, j, :))
                @. dest += temp
                binary_flat!(temp, view(data1_flat, i, j, :), view(tail2_tx, i, j, :))
                @. dest += temp
                binary_flat!(temp, view(tail1_tx, i, j, :), view(data2_flat, i, j, :))
                @. dest += temp
            end
        end
    end

    ldiv!(out, plan_t_out, out)
    @. out /= forward_tfactor

    tt_order = 2 * tailorder
    Btt = Matrix{ComplexF64}(undef, Nω, tt_order + 1)
    for n in 0:tt_order
        unit_tail = RegulatedRetardedTail(2 * ϵ, n, 1.0 + 0im)
        @inbounds for j in eachindex(ωs)
            Btt[j, n + 1] = fω(unit_tail, ωs[j])
        end
    end

    a1_flatcoef = reshape(a1, Nt, L, P1)
    a2_flatcoef = reshape(a2, Nt, L, P2)
    att = zeros(ComplexF64, tt_order + 1, L, Pout)
    if No == 2
        @inbounds for x in 1:L
            for n1 in 0:tailorder
                for n2 in 0:tailorder
                    att[n1 + n2 + 1, x, 1] += scalar_binary_payload(
                        binary!,
                        a1_flatcoef,
                        n1 + 1,
                        x,
                        a2_flatcoef,
                        n2 + 1,
                        x,
                        data1_payload_shape,
                        data2_payload_shape,
                    )
                end
            end
        end
    else
        binary_flat! = wrap_binary_payload(binary!, out_payload_shape, data1_payload_shape, data2_payload_shape)
        coefftemp = zeros(eltype(out), Pout)
        @inbounds for x in 1:L
            for n1 in 0:tailorder
                for n2 in 0:tailorder
                    binary_flat!(
                        coefftemp,
                        view(a1_flatcoef, n1 + 1, x, :),
                        view(a2_flatcoef, n2 + 1, x, :),
                    )
                    @views att[n1 + n2 + 1, x, :] .+= coefftemp
                end
            end
        end
    end

    mul!(reshape(out, Nω, :), Btt, reshape(att, tt_order + 1, :), 1, 1)

    ldiv!(out, plan_x_out, out)

    return out
end

function convolve_RRC_withtail!(
    binary!::F,
    out::AbstractArray,
    data1::AbstractArray,
    data2::AbstractArray,
    ωs::AbstractVector,
    tailindices::AbstractVector,
    ϵ::Float64;
    tailorder::Int=0,
    physicalindices::PI=nothing,
) where {F, PI<:Union{Nothing,AbstractVector}}
    return convolve_RRC_withtail_impl!(
        binary!,
        out,
        data1,
        data2,
        ωs,
        tailindices,
        ϵ,
        tailorder,
        physicalindices,
    )
end

convolve_RRC_withtail!(
    out::AbstractArray,
    data1::AbstractArray,
    data2::AbstractArray,
    ωs::AbstractVector,
    tailindices::AbstractVector,
    ϵ::Float64;
    tailorder::Int=0,
    physicalindices::Union{Nothing,AbstractVector}=nothing,
) = convolve_RRC_withtail!(
    default_binary!,
    out,
    data1,
    data2,
    ωs,
    tailindices,
    ϵ;
    tailorder=tailorder,
    physicalindices=physicalindices,
)
