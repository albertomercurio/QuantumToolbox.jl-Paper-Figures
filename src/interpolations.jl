#=
This file contains various interpolation functions and utilities that are Zygote-compatible.
=#

struct UniformQuadraticItp{T<:AbstractVector}
    t0::Float64
    Δ::Float64
    y::T
end

function UniformQuadraticItp(t::AbstractVector, y::AbstractVector)
    @assert length(t) == length(y) "t and y size mismatch"
    @assert length(t) ≥ 3 "need at least 3 points for quadratic"
    Δ = t[2] - t[1]
    @assert all(abs.(diff(t) .- Δ) .< 1e-10*abs(Δ)) "t must be uniform"
    return UniformQuadraticItp{typeof(y)}(float(t[1]), float(Δ), y)
end

function (itp::UniformQuadraticItp)(x::Real)
    t0, Δ, y = itp.t0, itp.Δ, itp.y
    N = length(y)
    s  = (x - t0)/Δ
    j  = clamp(round(Int, s) + 1, 2, N-1)     # center index
    ξ  = s - (j-1)                             # in [-0.5, 0.5] typically
    Lm = 0.5*ξ*(ξ - 1.0)
    L0 = 1.0 - ξ*ξ
    Lp = 0.5*ξ*(ξ + 1.0)
    @inbounds return y[j-1]*Lm + y[j]*L0 + y[j+1]*Lp
end

struct UniformHermiteCR{T<:AbstractVector}
    t0::Float64
    Δ::Float64
    y::T
end

function UniformHermiteCR(t::AbstractVector, y::AbstractVector)
    @assert length(t)==length(y) && length(y)≥4
    Δ = t[2]-t[1]; @assert all(abs.(diff(t) .- Δ) .< 1e-10*abs(Δ))
    UniformHermiteCR{typeof(y)}(float(t[1]), float(Δ), y)
end

function (itp::UniformHermiteCR)(x::Real)
    t0, Δ, y = itp.t0, itp.Δ, itp.y; N = length(y)
    s = (x - t0)/Δ
    i = clamp(floor(Int, s)+1, 1, N-1)      # 2..N-1
    u = s - (i-1)                            # in [0,1)

    # tangents (central inside; one-sided at the ends)
    @inbounds m_i   = (i == 1)   ? (y[2]   - y[1])   :
                      (i == N-1) ? (y[N-1] - y[N-2]) :
                                   0.5*(y[i+1] - y[i-1])
    @inbounds m_ip1 = (i+1 == N) ? (y[N]   - y[N-1]) :
                      (i == 1)   ? 0.5*(y[3] - y[1]) :
                                   0.5*(y[i+2] - y[i])

    u2 = u*u; u3 = u2*u
    h00 =  2u3 - 3u2 + 1
    h10 =      u3 - 2u2 + u
    h01 = -2u3 + 3u2
    h11 =      u3 -   u2
    @inbounds return h00*y[i] + h10*m_i + h01*y[i+1] + h11*m_ip1
end
