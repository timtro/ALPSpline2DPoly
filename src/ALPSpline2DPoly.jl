"""
    module ALPSpline

Arc-Length Parameterised Spline segments in ℝ², with parameterisation
approximated with cubic polynomial fitting.
TODO: upgrade to Chebychev polynomial fit.

Provides two segment types:
  - `LinearSegment`       — straight line between two endpoints.
  - `CubicBezierSegment`  — cubic Bézier with four control points.

Reference:
  T. de Wolff, https://tacodewolff.nl/posts/20190525-arc-length/
  N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM 2013.
"""
module ALPSpline2DPoly

using Unitful: Unit
using Unitful
using QuadGK: quadgk
using Roots

using StaticArrays
using LinearAlgebra

export Vec2, Vec3, UnitParam, LinearSegment, CubicBezierSegment, evaluate

const Vec2 = SVector{2,Float64}
const Vec3 = SVector{3,Float64}

const num_tol = 1e-12

struct UnitParam
  value::Float64
  function UnitParam(v::Real)
    0 ≤ v ≤ 1 || throw(DomainError(v, "UnitParam requires value ∈ [0, 1]"))
    new(Float64(v))
  end
end

import Base: +, -, *, /, ==, <, ≤, show

# Arithmetic returns plain Float64 (result may leave [0,1], so unwrap)
for op in (:+, :-, :*, :/)
  @eval ($op)(a::UnitParam, b::UnitParam) = ($op)(a.value, b.value)
  @eval ($op)(a::UnitParam, b::Real) = ($op)(a.value, b)
  @eval ($op)(a::Real, b::UnitParam) = ($op)(a, b.value)
  @eval ($op)(a::UnitParam, b::SVector) = ($op)(a.value, b)
end
Base.:^(a::UnitParam, b::Real) = a.value^b

(==)(a::UnitParam, b::UnitParam) = a.value == b.value
(<)(a::UnitParam, b::UnitParam) = a.value < b.value
show(io::IO, t::UnitParam) = print(io, "UnitParam($(t.value))")

"""
    LinearSegment(p0, p1)

Straight-line segment from `p0` to `p1` in ℝ².
Arc-length parameterisation is exact: s(t) = t · ‖p1 − p0‖.
"""
struct LinearSegment
  p0::Vec2
  p1::Vec2
  length::Unitful.Length

  function LinearSegment(p0, p1)
    v0, v1 = Vec2(p0), Vec2(p1)
    new(v0, v1, norm(v1 - v0)u"m")
  end
end

"""
    CubicBezierSegment(p0, p1, p2, p3)

Cubic Bézier segment in ℝ²:

    B(t) = (1−t)³ p0 + 3(1−t)²t p1 + 3(1−t)t² p2 + t³ p3,  t ∈ [0,1]

At construction ...
"""
struct CubicBezierSegment
  p0::Vec2
  p1::Vec2
  p2::Vec2
  p3::Vec2
  length::Unitful.Length
  t_to_s_coeff::Vec3
  s_to_t_coeff::Vec3

  function CubicBezierSegment(p0, p1, p2, p3)
    p₀, p₁, p₂, p₃ = Vec2(p0), Vec2(p1), Vec2(p2), Vec2(p3)

    speed = t -> norm(3 * (1 - t)^2 * (p₁ - p₀) + 6 * (1 - t) * t * (p₂ - p₁) + 3 * t^2 * (p₃ - p₂))
    total_length, err₂ = quadgk(speed, 0.0, 1.0, rtol=num_tol)

    ℓ₁, err₀ = quadgk(speed, 0.0, 1.0 / 3.0, rtol=num_tol)
    ℓ₂, err₁ = quadgk(speed, 0.0, 2.0 / 3.0, rtol=num_tol)
    ℓ₃ = total_length

    t_to_s = Vec3(
      13.5 * ℓ₁ - 13.5 * ℓ₂ + 4.5 * ℓ₃,  #
      -22.5 * ℓ₁ + 18.0 * ℓ₂ - 4.5 * ℓ₃, #
      9.0 * ℓ₁ - 4.5 * ℓ₂ + ℓ₃,
    )

    param_at = function (len)
      f = t -> (quadgk(speed, 0.0, t, rtol=num_tol)[1] - len)
      return find_zero((f, speed), len / total_length, Roots.Newton())
    end

    t₁ = param_at(total_length / 3.0)
    t₂ = param_at(2.0 * total_length / 3.0)

    s_to_t = Vec3(
      (13.5 * t₁ - 13.5 * t₂ + 4.5) / total_length^3,
      (-22.5 * t₁ + 18.0 * t₂ - 4.5) / total_length^2,
      (9.0 * t₁ - 4.5 * t₂ + 1) / total_length,
    )

    new(p₀, p₁, p₂, p₃, total_length * u"m", t_to_s, s_to_t)
  end
end

module util
using Unitful
using ..ALPSpline2DPoly: UnitParam, CubicBezierSegment
export t_to_s, s_to_t
function s_to_t(seg::CubicBezierSegment, s::Unitful.Length)::UnitParam
  0 * u"m" ≤ s ≤ seg.length || throw(DomainError(s, "s_to_t(seg, s) requires value s ∈ [0, seg.length]"))
  if (s == 0.0 * u"m")
    return UnitParam(0.0)
  end
  if (s == seg.length)
    return UnitParam(1.0)
  end

  α, β, γ = seg.s_to_t_coeff[1], seg.s_to_t_coeff[2], seg.s_to_t_coeff[3]

  s = ustrip(u"m", s)
  s² = s * s
  s³ = s² * s

  return UnitParam(α * s³ + β * s² + γ * s)
end

function t_to_s(seg::CubicBezierSegment, t::UnitParam)::Unitful.Length
  if (t == 0.0)
    return 0u"m"
  end
  if (t == 1.0)
    return seg.length
  end

  α = seg.t_to_s_coeff[1]
  β = seg.t_to_s_coeff[2]
  γ = seg.t_to_s_coeff[3]

  t² = t * t
  t³ = t² * t

  return (α * t³ + β * t² + γ * t) * u"m"
end
end # module util

using .util: t_to_s, s_to_t

function evaluate(seg::LinearSegment, t::UnitParam)::SVector
  seg.p0 + t * (seg.p1 - seg.p0)
end
function evaluate(seg::LinearSegment, s::Unitful.Length)::SVector
  t = UnitParam(s / seg.length)
  seg.p0 + t * (seg.p1 - seg.p0)
end

(γ::LinearSegment)(t::UnitParam) = evaluate(γ, t)
(γ::LinearSegment)(s::Unitful.Length) = evaluate(γ, s)


function evaluate(seg::CubicBezierSegment, t::UnitParam)::SVector
  t′ = 1.0 - t
  t′^3 * seg.p0 + 3t′^2 * t * seg.p1 + 3t′ * t^2 * seg.p2 + t^3 * seg.p3
end

function evaluate(seg::CubicBezierSegment, s::Unitful.Length)::SVector
  return evaluate(seg, s_to_t(seg, s))
end

(γ::CubicBezierSegment)(t::UnitParam) = evaluate(γ, t)
(γ::CubicBezierSegment)(s::Unitful.Length) = evaluate(γ, s)
end # module
