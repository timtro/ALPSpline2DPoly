using Test

using LinearAlgebra
using Unitful
using QuadGK: quadgk
using Roots

using Optim

using ALPSpline2DPoly
using ALPSpline2DPoly.util: t_to_s, s_to_t


const rtol = 1E-10

"""
    numerical_s_to_t_with_err

A straightforward Newton-Raphson algorithm for finding the
parameter of a given arclength, integrated by quadrature.

The error returned is the quadrature error, not an
estimate of the residual from the Newton iteration.

"""
function numerical_s_to_t_with_err(seg::CubicBezierSegment, s::Unitful.Length)
  p₀, p₁, p₂, p₃ = seg.p0, seg.p1, seg.p2, seg.p3
  s′ = ustrip(u"m", s)
  speed = t -> norm(3 * (1 - t)^2 * (p₁ - p₀) + 6 * (1 - t) * t * (p₂ - p₁) + 3 * t^2 * (p₃ - p₂))
  newton = function (atol=1e-10, maxiter=50)
    t = ustrip(seg.length) / s′
    for _ in 1:maxiter
      arc, err = quadgk(speed, 0.0, t, rtol=rtol)
      δ = (arc - s′) / speed(t)
      t -= δ
      abs(δ) < atol && return t, err
    end
    error("Newton did not converge")
  end

  t, ε = newton()

  return UnitParam(t), ε
end

function t_to_s_mean_absolute_error(seg::CubicBezierSegment)::Float64
  p₀, p₁, p₂, p₃ = seg.p0, seg.p1, seg.p2, seg.p3
  speed = t -> norm(3 * (1 - t)^2 * (p₁ - p₀) + 6 * (1 - t) * t * (p₂ - p₁) + 3 * t^2 * (p₃ - p₂))
  e = function (t::Float64)
    return abs(
      (quadgk(speed, 0.0, t, rtol=rtol)[1]) -
      ustrip(u"m", t_to_s(seg, UnitParam(t)))
    )
  end

  return quadgk(e, 0.0, 1.0, rtol=1E-8)[1]
end

function t_to_s_max_absolute_error(seg::CubicBezierSegment)
  p₀, p₁, p₂, p₃ = seg.p0, seg.p1, seg.p2, seg.p3
  speed = t -> norm(3 * (1 - t)^2 * (p₁ - p₀) + 6 * (1 - t) * t * (p₂ - p₁) + 3 * t^2 * (p₃ - p₂))
  e = function (t::Float64)
    return abs(
      (quadgk(speed, 0.0, t, rtol=rtol)[1]) -
      ustrip(u"m", t_to_s(seg, UnitParam(t)))
    )
  end

  result = optimize(t -> -e(t), 0.0, 1.0, Brent())
  if Optim.converged(result)
    return -Optim.minimum(result)
  else
    throw(ErrorException("t_to_s_max_absolute_error failed to converge."))
  end
end

@testset verbose = true "ALPSpline2DPoly" begin
  @testset "LinearSegment evaluation on UnitParam with known endpoints" begin
    γ = LinearSegment((0, 0), (1, 1))
    @test evaluate(γ, UnitParam(0)) ≈ Vec2(0, 0)
    @test evaluate(γ, UnitParam(0.5)) ≈ Vec2(0.5, 0.5)
    @test evaluate(γ, UnitParam(1)) ≈ Vec2(1, 1)

    @test γ(UnitParam(0)) ≈ Vec2(0, 0)
    @test γ(UnitParam(0.5)) ≈ Vec2(0.5, 0.5)
    @test γ(UnitParam(1)) ≈ Vec2(1, 1)
  end
  @testset "LinearSegment evaluation on UnitParam with random endpoints" begin
    p₀::Vec2, p₁::Vec2 = randn(Float64, 2), randn(Float64, 2)
    γ = LinearSegment(p₀, p₁)
    @test γ.length ≈ norm(p₁ - p₀) * u"m"

    @test γ(UnitParam(0)) ≈ p₀
    @test γ(UnitParam(0.5)) ≈ p₀ + 0.5 * (p₁ - p₀)
    @test γ(UnitParam(1)) ≈ p₁
  end
  @testset "CubicBezierSegment evaluation on UnitParam with known points" begin
    p₀::Vec2, p₁::Vec2, p₂::Vec2, p₃::Vec2 = (0, 0), (1, 1), (2, -1), (3, 0)
    γ = CubicBezierSegment(p₀, p₁, p₂, p₃)

    @test γ(UnitParam(0)) ≈ p₀
    @test γ(UnitParam(1)) ≈ p₃
  end
  @testset "LinearSegment evaluation on arclength, with known endpoints" begin
    γ = LinearSegment((0, 0), (1, 1))
    @test evaluate(γ, 0 * u"m") ≈ Vec2(0, 0)
    @test evaluate(γ, γ.length / 2) ≈ Vec2(0.5, 0.5)
    @test evaluate(γ, γ.length) ≈ Vec2(1, 1)

    @test γ(0 * u"m") ≈ Vec2(0, 0)
    @test γ(γ.length / 2) ≈ Vec2(0.5, 0.5)
    @test γ(γ.length) ≈ Vec2(1, 1)
  end
  @testset "CubicBezierSegment evaluation on arclength with known points" begin
    p₀::Vec2, p₁::Vec2, p₂::Vec2, p₃::Vec2 = (0, 0), (1, 1), (2, -1), (3, 0)
    γ = CubicBezierSegment(p₀, p₁, p₂, p₃)
    @test γ(0 * u"m") ≈ p₀

    t, err = numerical_s_to_t_with_err(γ, γ.length / 4)
    @test norm(γ(γ.length / 4) - γ(t)) ≤ 2E-2

    @test γ(γ.length / 2) ≈ Vec2(1.5, 0)
    @test γ(γ.length) ≈ p₃
  end
  @testset "CubicBezierSegment t_to_s Mean Absolute Error (MAE) on known curve" begin
    p₀::Vec2, p₁::Vec2, p₂::Vec2, p₃::Vec2 = (0, 0), (1, 1), (2, -1), (3, 0)
    γ = CubicBezierSegment(p₀, p₁, p₂, p₃)

    @test t_to_s_mean_absolute_error(γ) ≤ 2E-2
    @test t_to_s_max_absolute_error(γ) ≤ 5E-2
  end
end
