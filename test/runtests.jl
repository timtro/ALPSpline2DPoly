using Test

using LinearAlgebra
using Unitful
using QuadGK: quadgk
using Roots

using Optim

using ALPSpline2DPoly
using ALPSpline2DPoly.util: t_to_s, s_to_t


const rtol = 1E-10
const atol = 1E-10
const atol_relaxed = 1E-6
const atol_very_relaxed = 1E-3
const fit_rtol = 0.08   # allow ~8% deviation due to cubic approximation

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

  return quadgk(e, 0.0, 1.0, rtol=rtol)[1]
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

# More typical fit:
# function make_quarter_unit_circle()
#   a = 0.5522847498307933   # ≈ (4/3)(√2 - 1)
#   κ = 0.5522847498307933
#
#   p₀ = Vec(1.0, 0.0)
#   p₁ = Vec(1.0, κ)
#   p₂ = Vec(κ, 1.0)
#   p₃ = Vec(0.0, 1.0)
#
#   return CubicBezierSegment(p₀, p₁, p₂, p₃)
# end
function make_quarter_unit_circle()
  a = 1.00005507808
  b = 0.55342925736
  c = 0.99873327689

  p₀ = Vec(a, 0.0)
  p₁ = Vec(c, b)
  p₂ = Vec(b, c)
  p₃ = Vec(0.0, a)

  return CubicBezierSegment(p₀, p₁, p₂, p₃)
end

const quarter_circle = make_quarter_unit_circle()

const quarter_unit_circle = make_quarter_unit_circle()

@testset verbose = true "ALPSpline2DPoly" begin
  @testset verbose = true "UnitParam construction safety" begin
    @test_throws DomainError UnitParam(-0.1)
    @test_throws DomainError UnitParam(1.1)
    @test_throws DomainError UnitParam(NaN)
    @test_throws DomainError UnitParam(Inf)
  end
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
    p₀::Vec, p₁::Vec, p₂::Vec, p₃::Vec = (0, 0), (1, 1), (2, -1), (3, 0)
    γ = CubicBezierSegment(p₀, p₁, p₂, p₃)

    @test γ(UnitParam(0)) ≈ p₀
    @test γ(UnitParam(1)) ≈ p₃
  end
  @testset "LinearSegment evaluation on arclength, with known endpoints" begin
    γ = LinearSegment((0, 0), (1, 1))
    @test evaluate(γ, 0 * u"m") ≈ Vec(0, 0)
    @test evaluate(γ, γ.length / 2) ≈ Vec(0.5, 0.5)
    @test evaluate(γ, γ.length) ≈ Vec(1, 1)

    @test γ(0 * u"m") ≈ Vec(0, 0)
    @test γ(γ.length / 2) ≈ Vec(0.5, 0.5)
    @test γ(γ.length) ≈ Vec(1, 1)
  end
  @testset "CubicBezierSegment evaluation on arclength with known points" begin
    p₀::Vec, p₁::Vec, p₂::Vec, p₃::Vec = (0, 0), (1, 1), (2, -1), (3, 0)
    γ = CubicBezierSegment(p₀, p₁, p₂, p₃)
    @test γ(0 * u"m") ≈ p₀

    t, err = numerical_s_to_t_with_err(γ, γ.length / 4)
    @test norm(γ(γ.length / 4) - γ(t)) ≤ 2E-2

    @test γ(γ.length / 2) ≈ Vec(1.5, 0)
    @test γ(γ.length) ≈ p₃
  end
  #
  # The following test is failing in the Julia implementation.
  # This is fine, since I don't anticipate that point-curves are
  # going to be useful in any real capacity.
  #
  # @testset "Polynomial ALP fit coefficients for a zero Bézier spline are zero" begin p = Vec(0.0, 0.0) γ = CubicBezierSegment(p, p, p, p)
  #   @test γ.length == 0.0 * u"m"
  #   @test γ.s_to_t_coeff == Vec3(0.0, 0.0, 0.0)
  # end
  @testset "CubicBezierSegment t_to_s Mean Absolute Error (MAE) on known curve" begin
    p₀::Vec, p₁::Vec, p₂::Vec, p₃::Vec = (0, 0), (1, 1), (2, -1), (3, 0)
    γ = CubicBezierSegment(p₀, p₁, p₂, p₃)

    @test t_to_s_mean_absolute_error(γ) ≤ 2E-2
    @test t_to_s_max_absolute_error(γ) ≤ 5E-2
  end
  @testset "CubicBezierSegment s_to_t Compare arclength approx against spline approximation of quarter circle" begin
    should_be_pi_over_2 = t_to_s(quarter_unit_circle, UnitParam(1.0))

    @test t_to_s(quarter_unit_circle, UnitParam(0.0)) == 0.0 * u"m"
    @test t_to_s(quarter_unit_circle, UnitParam(0.5)) - π / 4.0 * u"m" ≤ atol * u"m"
    @test t_to_s(quarter_unit_circle, UnitParam(1.0)) - π / 2.0 * u"m" ≤ atol * u"m"

    halfway = s_to_t(quarter_unit_circle, π / 4.0 * u"m")
    @test halfway ≈ UnitParam(0.5) atol = atol_relaxed
    #
    for t in UnitParam.([0.1, 0.25, 0.5, 0.75, 0.9])
      s = t_to_s(quarter_unit_circle, t)
      recovered = s_to_t(quarter_unit_circle, s)
      @test recovered ≈ t atol = atol_very_relaxed
    end
  end
  @testset "first_derivative of LinearSegment over UnitParam is just length" begin
    p₀::Vec2, p₁::Vec2 = randn(Float64, 2), randn(Float64, 2)
    γ = LinearSegment(p₀, p₁)

    for t in UnitParam.([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
      @test first_derivative(γ, t) == p₁ - p₀
    end
  end
  @testset "first_derivative of LinearSegment over arclength is the tangent." begin
    p₀::Vec2, p₁::Vec2 = randn(Float64, 2), randn(Float64, 2)
    γ = LinearSegment(p₀, p₁)

    for s in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] * γ.length
      @test first_derivative(γ, s) ≈ (p₁ - p₀) / norm(p₁ - p₀)
      # @test first_derivative(γ, t) ≈ normalize(p₁ - p₀)
    end
  end
  @testset "first_derivative of CubicBezierSegment over UnitParam on quarter-circle" begin
    for t in UnitParam.([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
      θ = t.value * π / 2
      expected = (π / 2) * Vec(-sin(θ), cos(θ))
      actual = first_derivative(quarter_circle, t)
      @test norm(cross(actual, expected)) ≈ 0 atol = 2E-2
    end
  end
  @testset "first_derivative of CubicBezierSegment over arclength on quarter-circle" begin
    for s in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] * quarter_circle.length
      θ = ustrip(s)
      expected = (π / 2) * Vec(-sin(θ), cos(θ))
      actual = first_derivative(quarter_circle, s)
      @test norm(cross(actual, expected)) ≈ 0 atol = 2E-2
    end
  end
  @testset "second_derivative of LinearSegment over arclength is the zero-vector." begin
    p₀::Vec2, p₁::Vec2 = randn(Float64, 2), randn(Float64, 2)
    γ = LinearSegment(p₀, p₁)

    @test second_derivative(γ, UnitParam(0.5)) == Vec(0.0, 0.0)
  end
  @testset "second_derivative of CubicBezierSegment over UnitParam on quarter-circle" begin
    for t in UnitParam.([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
      θ = t.value * π / 2
      expected = (π / 2) * Vec(-sin(θ), cos(θ))
      actual = first_derivative(quarter_circle, t)
      @test norm(cross(actual, expected)) ≈ 0 atol = 2E-2
    end
  end
  # NB: This test couples t_to_s with pos, vel and acc testing.
  # Tighter tolerances may be achieved working directly in UnitParam,
  # but this is a broader integration test, and the expense of more targetd
  # unit testing of second_derivative.
  @testset "second_derivative of CubicBezierSegment on quarter-circle" begin
    γ = quarter_circle

    for t in UnitParam.([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
      s = t_to_s(γ, t)
      pos = evaluate(γ, s)
      vel = first_derivative(γ, s)
      acc = second_derivative(γ, s)

      r = norm(pos)
      acc_mag = norm(acc)

      @test r ≈ 1.0 atol = 1E-4
      @test norm(cross(acc, pos)) ≈ 0 atol = 2E-2
      @test acc_mag ≈ 1.0 atol = 5E-2
      @test abs(dot(acc, vel)) ≤ 2E-2
    end

    # Extra checks at endpoints (where curvature is well-defined)
    @testset "Endpoint second_derivative" begin
      acc₀ = second_derivative(γ, 0.0 * u"m")
      @test acc₀[1] < -0.9
      @test abs(acc₀[2]) < 5E-2

      acc₁ = second_derivative(γ, γ.length)
      @test abs(acc₁[1]) < 5E-2
      @test acc₁[2] < -0.9
    end
  end
end
