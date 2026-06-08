using Test
using ALPSpline2DPoly

using LinearAlgebra
using Unitful

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
end

# using .ALPSpline2DPoly

# fs_seg1 = s -> (evaluate(seg1, s * u"m"))
# xs1 = 0:0.1:ustrip(seg1.length)
# y = fs_seg1.(xs1)
# display(plot(getindex.(y, 1), getindex.(y, 2)))
#
# seg2 = CubicBezierSegment((0, 0), (1, 1), (2, -1), (3, 0))
# ft_seg2 = t -> (evaluate(seg2, UnitParam(t)))
# fs_seg2 = s -> (evaluate(seg2, s * u"m"))
# println(seg2)
# xs2 = 0:0.1:ustrip(seg2.length)
# y = fs_seg2.(xs2)
# display(plot(getindex.(y, 1), getindex.(y, 2)))
# sleep(10)
#
