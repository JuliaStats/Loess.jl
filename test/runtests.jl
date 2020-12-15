# These tests don't do much except ensure that the package loads,
# and does something sensible if it does.
using Loess
using Test
using Random
using Statistics

Random.seed!(100)
xs = 10 .* rand(100)
ys = sin.(xs) .+ 0.5 * rand(100)

model = loess(xs, ys)

us = collect(minimum(xs):0.1:maximum(xs))
vs = predict(model, us)

@test minimum(vs) >= -1.1
@test maximum(vs) <= +1.1


x = [13.0,14.0,14.35,15.0,16.0]
y = [0.369486,  0.355579, 0.3545, 0.356952, 0.36883]
model = loess(x,y)
@test Loess.predict(model,x) ≈ y

let x = 1:10, y = sin.(1:10)
    model = loess(x, y)
    @test model.xs == reshape(collect(Float64, 1:10), 10, 1)
    @test model.ys == y
    pred = [1.02866, 0.798561, 0.533528, 0.253913, -0.0325918, -0.319578, -0.648763]
    @test predict(model, 1.0:0.5:4.0) ≈ pred atol=1e-5
end

@test_throws DimensionMismatch loess([1.0 2.0; 3.0 4.0], [1.0])

@testset "Issue 28" begin
    @testset "Example 1" begin
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 2.0, 3.0, 4.0]
        @test_throws ArgumentError("neighborhood size must be larger than degree+1=3 but was 1. Try increasing the value of span.") loess(x, y, span = 0.25)
        @test_throws ArgumentError("neighborhood size must be larger than degree+1=3 but was 2. Try increasing the value of span.") loess(x, y, span = 0.33)
        @test predict(loess(x, y), x) ≈ x
    end

    @testset "Example 2" begin
        x = [1.0, 1.0, 2.0, 3.0, 4.0, 4.0]
        y = [1.0, 1.0, 2.0, 3.0, 4.0, 4.0]
        @test_throws ArgumentError("neighborhood size must be larger than degree+1=3 but was 2. Try increasing the value of span.") loess(x, y, span = 0.33)
        # For 0.4 and 0.5 these current don't hit the middle values. I suspect
        # the issue is related to the ties in x.
        @test_broken predict(loess(x, y, span = 0.4), x) ≈ x
        @test_broken predict(loess(x, y, span = 0.5), x) ≈ x
        @test predict(loess(x, y, span = 0.6), x) ≈ x
    end
end
