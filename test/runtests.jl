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
