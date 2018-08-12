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
