# These tests don't do much except ensure that the package loads,
# and does something sensible if it does.
using Loess
using Base.Test

srand(100)
xs = 10 .* rand(100)
ys = sin(xs) .+ 0.5 * rand(100)

model = loess(xs, ys)

us = collect(minimum(xs):0.1:maximum(xs))
vs = predict(model, us)

@test minimum(vs) >= -1.1
@test maximum(vs) <= +1.1
