
# Loess

This is a pure Julia loess implementation, based on the fast kd-tree based
approximation described in the original Cleveland, et al, implemented netlib
loess C/Fortran code, and used by many, including in R's loess function.


## Synopsis

`Loess` exports two function: `loess` and `predict`, than train and apply the model, respectively.


```julia
using Loess

xs = 10 .* rand(100)
ys = sin(xs) .+ 0.5 * rand(100)

model = loess(xs, ys)

us = collect(min(xs):0.1:max(xs))
vs = predict(model, us)

using Gadfly
p = plot(x=xs, y=ys, Geom.point, Guide.xlabel("x"), Guide.ylabel("y"),
         layer(Geom.line, x=zs, y=ws))
draw(SVG("loess.svg", 6inch, 3inch), p)
```


## Status

Multivariate regression is not yet fully implemented, but most of the parts
are aready there, and wouldn't require too much additional work.
