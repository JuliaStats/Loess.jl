# These tests don't do much except ensure that the package loads,
# and does something sensible if it does.
using Loess
using Test
using Random
using Statistics
using RDatasets

@testset "basic" begin
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
    @test response(model) == y
    @test modelmatrix(model) == reshape(x, :, 1)
    @test fitted(model) ≈ y
    @test all(isapprox(0; atol=1e-12), residuals(model))
end

@testset "reshaped views" begin
    # See: https://github.com/MakieOrg/AlgebraOfGraphics.jl/pull/462
    # and: https://github.com/JuliaStats/Loess.jl/pull/70
    @test loess(reshape(view(rand(4), 1:4), (4, 1)), rand(4)) isa Loess.LoessModel
end

@testset "sine" begin
    x = 1:10
    y = sin.(1:10)
    model = loess(x, y)
    @test model.xs == reshape(collect(Float64, 1:10), 10, 1)
    @test model.ys == y
    # Test values from R's loess
    pred = [1.0291697629100347, 0.5449767438422979, -0.0006429381818782, -0.7073242734363788, -0.8962317382167664, -0.2611478760650811, 0.6140341389957057, 0.8482042073056935, 0.4572835622716546, -0.5459427790447537]
    # The last element differs by a bit because the implementation that R uses widens the outer vertices a bit
    @test predict(model, x)[1:end - 1] ≈ pred[1:end - 1] atol=1e-5
    @test_broken predict(model, x)[end] ≈ pred[end] atol=1e-5
end

@testset "lots of ties" begin
    # adapted from https://github.com/JuliaStats/Loess.jl/pull/74#discussion_r1294303522
    x = repeat([π/4*i for i in -20:20], inner=101)
    y = sin.(x)

    model = loess(x,y; span=0.2)
    for i in -3:3
        @test predict(model, i * π) ≈ 0 atol=1e-12
        # not great tolerance but loess also struggles to capture the sine peaks
        @test abs(predict(model, i * π + π / 2 )) ≈ 0.9 atol=0.1
    end
end

@test_throws DimensionMismatch loess([1.0 2.0; 3.0 4.0], [1.0])

@testset "Issue 28" begin
    @testset "Example 1" begin
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 2.0, 3.0, 4.0]
        @test predict(loess(x, y, span = 0.25), x) ≈ y
        @test predict(loess(x, y, span = 0.33), x) ≈ y
        @test predict(loess(x, y), x) ≈ y
    end

    @testset "Example 2" begin
        x = [1.0, 1.0, 2.0, 3.0, 4.0, 4.0]
        y = [1.0, 1.0, 2.0, 3.0, 4.0, 4.0]
        @test predict(loess(x, y, span = 0.33), x) ≈ y
        @test predict(loess(x, y, span = 0.4), x) ≈ y
        @test predict(loess(x, y, span = 0.5), x) ≈ y
        @test predict(loess(x, y, span = 0.6), x) ≈ y
    end
end

@testset "cars" begin
    df = dataset("datasets", "cars")
    ft = loess(df.Speed, df.Dist)

    # Test values from R's loess expect outer vertices as they are made wider in the R/C/Fortran implementation
    @testset "vertices" begin
        @test sort(getindex.(keys(ft.predictions_and_gradients))) == [4.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 17.0, 19.0, 22.0, 25.0]
    end

    @testset "predict" begin
        # In R this is `predict(cars.lo, data.frame(speed = seq(5, 25, 1)))`.
        R_fit = [7.797353, 10.002308, 12.499786, 15.281082, 18.446568, 21.865315, 25.517015, 29.350386, 33.230660, 37.167935, 41.205226, 45.055736, 48.355889, 49.824812, 51.986702, 56.461318, 61.959729, 68.569313, 76.316068, 85.212121, 95.324047]
        R_se_fit = [7.568120, 5.945831, 4.990827, 4.545284, 4.308639, 4.115049, 3.789542, 3.716231, 3.776947, 4.091747, 4.709568, 4.245427, 4.035929, 3.753410, 4.004705, 4.043190, 4.026105, 4.074664, 4.570818, 5.954217, 8.302014]
        preds = predict(ft, 5:25; interval = :confidence)
        # The calcalations of the δs in code of Cleveland and Grosse are
        # based on a statistical approximation, see section 4 of their 1991
        # paper. We use the exact but expensive definition so the values
        # are slightly different
        @test preds.s ≈ 15.29496 rtol = 1e-3
        @test preds.ρ ≈ 44.6179 rtol = 5e-3

        for (i, Ry) in enumerate(R_fit)
            # The outer vertices are expanded by 0.105 in the original implementation. Not sure if we
            # want to do the same thing so meanwhile the results will deviate slightly between the
            # outermost vertices
            @test preds.predictions[i] ≈ Ry rtol = (4 <= i <= 18) ? 1e-7 : 1e-3
            @test preds.sₓ[i] ≈ R_se_fit[i] rtol = 1e-3
        end

        preds_all = predict(ft; interval = :confidence)
        @test preds_all.s ≈ 15.29496 rtol = 1e-3
        @test preds_all.ρ ≈ 44.6179 rtol = 5e-3
        @test preds_all.sₓ ≈ [9.885466, 9.885466, 4.990827, 4.990827, 4.545284, 4.308639, 4.115049, 4.115049, 4.115049, 3.789542, 3.789542, 3.716231, 3.716231, 3.716231, 3.716231, 3.776947, 3.776947, 3.776947, 3.776947, 4.091747, 4.091747, 4.091747, 4.091747, 4.709568, 4.709568, 4.709568, 4.245427, 4.245427, 4.035929, 4.035929, 4.035929, 3.753410, 3.753410, 3.753410, 3.753410, 4.004705, 4.004705, 4.004705, 4.043190, 4.043190, 4.043190, 4.043190, 4.043190, 4.074664, 4.570818, 5.954217, 5.954217, 5.954217, 5.954217, 8.302014] rtol = 1e-2

        @test_throws ArgumentError predict(ft, 5:25; interval = :x)
        @test_throws ArgumentError predict(ft, 5:25; interval = :prediction)
    end
end

@testset "small datasets. Issue 82" begin
    ft = loess([1.0], [1.0])
    @test predict(ft, 1.0) == 1.0
    @test_throws ArgumentError loess(Float64[], Float64[])
end

@testset "infinite recursion. Issue 60" begin
    x = collect(1.0:10.0)
    y = convert(Vector{Union{Nothing, Float64}}, x)
    @test_throws MethodError loess(x, y)
end
