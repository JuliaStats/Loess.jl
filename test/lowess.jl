@testset "Lowess Basic Tests" begin
    @testset "Testing with floating point numbers" begin
        for i = 1:500
            n = rand(6:100)
            xs = unique(sort(rand(1:100000000, 5*n)))[1:n] ./ 100000000
            ys = rand(1:100000000, 5*n)[1:n] ./ 100000000
            f = rand(1:100000000) / 100000000
            nsteps = Int(rand(1:10))
            delta = rand(0:100000000) / 100000000

            zs = lowess(xs, ys, f, nsteps, delta)
            @test length(zs) == length(ys)
        end
    end

    @testset "Testing with integers" begin
        for i = 1:500
            n = rand(6:100)
            xs = rand(1:100, n)
            xs = sort(xs)
            ys = rand(1:100, n)
            zs = lowess(xs, ys, rand(Uniform(0.1, 1.0)), rand(3:10), rand(Uniform(0.0, 1.0)))
            @test length(zs) == length(ys)
        end
    end
end

@testset "Lowess Bounds Test" begin
    xs = 10 .* rand(100)
    ys = sin.(xs) .+ 0.5 * rand(100)
    zs = lowess(xs, ys)
    @test minimum(zs) >= -1.1
    @test maximum(zs) <= 1.1
end

@testset "Comparing against results of the original C code using random inputs." begin
    @testset "test 1" begin
        xs  = [0.03, 0.16, 0.37, 0.58, 0.71, 0.85, 0.92, 0.99]
        ys  = [0.21, 0.62, 0.05, 0.92, 0.81, 0.5, 0.65, 0.23]
        f   = 0.645
        nsteps  = 4
        delta   = 0.506
        zs = [0.33975588875617213, 0.35757371725482023, 0.3863563632910978, 0.4728384416628679, 0.5263749663692019, 0.5840296852837153, 0.45763988305199876, 0.3312500808202824]
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 2" begin
        xs  = [0.34, 0.37, 0.38, 0.41, 0.51, 0.77, 0.96]
        ys  = [0.15, 0.26, 0.45, 0.94, 0.92, 0.75, 0.27]
        f   = 0.243
        nsteps  = 2
        delta   = 0.728
        zs = [0.15, 0.1558064516129032, 0.15774193548387097, 0.1635483870967742, 0.18290322580645163, 0.23322580645161292, 0.27]
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 3" begin
        xs  = [0.17, 0.19, 0.34, 0.37, 0.46, 0.51, 0.66]
        ys  = [0.12, 0.19, 0.76, 0.89, 0.81, 0.73, 0.41]
        f   = 0.058
        nsteps  = 4
        delta   = 0.539
        zs = [0.12, 0.13183673469387755, 0.2206122448979592, 0.23836734693877548, 0.2916326530612245, 0.32122448979591833, 0.41]
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 4" begin
        xs  = [0.26, 0.39, 0.48, 0.51, 0.59, 0.94]
        ys  = [0.85, 0.24, 0.38, 0.97, 0.79, 0.23, 0.89]
        f   = 0.056
        nsteps  = 5
        delta   = 0.516
        zs = [0.85, 0.8263636363636363, 0.81, 0.8045454545454546, 0.79, 0.23]
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 5" begin
        xs  = [0.06, 0.32, 0.46, 0.78, 0.94, 0.99]
        ys  = [0.26, 0.91, 0.14, 0.33, 0.57, 0.99, 0.56]
        f   = 0.952
        nsteps  = 1
        delta   = 0.427
        zs = [0.4003188954738399, 0.4282100726879575, 0.44322839888017473, 0.4404274424163527, 0.7243844287810028, 0.8131209870199563]
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

end 
