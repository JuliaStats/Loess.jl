using Distributions

@testset "Lowess Basic Tests" begin
    @testset "Testing with floating point numbers" begin
        for i = 1:500
            n = rand(6:100)
            xs = rand(Uniform(1.0, 100.0), n)
            xs = sort(xs)
            ys = rand(Uniform(1.0, 100.0), n)
            zs = lowess(xs, ys, rand(Uniform(0.1, 1.0)), rand(3:10), rand(Uniform(0.0, 1.0)))
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
xs  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
ys  = [0.56, 0.15, 0.85, 0.5, 0.05, 0.23, 0.12, 0.94]
f   = 0.088
nsteps  = 1
delta   = 0.668
zs = [0.56, 0.6142857142857143, 0.6685714285714286, 0.7228571428571429, 0.7771428571428571, 0.8314285714285713, 0.8857142857142857, 0.94]
@test lowess(xs, ys, f, nsteps, delta) == zs
end

@testset "test 2" begin
xs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
ys = [0.62, 0.04, 0.72, 0.59, 0.35, 0.8, 0.78, 0.58]
f = 0.697
nsteps = 4
delta = 0.62
zs = [0.4490432634298429, 0.4791567593889915, 0.5092702553481401, 0.5393837513072886, 0.5694972472664371, 0.5996107432255857, 0.6297242391847342, 0.6598377351438828]
@test lowess(xs, ys, f, nsteps, delta) == zs
end

@testset "test 3" begin
xs =    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
ys =    [0.19, 1.0, 0.82, 0.12, 0.62, 0.18, 0.58, 0.06]
f = 0.822
nsteps = 4 
delta = 0.154
zs = [0.51794977949136, 0.47320890634488694, 0.4284680331984139, 0.38372716005194074, 0.33898628690546767, 0.2942454137589946, 0.24950454061252153, 0.20476366746604843]
@test lowess(xs, ys, f, nsteps, delta) == zs
end

@testset "test 4" begin
xs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
ys = [0.94, 0.89, 0.95, 0.84, 0.34, 0.66, 0.57]
f = 0.313
nsteps = 3
delta = 0.024
zs = [0.94, 0.945, 0.95, 0.645, 0.34, 0.45499999999999996, 0.57]
@test lowess(xs, ys, f, nsteps, delta) == zs
end

@testset "test 5" begin
xs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
ys = [0.87, 0.88, 0.47, 0.48, 0.15, 0.53, 0.83]
f = 0.397
nsteps =  1
delta = 0.164
zs = [0.87, 0.8633333333333333, 0.8566666666666667, 0.8499999999999999, 0.8433333333333333, 0.8366666666666666, 0.83]
@test lowess(xs, ys, f, nsteps, delta) == zs
end

end 
