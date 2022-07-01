@testset "Lowess Bounds Test" begin
    xs = 10 .* rand(100)
    ys = sin.(xs) .+ 0.5 * rand(100)
    zs = lowess(xs, ys)
    @test minimum(zs) >= -1.1
    @test maximum(zs) <= 1.1
end

@testset "Comparing against the results of the original C code using random floating point inputs." begin
    @testset "test 1" begin
        xs  = [0.03, 0.16, 0.37, 0.58, 0.71, 0.85, 0.92, 0.99]
        ys  = [0.21, 0.62, 0.05, 0.92, 0.81, 0.5, 0.65, 0.23]
        f   = 0.645
        nsteps  = 4
        delta   = 0.506
        zs = [0.33975588875617213, 0.35757371725482023, 0.3863563632910978, 0.4728384416628679, 0.5263749663692019, 0.5840296852837153, 0.45763988305199876, 0.3312500808202824] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 2" begin
        xs  = [0.34, 0.37, 0.38, 0.41, 0.51, 0.77, 0.96]
        ys  = [0.15, 0.26, 0.45, 0.94, 0.92, 0.75, 0.27]
        f   = 0.243
        nsteps  = 2
        delta   = 0.728
        zs = [0.15, 0.1558064516129032, 0.15774193548387097, 0.1635483870967742, 0.18290322580645163, 0.23322580645161292, 0.27] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 3" begin
        xs  = [0.17, 0.19, 0.34, 0.37, 0.46, 0.51, 0.66]
        ys  = [0.12, 0.19, 0.76, 0.89, 0.81, 0.73, 0.41]
        f   = 0.058
        nsteps  = 4
        delta   = 0.539
        zs = [0.12, 0.13183673469387755, 0.2206122448979592, 0.23836734693877548, 0.2916326530612245, 0.32122448979591833, 0.41] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 4" begin
        xs  = [0.26, 0.39, 0.48, 0.51, 0.59, 0.94]
        ys  = [0.85, 0.24, 0.38, 0.97, 0.79, 0.23, 0.89]
        f   = 0.056
        nsteps  = 5
        delta   = 0.516
        zs = [0.85, 0.8263636363636363, 0.81, 0.8045454545454546, 0.79, 0.23] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 5" begin
        xs  = [0.06, 0.32, 0.46, 0.78, 0.94, 0.99]
        ys  = [0.26, 0.91, 0.14, 0.33, 0.57, 0.99, 0.56]
        f   = 0.952
        nsteps  = 1
        delta   = 0.427
        zs = [0.4003188954738399, 0.4282100726879575, 0.44322839888017473, 0.4404274424163527, 0.7243844287810028, 0.8131209870199563] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 6" begin
        xs  =  [-0.99, -0.92, -0.85, -0.71, -0.58, -0.37, -0.16, -0.03]
        ys  = [-0.51, -0.37, -0.34, -0.15, -0.26, -0.45, -0.94, -0.92]
        f   = 0.749
        nsteps  = 2
        delta   = 0.242
        zs = [-0.49206113604574986, -0.40624347814925665, -0.32042582025276317, -0.24428989075081292, -0.2748719280832276, -0.5186919879597951, -0.8073481516180832, -0.9863029571533182]
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end
end

@testset "Comparing against the results of the original C code using random integer inputs." begin
    @testset "test 1" begin
        xs  = [3, 16, 37, 58, 71, 85, 92, 99]
        ys  = [34, 17, 19, 46, 37, 12, 19, 76]
        f   = 0.888
        nsteps  = 5
        delta   = 0.722
        zs = [24.16160454268896, 25.68812005828106, 28.65470533137912, 32.41594186202208, 33.21325359358515, 36.66797225555004, 38.219534918062166, 39.83448459003582] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 2" begin
        xs  = [5, 6, 39, 51, 54, 77, 94]
        ys  = [56, 96, 7, 43, 50, 6, 18]
        f = 0.457
        nsteps = 1
        delta = 0.211
        zs = [55.99999999999997, 95.99999999999997, 6.999999999999997, 43.000000000000085, 50.000000000000135, 5.999999999999999, 18.00000000000001] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 3" begin
        xs  = [8, 14, 34, 53, 56, 60, 69, 74]
        ys  = [55, 25, 61, 5, 60, 94, 75, 69]
        f   = 0.741
        nsteps  = 1
        delta   = 0.845
        zs = [41.2806572273703, 44.10805712622063, 49.479808643597146, 20.261120010393306, 46.05149712706894, 66.5659090517141, 75.37151361313856, 69.17534870215226] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 4" begin
        xs  = [12, 17, 19, 30, 33, 57, 60, 87]
        ys  = [1, 95, 21, 16, 94, 57, 82, 18]
        f   = 0.565
        nsteps  = 4
        delta   = 0.379
        zs = [10.685926120690844, 39.53280233940788, 48.05950850869127, 42.4207247256067, 77.96343040038715, 68.67685655483697, 82.00000000000007, 18.97936991085022] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 5" begin
        xs  = [36, 45, 46, 48, 57, 58, 93, 99]
        ys  = [40, 85, 30, 31, 52, 40, 79, 30]
        f   = 0.937
        nsteps  = 5
        delta   = 1.0
        zs = [44.354647553697596, 44.695277023911935, 44.73964212703127, 44.82591036430719, 44.68923172948601, 44.545298404683564, 52.24471897990482, 51.77469610389949] # Attained using ccall
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end

    @testset "test 6" begin
        xs  = [-99, -92, -85, -71, -58, -37, -16, -3]
        ys  = [-51, -37, -34, -15, -26, -45, -94, -92]
        f   = 0.749
        nsteps  = 2
        delta   = 0.242
        zs = [-49.20632754527827, -40.64178192504818, -32.04272018995434, -24.42894165995633, -27.48719208900706, -51.869198795979536, -80.73481516180829, -98.63029571533184]
        @test lowess(xs, ys, f, nsteps, delta) == zs
    end
end
