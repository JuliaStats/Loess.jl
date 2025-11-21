using BenchmarkTools, Loess, Random

const SUITE = BenchmarkGroup()

SUITE["random"] = BenchmarkGroup()

for i in 2:4
    n = 10^i
    x = rand(MersenneTwister(42), n)
    y = sqrt.(x)
    SUITE["random"][string(n)] = @benchmarkable loess($x, $y)
end

SUITE["ties"] = BenchmarkGroup()
let
    x = repeat([π/4*i for i in -20:20], inner=101)
    y = sin.(x)
    SUITE["ties"]["sine"] = @benchmarkable loess($x, $y; span=0.2)
end
