using BenchmarkTools, Loess, Random

const SUITE = BenchmarkGroup()

SUITE["random"] = BenchmarkGroup()

for i in 2:3
    n = 10^i
    x = rand(MersenneTwister(42), n)
    y = sqrt.(x)
    SUITE["random"][string(n)] = @benchmarkable loess($x, $y)
end
