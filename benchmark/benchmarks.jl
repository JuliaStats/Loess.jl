using BenchmarkTools, Loess, Random

const SUITE = BenchmarkGroup()

SUITE["random"] = BenchmarkGroup()

SUITE["random"] = begin
    for i in 2:3
        n = 10^i
        x = rand(MersenneTwister(42), n)
        y = sqrt.(x)
        @benchmarkable loess($x, $y)
    end
end
