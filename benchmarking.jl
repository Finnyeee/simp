using BenchmarkTools
using NNlib
using Distributions

BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000


function loopy_draws(p)
    return [rand(Multinomial(5,softmax(p))) for _ in 1:4]
end

function vectorized_draws(p)
    d = [p for _ in 1:4]
    return rand.(Multinomial.(5,softmax.(d)))
end


@benchmark loopy_draws(p) setup=(p=[0.2,0.25,0.3,0.15,0.1])


# @benchmark vectorized_draws(p) setup=(p=[0.2,0.25,0.3,0.15,0.1])