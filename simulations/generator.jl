using ProgressMeter
using DelimitedFiles
using JSON

include(pwd()*"/src/Memories.jl")
include(pwd()*"/src/Simplicity.jl")


using Distributions
using .Memories
using .Simplicity

const P::Int64 = 2
const GAMMA::Float64 = 0.95
const ALPHA::Float64 = 0.05
const GRID::Int64 = 15
const TAU::Float64 = 0.8
const BETA::Float64 = 2e-5
const MU::Float64 = 0.25
const OUTSIDE_OPTION::Float64 = 0.0



costs = 1.0
qualities = 2.0
action_space = [[1.43, 1.97]]
state_space = NaN # Left uninitialized, gets initialized inside train()

model = Model(P,GAMMA,ALPHA,costs,qualities,MU,OUTSIDE_OPTION,GRID,TAU,BETA,action_space,state_space)

const TYPE::String = ARGS[1]
const K::Int64 = parse(Int64,ARGS[2])
const ITERATIONS::Int64 = parse(Int64,ARGS[3]) #1e6

const RESULTS = Array{Dict{Int64,Matrix{Float64}}}(undef,K)
@showprogress for i in 1:K
    outcome = train(model, ITERATIONS, boltzmann, payoffs, TYPE)
    RESULTS[i] = outcome
end

open("data_$TYPE.json", "w") do f
    JSON.print(f, JSON.json(RESULTS))
end
