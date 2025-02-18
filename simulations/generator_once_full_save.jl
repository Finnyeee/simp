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
const TAU::Float64 = 0.25
const BETA::Float64 = 0.000001
const MU::Float64 = 0.25
const OUTSIDE_OPTION::Float64 = 0.0



costs = 1.0
qualities = 2.0
action_space = [[1.43, 1.97]]
state_space = NaN # Left uninitialized, gets initialized inside train()

model = Model(P,GAMMA,ALPHA,costs,qualities,MU,OUTSIDE_OPTION,GRID,TAU,BETA,action_space,state_space)

const TYPE::String = ARGS[1]
const K::Int64 = parse(Int64,ARGS[2])
const ITERATIONS::Int64 = Int(parse(Float64,ARGS[3])) #1e6


outcome, prices = train_once_save_all(model, ITERATIONS, egreedy, payoffs, TYPE, costs)

open("data_all_$TYPE.json", "w") do f
    JSON.print(f, JSON.json(outcome))
end

open("data_all_price_$TYPE.json", "w") do f
    JSON.print(f, JSON.json(prices))
end