using ProgressMeter
using DelimitedFiles
using JSON

include(pwd()*"/src/Memories.jl")
include(pwd()*"/src/Simplicity.jl")


using Distributions
using .Memories
using .Simplicity

const P = 2
const GAMMA = 0.95
const ALPHA = 0.05
const GRID = 15
const TAU = 0.025
const BETA = 0 # egreedy with no decay
const MU = 0.25
const OUTSIDE_OPTION = 0.0
const OPTIONS = []



costs = 1.0
qualities = 2.0
action_space = [[1.43, 1.97]]
state_space = NaN # Left uninitialized, gets initialized inside train()

model = Model(P,GAMMA,ALPHA,costs,qualities,MU,OUTSIDE_OPTION,GRID,TAU,BETA,action_space,state_space,OPTIONS)

const TYPE = ARGS[1]
const K = parse(Int64,ARGS[2])
const ITERATIONS = Int(parse(Float64,ARGS[3])) #1e6

temp = JSON.parse(JSON.parsefile(pwd()*"/simulations/configs/$TYPE.json"))
memories=  [Dict(parse(Int,string(k))=>[identity.(v[1]), identity.(v[2])] for (k,v) in pairs(temp[i])) for i in 1:length(temp)]

for (f,memory) in enumerate(memories)
    print("Memory is ", memory, "\n")
    RESULTS = Array{Dict{Int64, Dict{Vector{Int64}, Vector{Float64}}}}(undef,K)
    @showprogress for i in 1:K
        outcome = train(model, ITERATIONS, egreedy, payoffs, memory)
        RESULTS[i] = outcome
    end

    open("data_full_memory_long/data_async_$TYPE" * "_$f.json", "w") do f
        JSON.print(f, JSON.json(RESULTS))
    end
end