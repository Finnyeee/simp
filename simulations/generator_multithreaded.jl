using Distributed

function include_everywhere(filepath)
    include(filepath) # Load on Node 1 first, triggering any precompile
    if nprocs() > 1
        fullpath = joinpath(@__DIR__, filepath)
        @sync for p in workers()
            @async remotecall_wait(include, p, fullpath)
        end
    end
end

using ProgressMeter
using DelimitedFiles
using JSON

const THREADS = parse(Int,ARGS[1])

addprocs(THREADS,exeflags="--project=simplicity")

@everywhere include(pwd()*"/src/Memories.jl")
@everywhere include(pwd()*"/src/Simplicity.jl")

@everywhere begin
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

    const ITERATIONS::Int64 = 1e6

    costs = 1.0
    qualities = 2.0
    action_space = [[1.43, 1.97]]
    state_space = NaN # Left uninitialized, gets initialized inside train()

    model = Model(P,GAMMA,ALPHA,costs,qualities,MU,OUTSIDE_OPTION,GRID,TAU,BETA,action_space,state_space)

end

const TYPE::String = ARGS[2]
const K::Int64 = parse(Int64,ARGS[3])

const TYPE_ARRAY::Array{String} = [TYPE for _ in 1:THREADS]

const RESULTS = Array{Array{Dict{Int64,Matrix{Float64}}}}(undef,K)
@showprogress for i in 1:K
    outcomes = pmap(type -> train(model, ITERATIONS, boltzmann, payoffs, type), TYPE_ARRAY)
    RESULTS[i] = outcomes
end

open("data_$TYPE.json", "w") do f
    JSON.print(f, JSON.json(vcat(RESULTS)))
end
