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

    const P = 2
    const GAMMA = 0.95
    const ALPHA = 0.05
    const GRID = 15
    const TAU = 0.02  # Fixed to match sync
    const BETA = 0    # Fixed to match sync (egreedy with no decay)
    const MU = 0.25
    const OUTSIDE_OPTION = 0.0
    const OPTIONS = ["synchronous"]  # Fixed to match sync

    costs = 1.0
    qualities = 2.0
    action_space = [[1.43, 1.97]]
    state_space = NaN # Left uninitialized, gets initialized inside train()

    model = Model(P,GAMMA,ALPHA,costs,qualities,MU,OUTSIDE_OPTION,GRID,TAU,BETA,action_space,state_space,OPTIONS)

end

const K = parse(Int64,ARGS[2])
const ITERATIONS = Int(parse(Float64,ARGS[3])) #1e6

# Define payoffs function (assuming it's defined in Simplicity module)
@everywhere payoffs = Simplicity.payoffs

# Process all batches from 01 to 16
for batch_num in 1:16
    TYPE = "batch_$(lpad(batch_num, 2, "0"))"
    println("Processing $TYPE...")
    
    # Parse the configuration file
    temp = JSON.parse(JSON.parsefile(pwd()*"/simulations/configs/$TYPE.json"))
    memories = [Dict(parse(Int,string(k))=>[identity.(v[1]), identity.(v[2])] for (k,v) in pairs(temp[i])) for i in 1:length(temp)]

    # Process each memory configuration
    for (f,memory) in enumerate(memories)
        print("Memory is ", memory, "\n")
        
        # Create array of memory configs for parallel processing
        MEMORY_ARRAY = [memory for _ in 1:THREADS]
        
        RESULTS = Array{Dict{Int64, Dict{Vector{Int64}, Vector{Float64}}}}(undef,K)
        @showprogress for i in 1:K
            outcomes = pmap(mem -> train(model, ITERATIONS, egreedy, payoffs, mem), MEMORY_ARRAY)
            # Take the first result since all workers process the same memory config
            RESULTS[i] = outcomes[1]
        end

        # Create output directory if it doesn't exist
        mkpath("data_full_memory")
        
        open("data_full_memory/data_multithreaded_$TYPE" * "_$f.json", "w") do file
            JSON.print(file, JSON.json(RESULTS))
        end
    end
end
