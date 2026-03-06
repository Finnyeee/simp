# Run 2 million iterations, one run per symmetric 2-bin memory type,
# and save the full Q matrix at every iteration.
#
# Usage: julia --project=. run_2bin_qhistory_2m.jl
#
# Output: data_full_memory/qhistory_2m_symmetric_2bin_<k>.json for each memory type k
#         Each file has: "final_Q" (serialized), "Q_history" (array of 2e6 Q snapshots).

using JSON

include(pwd() * "/src/Memories.jl")
include(pwd() * "/src/Simplicity.jl")

using Distributions
using .Memories
using .Simplicity

const P = 2
const GAMMA = 0.95
const ALPHA = 0.05
const GRID = 15
const TAU = 0.02
const BETA = 0
const MU = 0.25
const OUTSIDE_OPTION = 0.0
const OPTIONS = ["synchronous"]

costs = 1.0
qualities = 2.0
action_space = [[1.43, 1.97]]
state_space = NaN

model = Model(P, GAMMA, ALPHA, costs, qualities, MU, OUTSIDE_OPTION, GRID, TAU, BETA, action_space, state_space, OPTIONS)

payoffs = Simplicity.payoffs

const ITERATIONS = 2_000_000
const TYPE = "symmetric_2bin"

function serialize_Q(Q)
    out = Dict{String, Dict{String, Vector{Float64}}}()
    for (player_id, state_dict) in Q
        out[string(player_id)] = Dict(join(state, "_") => copy(v) for (state, v) in state_dict)
    end
    return out
end

println("Loading config $TYPE...")
config_path = pwd() * "/simulations/configs/$TYPE.json"
isfile(config_path) || error("Config not found: $config_path")
temp = JSON.parse(JSON.parsefile(config_path))
memories = [Dict(parse(Int, string(k)) => [identity.(v[1]), identity.(v[2])] for (k, v) in pairs(temp[i])) for i in 1:length(temp)]

mkpath("data_full_memory")

for (f, memory) in enumerate(memories)
    println("Memory type $f / $(length(memories)): $memory")
    final_Q, Q_history = train_save_Q_history(model, ITERATIONS, egreedy, payoffs, memory)

    # Trim Q_history to last 500 iterations to keep files analyzable.
    # Training still runs for ITERATIONS steps; we just discard the early snapshots.
    history_len = length(Q_history)
    keep = min(500, history_len)
    Q_history_trimmed = Q_history[(history_len - keep + 1):history_len]

    out = Dict(
        "final_Q" => serialize_Q(final_Q),
        "Q_history" => Q_history_trimmed,
        "iterations" => ITERATIONS,
        "history_length" => keep,
        "memory" => memory
    )
    outpath = "data_full_memory/qhistory_2m_$(TYPE)_$f.json"
    println("Writing $outpath ...")
    open(outpath, "w") do file
        JSON.print(file, JSON.json(out))
    end
    println("Done memory $f.")
end

println("All done.")
