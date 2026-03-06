#!/usr/bin/env julia
# One-off script: trim existing qhistory_2m_*.json files to last 5000 iterations.
# Saves trimmed output as <basename>_trimmed.json in the same directory.
#
# Usage: julia --project=. scripts/trim_qhistory_to_last_500.jl [data_dir]
#        data_dir defaults to data_full_memory/
#
# Note: Each file is loaded fully into memory (~4.5 GB per file). Run one at a time
# if RAM is limited; you can invoke the script once per file if needed.

using JSON

const KEEP = 5000

function main()
    data_dir = length(ARGS) >= 1 ? ARGS[1] : "data_full_memory"
    data_dir = abspath(data_dir)
    isdir(data_dir) || error("Directory not found: $data_dir")

    pattern = "qhistory_2m_"
    files = filter(readdir(data_dir; join = true)) do p
        isfile(p) && occursin(pattern, basename(p)) && !occursin("_trimmed.", p)
    end

    if isempty(files)
        println("No qhistory_2m_*.json files found in $data_dir")
        return
    end

    println("Found $(length(files)) file(s) to trim. Keeping last $KEEP iterations.")
    for path in files
        trim_one(path, KEEP)
    end
    println("Done.")
end

function trim_one(path::String, keep::Int)
    base = splitext(basename(path))[1]
    out_path = joinpath(dirname(path), base * "_trimmed.json")
    println("Reading $path ...")
    data = JSON.parsefile(path)
    # Handle double-encoded JSON (file contains a JSON string of the object)
    if data isa String
        data = JSON.parse(data)
    end
    Q_history = data["Q_history"]
    n = length(Q_history)
    k = min(keep, n)
    from = n - k + 1
    data["Q_history"] = Q_history[from:n]
    data["history_length"] = k
    println("  Writing $out_path (last $k iterations) ...")
    open(out_path, "w") do io
        JSON.print(io, data)
    end
    println("  Done.")
end

main()
