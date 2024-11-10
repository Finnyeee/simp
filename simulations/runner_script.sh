for k in $(seq 7 14); do
    memory_type="incremental_merge_from_top_${k}"
    julia ~/simplicity/simulations/generator_decay.jl $memory_type 10 1e7
done