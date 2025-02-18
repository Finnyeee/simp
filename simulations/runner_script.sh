for k in $(seq 1 14); do
    memory_type="incremental_merge_from_bottom_${k}"
    julia --project=@. ~/simplicity/simulations/generator_decay_sync.jl $memory_type 10 1e7
done