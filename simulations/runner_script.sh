for k in $(seq 1 14); do
    memory_type="threshold_fixed_${k}"
    julia ~/simplicity/simulations/generator_decay.jl $memory_type 10 1e7
done