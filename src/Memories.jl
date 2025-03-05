module Memories

#export __pooling
#export __full_monitoring
export constructor
export flatten_states
export unflatten_states

"""
    __pooling(action, thresholds)
This function pools the actions of the players according to the thresholds specified in the model.
See the description of the function __full_monitoring for more details.
It is a helper function for __full_monitoring.
It is a private function, and should not be called directly.
"""
function __pooling(action, thresholds::AbstractVector)
    if thresholds == []
        return action
    end
    bin = 1
    for threshold in thresholds
        if action > threshold
            bin += 1
        else
            break
        end
    end
    return bin
end

"""
    __full_monitoring(last_prices, identity; thresholds)
This function controls the pooling of past prices for the players. 
It is the highest level of generality, and includes all other memory forms in the library.
It requires a dict of thresholds for each player, which specifies for each player where pooling for a state ends. 
For example, if thresholds[1] = [[1,4,6],[2]], then player 1 knows that:
- its last price was either 1, between 2 and 4, between 4 and 6, or above 6, and 
- its opponent's last price was either between 1 and 2, or above 3.
This is a private method, and should not be called directly.
"""
function __full_monitoring(last_prices::AbstractVector, identity::Int64; thresholds = nothing)
    if thresholds[identity] == nothing || thresholds[identity] ==[]
        return vcat(last_prices, identity)
    else
        thresholds_identity = thresholds[identity]
        pooled_prices = [__pooling(price, thresholds_identity[i]) for (i,price) in enumerate(last_prices)]
        return vcat(pooled_prices, identity)
    end
end

"""
    constructor(mode, monitoring)
This function creates a dictionary of monitoring functions for each player in the model.

"""
function constructor(model, monitoring)
    dict = Dict()
    for (i,j,identity) in Iterators.product(1:model.grid, 1:model.grid, 1:model.p)
        dict[([i,j],identity)] = __full_monitoring([i,j],identity; thresholds=monitoring)
    end
    return dict
end

"""
    flatten_states(model, state)

Turns states into indices
  *  model - Model instance. Not typed because of inclusion order.
  *  state - an array of past prices
"""
#TODO(b/1): implement logic for p =/= 2
function flatten_states(model, state)
    if length(state) == 2
        return state[1] + (state[2]-1)*model.grid
    else
        return state[1] 
    end
end

"""
    unflatted_states(model, index)

Turns indices into states
  *  model - Model instance. Not typed because of inclusion order.
  *  index - an index corresponding to past prices
"""
#TODO(b/1): implement logic for p =/= 2
#TODO(b/3): fix unflattening map

function unflatten_states(model, index)
    return [index % model.grid, div(index,model.grid)]
end

end 