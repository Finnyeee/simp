module Memories

export constructor
export full_monitoring
export competitor_only
export incremental_merge_from_bottom
export flatten_states
export unflatten_states

"""
    full_monitoring(last_prices, identity)

Basic memory type, returns the entire past period vector of prices.
  *  last_prices - vector of prices in the past period
  *  identity - identity of the firm currently considered
"""
function full_monitoring(last_prices, identity)
    return vcat(last_prices,identity)
end

"""
    competitor_only(last_prices, identity)

Memory type. Returns only the price of the competitor in the last period.
  *  last_prices - vector of prices in the past period
  *  identity - identity of the firm currently considered
"""
function competitor_only(last_prices, identity)
    if !(identity in 1:2)
        throw(DimensionMismatch("Too many players! Only implemented for p = 2."))
    end
    
    #Trick: swapping identities this way only works with 2 firms.
    return [last_prices[3-identity], identity]    
end

"""
    incremental_merge_from_bottom(last_prices, identity, k)

Memory type. Returns only the price of the competitor in the last period, but returns 
a single state if the competitor's price is one of the bottom k prices.
  *  last_prices - vector of prices in the past period
  *  identity - identity of the firm currently considered
  *  k - the number of states to merge from the bottom
"""
function incremental_merge_from_bottom(last_prices,identity,k)
    if !(identity in 1:2)
        throw(ValueError("Too many players! Only implemented for p = 2."))
    end
    #Trick: swapping identities this way only works with 2 firms.
    if last_prices[3-identity] in 1:k+1
        return [1, identity]
    else
        return [last_prices[3-identity],identity]
    end
end

"""
    constructor(model, type)

Constructor for dictionary mapping histories to state spaces.
  *  model - Model instance. Not typed because of inclusion order.
  *  type - a string serving as switch between memory types
"""
#TODO(b/1): implement logic for p =/= 2
function constructor(model, type::String)
    dict = Dict()
    bottom_filter = match(r"^incremental_merge_from_bottom_(\d+)$", type)
    if type == "full_monitoring"
        for (i,j,identity) in Iterators.product(1:model.grid, 1:model.grid, 1:model.p)
            dict[([i,j],identity)] = full_monitoring([i,j],identity)
        end
    elseif type == "competitor_only"
        for (i,j,identity) in Iterators.product(1:model.grid, 1:model.grid, 1:model.p)
            dict[([i,j],identity)] = competitor_only([i,j],identity)
        end
    elseif bottom_filter !== nothing
        for (i,j,identity) in Iterators.product(1:model.grid, 1:model.grid, 1:model.p)
            dict[([i,j],identity)] = incremental_merge_from_bottom([i,j],identity,parse(Int,bottom_filter.captures[1]))
        end 
    else 
        throw(ArgumentError("Memory type $(type) is not implemented"))
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
    constructor(model, type)

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