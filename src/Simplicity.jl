module Simplicity

using Random
using Distributions
using StatsBase
using LinearAlgebra
using NNlib
using Main.Memories
#using ProgressMeter

export Simulation
export Model
export boltzmann
export egreedy
export payoffs
export secondprice
export train
export price_grid
export train_once_save_all

"""
    simulation(B, P, Q)

Base structure for simulation results
  *  B - nxp array of bids
  *  P - nx1 array of prices
  *  Q - nxpxgrid array of Q matrices
"""
struct Simulation  
    P 
    Q
end


"""
    model(gamma,alpha,value,p,grid,no_bid,exp,beta,reserve,format,top)

Base structure for an auction simulation
  *  p - number of players
  *  gamma - discount factor
  *  alpha - learning rate, constant
  *  costs - vector of production costs (px1). If it is a number, same cost across firms
  *  grid - the number of prices for each firm
  *  tau - the temperature parameter 
  *  beta - the experimentation decay parameter 
            - baseline 2e-4
            - if ==0 -> continuous exploration
  *  action_space - a vector of pairs (bottom, top) representing the highest 
                    and lowest prices in the action space. If a single pair, same
                    across all firms
"""
mutable struct Model
    p::Int64
    gamma::Float64
    alpha::Float64
    costs
    qualities
    substitution::Float64
    outside_option::Float64
    grid::Int64
    tau::Float64
    beta::Float64
    action_space
    state_space
    options::Vector{String}
end

Model(p,gamma,alpha,costs,qualities,substitution,outside_option,grid,tau,beta,action_space) = 
        Model(p,gamma,alpha,costs,qualities,substitution,outside_option,grid,tau,beta,action_space,[]) # initialize model without options

        # Model(p,gamma,alpha,costs,grid,tau,beta) = 
#         Model(p,gamma,alpha,costs,grid,tau,beta,(costs,5*costs) #initialize the value of action_space


"""
    boltzmann(model, Q, states, tau, j)::Vector{Int64}

Basic softmax/boltzmann action choice
  *  model - instance of Model
  *  Q - current Q-functions
  *  states - state of the world
  *  tau - boltzmann stage temperature 
  *  j - identity of player
"""
function boltzmann(model::Model, Q, states, tau::Float64, j::Int64)
    return [sample(1:model.grid, Weights(softmax(tau.*Q[i][states[i]]))) for i in 1:model.p]
end

"""
    eGreedy(model, Q, states, epsilon, j)::Vector{Int64}

Basic e-greedy action choice
  *  model - instance of Model
  *  Q - current Q-functions
  *  states - state of the world
  *  epsilon - experimentation probability 
  *  j - identity of player
"""
function egreedy(model::Model, Q, states , epsilon::Float64, j::Int64)
    samples = []
    for i in 1:model.p
        best_action = argmax(Q[i][states[i]])
        base_weight = fill(1/(epsilon*model.grid), model.grid)
        base_weight[best_action] += 1-(1/epsilon)
        push!(samples,sample(1:model.grid,ProbabilityWeights(base_weight)))
    end
    return samples
end


"""
    payoffs(model, prices)::Vector{Float64}

Bertrand profits within round
  *  model - model structure
  *  prices - prices chosen
"""
function payoffs(model, prices)
    return (prices .- model.costs).*softmax((model.qualities .- prices)./model.substitution)

end


"""
    Qupdate(model,prices,outcomes,states,i)

Function responsible for the update of the Q matrix
  *  model - model structure
  *  prices - prices chosen (px1)
  *  outcomes - outcome from the stage game (px1)
  *  states - the state used to condition the strategy (px1)-tuples
  *  i - identity of the agent being updated
"""
function Qupdate(model::Model, Q, prices, outcomes, states, states_next, i::Int64)
    Q[i][states[i]][prices[i]] += model.alpha * (outcomes[i] + model.gamma*maximum(Q[i][states_next[i]]) - Q[i][states[i]][prices[i]]) 
    return Q[i]
end

"""
    Qupdate_synchronous(model,prices,payoff,states,i)

Function responsible for the update of the Q matrix
  *  model - model structure
  *  prices - prices chosen (px1)
  *  payoff - function to compute payoffs
  *  states - the state used to condition the strategy (px1)-tuples
  *  i - identity of the agent being updated   
"""
function Qupdate_synchronous(model::Model, Q, prices, payoff, price_grid, states, states_next, i::Int64)

    for j in 1:model.grid
        prices_vec = deepcopy([price_grid[k][prices[k]] for k in 1:model.p])
        prices_vec[i] = price_grid[i][j]
        outcomes = payoff(model, prices_vec)
        Q[i][states[i]][j] += model.alpha * (outcomes[i] + model.gamma*maximum(Q[i][states_next[i]]) - Q[i][states[i]][j]) 
    end

    return Q[i]
end



"""
    random_initializer(matrix)

Simple randomizer for initialization
  *  matrix - two-dimensional array
"""
function random_initializer(vector::Vector{Float64})
    return vector .+ (rand(size(vector)) .* 10 .- 5) 
end

"""
    initialize(model::Model, type::String)

Function responsible for the update of the Q matrix
  *  model - model structure
  *  monitoring - dictionary. For more details, see train.
"""
function initialize(model::Model, monitoring)

    monitoring_technology = constructor(model, monitoring)

    #Preprocessing 
    if length(model.costs) == 1
        model.costs = ones(model.p)*Float64(model.costs)
    elseif !(length(model.costs) in [1, model.p])
        throw(ArgumentError("Costs $(model.costs) should be of size $(model.p) or $(1)"))
    end
 
    if length(model.qualities) == 1
        model.qualities = ones(model.p)*Float64(model.qualities)
    elseif !(length(model.qualities) in [1, model.p])
        throw(ArgumentError("Qualities $(model.qualities) should be of size $(model.p) or $(1)"))
    end

    if length(model.action_space) == 1
        model.action_space = [model.action_space[1] for _ in 1:model.p]
    elseif !(length(model.qualities) in [1, model.p])
        throw(ArgumentError("Action space $(model.action_space) should be of size $(model.p) or $(1)"))
    end

    optimism = [2*(model.action_space[i][end] - model.costs[i]) for i in 1:model.p]

    """
    This simplifies the representation of the Q matrix. It allows us to look only at the states
    effectively used by the players. 
    """
    relevant_keys = [[] for _ in 1:model.p]
    for (key, value) in monitoring_technology
        if !(value[1:2] in relevant_keys[key[2]])
            push!(relevant_keys[key[2]], value[1:2])
        end
    end

    Q = Dict(i => Dict(relevant_keys[i] .=> [random_initializer(
        fill(optimism[i], model.grid)) for _ in relevant_keys[i]]) for i in 1:model.p)

    return [Q, rand(1:model.grid, model.p), monitoring_technology]

    #return 
    #[random_initializer(fill(optimism[i], length(state_space), model.grid)) for i in 1:model.p], rand(1:model.grid, model.p) # Initialized Q matrix and states 
end

"""
    _price_grid(model::Model)

Generates the price_grid array used to map indices into prices.
  *  model - model structure
"""
function _price_grid(model::Model)
    price_grid_array = [zeros(model.grid) for _ in 1:model.p]
    for i in 1:model.p
        increment = (model.action_space[i][2] - model.action_space[i][1])/(model.grid-1)
        price_grid_array[i] = [model.action_space[i][1] + increment*(j-1) for j in 1:model.grid]
    end
    return price_grid_array
end


"""
    train(model,n,policy,payoffs,monitoring)

Simulate an auction according to format
  *  model - model structure
  *  n - number of iterations
  *  policy - function for action selection
         use eGreedy for e-greedy
         use eGreedy for greedy
         use Pushdown for downward force
  *  payoffs - function for payoff computation
  *  monitoring - dictionary for the construction of monitoring technology. 
                  Suggested config: save a json dictionary of arrays where 
                  each key is a player and each value is an array of threshold arrays
                  for each player. For example, player 1 could have thresholds [[1,4,6],[2]]. 
"""
function train(model::Model,n::Int64,policy,payoffs,monitoring)
    
    # Initializations
    Q, states_raw, monitoring_technology = initialize(model, monitoring)
    states_next = nothing

    price_grid = _price_grid(model)

    #P = zeros(n,model.p)

    # TODO(b/1) generalize logic to p =/= 2
    #Q_history = [zeros(n,length(Q[1][:,1]),model.grid) for _ in 1:model.p]
        
    if model.beta==0
        indicator_beta = 0
    else
        indicator_beta = 1
    end
    
    #Training
    for i=1:n

        # Update entropy of the policy 
        temperature = 1/(model.tau .*((1-indicator_beta) + Base.exp(-model.beta*i)*(indicator_beta)))
        
        # TODO(b/2): check type is vector of vectors
        if states_next == nothing
            states = [monitoring_technology[(states_raw, identity)][1:end-1] for identity in 1:model.p]
        else
            states = deepcopy(states_next)
        end

        # Random Experimentation 
        prices_index = policy(model, Q, states, temperature, i) #returns vector of dimension p (2)
        
        prices = [price_grid[i][prices_index[i]] for i in 1:model.p]

        states_raw = prices_index
        states_next = [monitoring_technology[(states_raw, identity)][1:end-1] for identity in 1:model.p]

        payoff = payoffs(model, prices)
        # Update the Q-values
        for j=1:model.p
            if model.options == []
                Q[j] = Qupdate(model, Q, prices_index, payoff, states, states_next, j)
            elseif model.options[1] == "synchronous"
                Q[j] = Qupdate_synchronous(model, Q, prices_index, payoffs, price_grid, states, states_next, j)
            else
                throw(ArgumentError("Option $(model.options[1]) is not implemented"))
            end
            #   println("states: ", states, "\nstates_flat: ", states_flat, "\nstates_next_flat: ", states_next_flat, "\nprice_index:", prices_index)
            #Q[j] = Qupdate(model, Q, prices_index, payoff, states_flat, states_next_flat, j)
            #Q_history[j][i,:,:] = Q[j]
        end

    end
    #display("text/plain",Q[1])
    # return Simulation(P, Q_history) -- commented out to save RAM
    return Dict([(i,Q[i]) for i in 1:model.p])
end

"""
    train_once_save_all(model,n,policy,payoffs)

Simulate an auction according to format
  *  model - model structure
  *  n - number of iterations
  *  policy - function for action selection
         use eGreedy for e-greedy
         use eGreedy for greedy
         use Pushdown for downward force
  *  payoffs - function for payoff computation
"""
function train_once_save_all(model::Model,n::Int64,policy,payoffs,type,costs)
    
    # Initializations
    Q, states = initialize(model, type)

    monitoring_technology = constructor(model, type)

    price_grid = _price_grid(model)

    prices_history = zeros(Float64, n, model.p) 
    #P = zeros(n,model.p)

    # TODO(b/1) generalize logic to p =/= 2
    #Q_history = [zeros(n,length(Q[1][:,1]),model.grid) for _ in 1:model.p]
        
    if model.beta==0
        indicator_beta = 0
    else
        indicator_beta = 1
    end
    
    #Training
    for i=1:n

        # Update entropy of the policy 
        temperature = 1/(model.tau .*((1-indicator_beta) + Base.exp(-model.beta*i)*(indicator_beta)))
        
        # TODO(b/2): check type is vector of vectors
        states_flat = [flatten_states(model, monitoring_technology[(states,identity)][1:end-1]) for identity in 1:model.p] 


        # Random Experimentation 
        prices_index = policy(model, Q, states_flat, temperature, i) #returns vector of dimension p (2)
        
        prices = [price_grid[i][prices_index[i]] for i in 1:model.p]

        prices_history[i,:] = prices

        #P[i,:] = prices
        states = prices_index
        states_next_flat = [flatten_states(model, monitoring_technology[(prices_index,identity)][1:end-1]) for identity in 1:model.p]

        payoff = payoffs(model, prices)
        
        # Update the Q-values
        for j=1:model.p
            #   println("states: ", states, "\nstates_flat: ", states_flat, "\nstates_next_flat: ", states_next_flat, "\nprice_index:", prices_index)
            Q[j] = Qupdate(model, Q, prices_index, payoff, states_flat, states_next_flat, j)
            #Q_history[j][i,:,:] = Q[j]
        end
    end
    # return Simulation(P, Q_history) -- commented out to save RAM
    return Dict([(i,Q[i]) for i in 1:model.p]), prices_history
end


end