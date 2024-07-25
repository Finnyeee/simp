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
export Qupdate
export initialize
export train


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
end

# Model(p,gamma,alpha,costs,grid,tau,beta) = 
#         Model(p,gamma,alpha,costs,grid,tau,beta,(costs,5*costs) #initialize the value of action_space


"""
    boltzmann(model, tau, j)::Vector{Int64}

Basic e-greedy action choice
  *  model - instance of Model
  *  states - state of the world
  *  tau - boltzmann stage temperature 
  *  j - identity of player
"""
function boltzmann(model::Model, Q, states, tau::Float64, i::Int64)
    return [sample(1:model.grid, Weights(softmax(tau.*Q[i][states[i],:]))) for i in 1:model.p]
end

"""
    eGreedy(Q,m,eprobn_,j,n)

Basic e-greedy action choice
  *  Q - current Q-functions
  *  m - instance of Model
  *  eprobn_ - experimentation parameter 1-eprob in stage i 
  *  j - current iteration
  *  n - total iterations
"""
function egreedy(Q,m::Model,eprobn_::Float64,j,n)
    b = Array{Int8}(undef,m.p)
    for i in 1:m.p
        epsilon = rand(Uniform(0,1))
        if epsilon>eprobn_
            b[i] = rand(1:m.grid+m.no_bid)
        else
            b[i]=sample(findall(Q[:,i].==maximum(Q[:,i])))
        end
    end
    return b
end


"""
    payoffs(model, prices)::Vector{Float64}

Profit calculator within round
  *  model - model structure
  *  prices - prices chosen
"""
function payoffs(model, prices)
    return softmax((model.qualities .- prices)./model.substitution)

end;

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
    Q[i][states[i],prices[i]] += model.alpha * (outcomes[i] + model.gamma*maximum(Q[i][states_next[i],:]) - Q[i][states[i],prices[i]]) 
    return Q[i]
end


"""
    initialize(model::Model, type::String)

Function responsible for the update of the Q matrix
  *  model - model structure
  *  type - string that controls memory types
"""
function initialize(model::Model, type::String)
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
        model.action_space = [model.action_space for _ in 1:model.p]
    elseif !(length(model.qualities) in [1, model.p])
        throw(ArgumentError("Action space $(model.action_space) should be of size $(model.p) or $(1)"))
    end

    optimism = [model.action_space[i][end] - model.costs[i] for i in 1:model.p]

    #TODO(b/1) generalize logic to p =/= 2
    bottom_filter = match(r"^incremental_merge_from_bottom_(\d+)$", type)
    if type == "full_monitoring"
        state_space = Iterators.product(1:model.grid,1:model.grid)
    elseif type == "competitor_only"
        state_space = 1:model.grid
    elseif bottom_filter !== nothing
        state_space = 1:model_grid
    else
        throw(ArgumentError("Memory type $(type) is not implemented"))
    end

    return [fill(optimism[i], length(state_space), model.grid) for i in 1:model.p], rand(1:model.grid, model.p) # Initialized Q matrix and states 
end

"""
    price_grid(model)

Generates the price_grid array used to map indices into prices.
  *  model - model structure
"""
function price_grid(model):
    price_grid_array = [zeros(model.grid) for _ in model.p]
    for i in 1:model.p
        increment = (model.action_space[i][1] - model.action_space[i][0])/(model.grid-1)
        price_grid_array[i] = [model.action_space[i][0] + increment*(j-1) for j in 1:model.grid]
    end
    return price_grid_array
end


"""
    train(model,n,policy,payoffs)

Simulate an auction according to format
  *  model - model structure
  *  n - number of iterations
  *  policy - function for action selection
         use eGreedy for e-greedy
         use eGreedy for greedy
         use Pushdown for downward force
  *  payoffs - function for payoff computation
"""
function train(model::Model,n::Int64,policy,payoffs,type)
    
    # Initializations
    Q, states = initialize(model, type)

    monitoring_technology = constructor(model, type)

    price_grid = price_grid(model)

    P = zeros(n,model.p)

    # TODO(b/1) generalize logic to p =/= 2
    Q_history = [zeros(n,length(Q[1][:,1]),model.grid) for _ in 1:model.p]
        
    if model.beta==0
        indicator_beta = 0
    else
        indicator_beta = 1
    end
    
    #Training
    for i=1:n

        # Update entropy of the policy 
        temperature = model.tau .*(1-indicator_beta) + model.tau *Base.exp(-model.beta*i)*(indicator_beta)
        
        # TODO(b/2): check type is vector of vectors
        states_flat = [flatten_states(model, monitoring_technology[(states,identity)][1:end-1]) for identity in 1:model.p] 

        # Random Experimentation 
        prices_index = policy(model, Q, states_flat, temperature, i) #returns vector of dimension p (2)
        
        prices = [price_grid[i][prices_index[i]] for i in 1:model.p]

        P[i,:] = prices
        states = prices_index
        states_next_flat = [flatten_states(model, monitoring_technology[(prices_index,identity)][1:end-1]) for identity in 1:model.p]

        payoff = payoffs(model, prices)
        
        # Update the Q-values
        for j=1:model.p
            Q[j] = Qupdate(model, Q, prices_index, payoff, states_flat, states_next_flat, j)
            Q_history[j][i,:,:] = Q[j]
        end
    end
    return Simulation(P, Q_history)
end


end