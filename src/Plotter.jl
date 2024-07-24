module Plotter

using Plots
using Statistics
using DelimitedFiles
using ProgressMeter
#using GLMakie
using Main.Simplicity

export convergence
export freq_table
export heatmap2dim
export repeat_auctions
export data_to_file



"""
    convergence(m,S,n)

Checks for convergence. Returns 1 if all players converged.

  *  m - instance of Model
  *  S - instance of Simulation based on model m
  *  n - iterations
"""
function convergence(m::Model,s::Simulation,n)
    maxima = [Set([findmax(s.Q[j,:,player])[2] for j=n-999:n]) for player=1:m.p]
    for i in maxima
        if length(i) >1
            return 0
        end
    end
    return 1
end


"""
    freq_table(m,B,a,b)

Computes the frequencies of bids for bidders a and b


  * # m - instance of model
  * # B - vector of bids (Simulation.B or output of repeat_auctions)
  * # a,b - identities of bidders, integers <= m.p
"""
function freq_table(m::Model,B,a,b)
    ctr = Array{Float64}(undef, m.grid, m.grid)
    for i in 1:m.grid
        index = findall(B[:,a]./(m.grid+1).==i./(m.grid+1))
        for j in 1:m.grid
            ctr[i,j] = length(findall(B[index,b]./(m.grid+1).==j./(m.grid+1)))   
        end
    end
    return ctr
end


"""
    heatmap2dim(m,s,name,a,b)

Plots a 2D-heatmap of bids for players a and b

  *Default*: a=1,b=2

  * # m - instance of model
  * # B - vector of bids (Simulation.B or output of repeat_auctions)
  * # name - string for plot title. Default: fig
"""
function heatmap2dim(m,B,name::String="fig",a::Int64=1,b::Int64=2)
    # Create a matrix with the frequency values
    
    ctr = freq_table(m,B,a,b)
    
    
    labels = String[]
    for i in 1:m.grid
        if i%2==0
            push!(labels,"$(i./(m.grid+1).*m.top)")
        else
            push!(labels,"")
        end
    end
    
    Plots.heatmap(xticks=(1:m.grid, labels),yticks=(1:m.grid, labels),ctr./length(B[:,a]), c = :heat,size=(1200,1000))

    fontsize = 5
    nrow, ncol = size(ctr)
    ann = [(j,i, Plots.text(round(ctr[i,j], digits=2), fontsize, :black, :center))
                for i in 1:nrow for j in 1:ncol]
    annotate!(ann, linecolor=:black)
    savefig(name)
end


"""
    repeat_auctions(m,n,k,policy,fringe,synch)

Simulates k auctions for n iterations and outputs median and mean bids in the last episodes

  *  m - instance of Model
  *  n - number of episodes per simulation
  *  k - number of simulations
  *  policy - egreedy/pushdown/local
  *  fringe - instance of Fringe
  *  synch - choice of feedback
"""
function repeat_auctions(m::Model,n::Int64,k::Int64,policy,fringe::Fringe,synch::Int64=0)
    Output = Array{Float64}(undef, k, m.p.+1)
    @showprogress for i in 1:k
        local s = train(m,n,policy,fringe,synch)
        Output[i,:] = [median(s.B[n-999:n,:],dims = 1) convergence(m,s,n)]
    end

    return Output
end


"""
    data_to_file(Output,name)

Writes simulation results to a .csv file

  *  Output - output produced by repeat_auctions
  *  name - name of the .csv output. Default: 'fig'
"""
function data_to_file(Output,name::String="fig")

    header = permutedims([["Player $i" for i=1:size(Output)[2]-1];["Convergence"]])
    writedlm(name*".csv",  [header ; Output], ',')

    return nothing
end


end