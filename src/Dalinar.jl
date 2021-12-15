module Dalinar

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Plots
using JLD: save, load
using Debugger
using Random
using LinearAlgebra
using PyCall
using Distances
using NearestNeighbors
using IterTools
using Distributions
using ThreadsX

Random.seed!(0)

abstract type State end
abstract type Action end
abstract type Planner end
abstract type Parameters end
abstract type Transition end


include("graph/astar.jl")
# --------- gridworld ------------------
include("env/gridworld/gridworld.jl")
include("planner/gridworld/gridworld_planner.jl")
include("agent/gridworld/gridworld_agent.jl")
include("experiment/gridworld/run_all.jl")
# --------- mountain car ---------------
include("env/mountain_car/mountain_car.jl")
include("planner/mountain_car/mountain_car_planners.jl")
include("agent/mountain_car/mountain_car_agents.jl")
include("experiment/mountain_car/run_all.jl")

end
