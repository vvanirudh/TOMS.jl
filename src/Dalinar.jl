module Dalinar

abstract type State end
abstract type Action end
abstract type Planner end
abstract type Parameters end

include("graph/astar.jl")
# --------- gridworld ------------------
include("env/gridworld.jl")
include("planner/gridworld_planner.jl")
include("agent/gridworld_agent.jl")
# --------- mountain car ---------------
include("env/mountain_car.jl")
include("planner/mountaincar_planner.jl")
include("agent/mountaincar_agent.jl")

function gridworld_main()
    gridworld = create_example_gridworld()
    planner = GridworldPlanner(gridworld, 100)
    agent = GridworldAgent(gridworld, planner)
    run(agent)
end

function mountaincar_main()
    mountaincar = MountainCar(0.045)
    model = MountainCar(0.0)
    planner = MountainCarPlanner(model, 1000, mountaincar.true_params)
    agent = MountainCarAgent(mountaincar, planner)
    run(agent)
end

end