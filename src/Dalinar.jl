module Dalinar

include("mdp/mdp.jl")
include("graph/astar.jl")
include("env/gridworld.jl")
include("planner/gridworld_planner.jl")

function gridworld_main()
    size = 1000
    grid = fill(0, (size, size))
    grid[Int(floor(size/4)):size, Int(floor(size/2))] .= 1
    grid[Int(floor(size/2)), Int(floor(size/2))+2:end] .= 1

    start_state = GridworldState(1, 1)
    goal_state = GridworldState(size, size)

    gridworld = Gridworld(grid, false, start_state, goal_state)
    planner = GridworldPlanner(gridworld, 2)

    state = init(gridworld)
    iteration = 0
    while !checkGoal(gridworld, state)
        iteration += 1
        # println("Iteration", iteration)
        render(gridworld, state)
        best_action, info = act(planner, state)
        updateResiduals!(planner, info)
        state, cost = step(gridworld, state, best_action)
    end
    println("reached goal ", iteration)
    render(gridworld, state)
end

end
