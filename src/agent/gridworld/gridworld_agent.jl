struct GridworldAgent
    gridworld::Gridworld 
    planner::GridworldPlanner
end

function run(agent::GridworldAgent; max_steps=1e6)
    state = init(agent.gridworld)
    num_steps = 0
    while !checkGoal(agent.gridworld, state) && num_steps<max_steps
        num_steps += 1
        render(agent.gridworld, state)
        best_action, info = act(agent.planner, state)
        updateResiduals!(agent.planner, info)
        state, cost = step(agent.gridworld, state, best_action)
    end
    if num_steps < max_steps
        println("Reached goal in ", num_steps, " steps")
    else
        println("Did not reach goal")
    end
    render(agent.gridworld, state)
end