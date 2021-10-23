struct MountainCarAgent
    mountaincar::MountainCar
    planner::MountainCarPlanner
end

function run(agent::MountainCarAgent; max_steps=1e5)
    state = init(agent.mountaincar)
    num_steps = 0
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        # render(agent.mountaincar, state)
        best_action, info = act(agent.planner, state)
        updateResiduals!(agent.planner, info)
        state, cost = step(agent.mountaincar, state, best_action, agent.mountaincar.true_params)
    end
    if num_steps < max_steps
        println("Reached goal in ", num_steps, " steps")
    else
        println("Did not reach goal")
    end
    # render(agent.mountaincar, state)
end