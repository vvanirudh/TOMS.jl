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
    # saveHeuristic(agent.planner)
end

struct MountainCarCMAXAgent
    mountaincar::MountainCar
    cmax_planner::MountainCarCMAXPlanner
end

function run(agent::MountainCarCMAXAgent; max_steps=1e5)
    state = init(agent.mountaincar)
    num_steps = 0
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        best_action, info = act(agent.cmax_planner, state)
        new_state, cost = step(agent.mountaincar, state, best_action, agent.mountaincar.true_params)
        # Check if sim prediction matches true next state
        sim_state, _ = step(agent.cmax_planner.planner.mountaincar, state, best_action, agent.cmax_planner.planner.mountaincar.true_params)
        if sim_state != new_state
            # println("Discrepancy detected")
            addDiscrepancy!(agent.cmax_planner, state, best_action)
            # replan
            _, info = act(agent.cmax_planner, state)
        end
        updateResiduals!(agent.cmax_planner, info)
        state = new_state
    end
    if num_steps < max_steps
        println("Reached goal in ", num_steps, " steps")
    else
        println("Did not reach goal")
    end
end