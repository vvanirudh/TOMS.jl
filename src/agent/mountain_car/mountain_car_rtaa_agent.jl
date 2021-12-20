# ------------------ RTAA* agent ------------------------ 
struct MountainCarRTAAAgent
    mountaincar::MountainCar
    planner::MountainCarRTAAPlanner
end

function run(agent::MountainCarRTAAAgent; max_steps = 1e5, debug = false)
    state = init(agent.mountaincar; cont = true)
    num_steps = 0
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        best_action, info = act(agent.planner, cont_state_to_disc(agent.mountincar, state))
        updateResiduals!(agent.planner, info)
        state, cost =
            step(agent.mountaincar, state, best_action, true_params, debug = debug)
        if debug
            println(state.position, " ", state.speed, " ", best_action.id)
        end
    end
    if num_steps < max_steps
        println("Reached goal in ", num_steps, " steps")
    else
        println("Did not reach goal")
    end
    num_steps
end
