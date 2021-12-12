using DataStructures

# ------------------ RTAA* agent ------------------------ 
struct MountainCarRTAAAgent
    mountaincar::MountainCar
    planner::MountainCarRTAAPlanner
end

function run(agent::MountainCarRTAAAgent; max_steps = 1e5, debug = false)
    state = init(agent.mountaincar)
    num_steps = 0
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        best_action, info = act(agent.planner, state)
        updateResiduals!(agent.planner, info)
        state, cost =
            step(agent.mountaincar, state, best_action, true_params, debug = debug)
        if debug
            cont_state = disc_state_to_cont(agent.mountaincar, state)
            println(cont_state.position, " ", cont_state.speed, " ", best_action.id)
        end
    end
    if num_steps < max_steps
        println("Reached goal in ", num_steps, " steps")
    else
        println("Did not reach goal")
    end
    num_steps
end
