# ------------------ FiniteModelClass agent ------------------------ 
struct MountainCarFiniteModelClassAgent
    mountaincar::MountainCar
    planners::Vector{MountainCarRTAAPlanner}
    transitions::Queue{MountainCarTransition}
    horizon::Int64
end

function MountainCarFiniteModelClassAgent(mountaincar::MountainCar, planners::Vector{MountainCarRTAAPlanner})
    transitions = Queue{MountainCarTransition}()
    horizon = 100
    MountainCarFiniteModelClassAgent(mountaincar, planners, transitions, horizon)
end

function run(agent::MountainCarFiniteModelClassAgent; max_steps=1e4, debug=false)
    state = init(agent.mountaincar)
    num_steps = 0
    # values = evaluate_values(agent, state)
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        # Choose the best planner
        values = evaluate_values(agent, state)
        bellman_losses = evaluate_bellman_losses(agent)
        total_losses = values + bellman_losses
        best_planner_idx = argmin(total_losses)
        if debug
            println("Num steps ", num_steps, " Values ", values, " Bellman ", bellman_losses, " best planner ", best_planner_idx)
        end
        best_planner = agent.planners[best_planner_idx]
        # Get action according to best planner
        best_action, info = act(best_planner, state)
        # update residuals
        updateResiduals!(best_planner, info)
        # step in env
        new_state, cost = step(agent.mountaincar, state, best_action, true_params)
        # Add transition to Queue
        if length(agent.transitions) >= agent.horizon
            dequeue!(agent.transitions)
        end
        enqueue!(agent.transitions, MountainCarTransition(state, best_action, cost, new_state))
        state = new_state
    end
    if num_steps < max_steps
        println("Reached goal in ", num_steps, " steps")
    else
        println("Did not reach goal")
    end
    num_steps
end

function evaluate_bellman_losses(agent::MountainCarFiniteModelClassAgent)
    bellman_losses = Vector{Float64}()
    for planner in agent.planners
        bellman_loss = 0.0
        for transition in agent.transitions
            initial_state = transition.initial_state
            final_state = transition.final_state
            action = transition.action
            predicted_state, _ = step(planner.mountaincar, initial_state, action, planner.params)
            value_final = getHeuristic(planner, final_state)
            value_predicted = getHeuristic(planner, predicted_state)
            # TODO: Should this be squared error?
            # bellman_loss += abs(value_final - value_predicted)
            bellman_loss += value_final - value_predicted
        end
        # if length(agent.transitions) > 0
        #     bellman_loss /= length(agent.transitions)
        # end
        push!(bellman_losses, bellman_loss)
    end
    bellman_losses
end

function evaluate_values(agent::MountainCarFiniteModelClassAgent, state::MountainCarDiscState)
    values = Vector{Float64}()
    for planner in agent.planners
        push!(values, getHeuristic(planner, state))
    end
    values
end