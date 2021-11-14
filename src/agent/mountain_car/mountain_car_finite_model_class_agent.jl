# ------------------ Utils -----------------------------------------

function getLeastKIndices(losses::Vector{Float64}, k::Int64)::Vector{Int64}
    sorted_indices = sortperm(losses)
    sorted_indices[1:k]
end

function getLessThanThresholdIndices(losses::Vector{Float64}, threshold::Float64)::Vector{Int64}
    indices = []
    for idx in 1:length(losses)
        loss = losses[idx]
        if loss < threshold
            push!(indices, idx)
        end
    end
    indices
end

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

function run(agent::MountainCarFiniteModelClassAgent; max_steps=1e4, debug=false)::Int64
    state = init(agent.mountaincar)
    num_steps = 0
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        # Choose the best planner
        values = evaluateValues(agent, state)
        bellman_losses = evaluateBellmanLosses(agent)
        total_losses = values + bellman_losses
        best_planner_idx = argmin(total_losses)
        # idxs = getLeastKIndices(bellman_losses, 3)
        # best_planner_idx = argmin(values[idxs])
        if debug
            println("Num steps ", num_steps, " Values ", values, " Bellman ", bellman_losses, " best planner ", best_planner_idx)
        end
        best_planner = agent.planners[best_planner_idx]
        # Get action according to best planner
        best_action, info = act(best_planner, state)
        # update residuals
        # updateResiduals!(best_planner, info)
        # step in env
        new_state, cost = step(agent.mountaincar, state, best_action, true_params)
        # Add transition to Queue
        # if length(agent.transitions) >= agent.horizon
        #     dequeue!(agent.transitions)
        # end
        enqueue!(agent.transitions, MountainCarTransition(state, best_action, cost, new_state, best_planner_idx))
        state = new_state
    end
    if num_steps < max_steps
        println("Reached goal in ", num_steps, " steps")
    else
        println("Did not reach goal")
    end
    num_steps
end

function evaluateBellmanLosses(agent::MountainCarFiniteModelClassAgent)
    bellman_losses = Vector{Float64}()
    for planner_idx in 1:length(agent.planners)
        planner = agent.planners[planner_idx]
        bellman_loss = 0.0
        for transition in agent.transitions
            idx = transition.id
            if idx != planner_idx
                continue
            end
            initial_state = transition.initial_state
            final_state = transition.final_state
            action = transition.action
            predicted_state, _ = step(planner.mountaincar, initial_state, action, planner.params)
            value_final = getHeuristic(planner, final_state)
            value_predicted = getHeuristic(planner, predicted_state)
            # TODO: Should this be squared error?
            bellman_loss += abs(value_final - value_predicted)
            # bellman_loss += value_final - value_predicted
        end
        # if length(agent.transitions) > 0
        #     bellman_loss /= length(agent.transitions)
        # end
        push!(bellman_losses, bellman_loss)
    end
    bellman_losses
end

function evaluateValues(agent::MountainCarFiniteModelClassAgent, state::MountainCarDiscState)
    values = Vector{Float64}()
    for planner in agent.planners
        push!(values, getHeuristic(planner, state))
    end
    values
end

# ------------------ FiniteModelClass local agent ------------------------
struct MountainCarFiniteModelClassLocalAgent
    finite_model_class_agent::MountainCarFiniteModelClassAgent
    discrepancy_region::Vector{Float64}
end

function MountainCarFiniteModelClassLocalAgent(mountaincar::MountainCar, planners::Vector{MountainCarRTAAPlanner})
    disc_rock_position = cont_state_to_disc(mountaincar, MountainCarState(mountaincar.rock_position, 0)).disc_position
    discrepancy_region_radius = 10
    discrepancy_region = [disc_rock_position - discrepancy_region_radius, disc_rock_position + discrepancy_region_radius]
    agent = MountainCarFiniteModelClassAgent(mountaincar, planners)
    MountainCarFiniteModelClassLocalAgent(agent, discrepancy_region)
end

function run(agent::MountainCarFiniteModelClassLocalAgent; max_steps=1e4, debug=false)
    state = init(agent.finite_model_class_agent.mountaincar)
    num_steps = 0
    while !checkGoal(agent.finite_model_class_agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        # Choose the best planner, by default it is the planner with the true model without rock
        best_planner = agent.finite_model_class_agent.planners[1]
        if inDiscrepancyRegion(agent, state)
            # Need to actually pick the best planner
            values = evaluateValues(agent.finite_model_class_agent, state)
            bellman_losses = evaluateBellmanLosses(agent.finite_model_class_agent)
            # total_losses = values + bellman_losses
            total_losses = bellman_losses
            best_planner_idx = argmin(total_losses)
            best_planner = agent.finite_model_class_agent.planners[best_planner_idx]
            if debug
                println("Num steps ", num_steps, " Values ", values, " Bellman ", bellman_losses, " best planner ", best_planner_idx)
            end
        end
        # Get action according to best planner
        best_action, info = act(best_planner, state)
        # update residuals
        updateResiduals!(best_planner, info)
        # step in env
        new_state, cost = step(agent.finite_model_class_agent.mountaincar, state, best_action, true_params)
        if inDiscrepancyRegion(agent, state)
            # Add to transitions
            enqueue!(agent.finite_model_class_agent.transitions, MountainCarTransition(state, best_action, cost, new_state, best_planner_idx))
        end
        state = new_state
    end
    if num_steps < max_steps
        println("Reached goal in ", num_steps, " steps")
    else
        println("Did not reach goal")
    end
    num_steps
end

function inDiscrepancyRegion(agent::MountainCarFiniteModelClassLocalAgent, state::MountainCarDiscState)
    state.disc_position <= agent.discrepancy_region[2] && state.disc_position >= agent.discrepancy_region[1]
end