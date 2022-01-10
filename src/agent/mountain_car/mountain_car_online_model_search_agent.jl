struct MountainCarOnlineModelSearchAgent
    mountaincar::MountainCar
    model::MountainCar
    num_eval_samples::Int64
end

function run(
    agent::MountainCarOnlineModelSearchAgent,
    rng::MersenneTwister;
    max_steps = 3e3,
    debug = false,
    max_likelihood = false,
    optimistic = false,
    epsilon = 0.0,
)
    # Start with initial policy
    start_params = MountainCarParameters(-0.0025, 3)
    params = vec(start_params)
    policy, _, _ = value_iteration(agent.model, params)
    num_steps = 0.0
    state = init_random(agent.mountaincar, rng)
    actions = getActions(agent.mountaincar)
    data::Array{MountainCarContTransition} = []
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        toss = rand(rng, Uniform(0, 1))
        action = actions[policy[cont_state_to_idx(agent.mountaincar, state)]]
        if toss < epsilon
            # random action
            action = actions[rand(rng, [0, 1])]
        end
        next_state, cost = step(agent.mountaincar, state, action, true_params)
        push!(data, MountainCarContTransition(state, action, cost, next_state))
        if num_steps % agent.num_eval_samples == 0
            # Update policy
            if max_likelihood
                params = get_least_squares_fit(agent.model, params, data)
            else
                optimization_params = MountainCarOptimizationParameters(
                    [0.0012, 0.5], vec(start_params), 75, false,
                )
                horizon = 500
                params = return_based_model_search(
                    agent.model,
                    data,
                    optimization_params,
                    horizon;
                    debug = debug,
                    optimistic = optimistic,
                )
            end
            if debug
                println("Moving to params ", params)
            end
            policy, _, _ = value_iteration(agent.model, params)
        end
        state = next_state
    end
    num_steps
end

function run_cmax(
    agent::MountainCarOnlineModelSearchAgent,
    rng::MersenneTwister;
    max_steps = 1e4,
    debug = false,
    cmax_threshold = 0.01,
)
    start_params = MountainCarParameters(-0.0025, 3)
    cost_matrix = generate_cost_matrix(agent.model)
    transition_matrix = generate_transition_matrix(agent.model, vec(start_params))
    policy, _, _ = value_iteration(agent.model, transition_matrix, cost_matrix)
    num_steps = 0.0
    state = init_random(agent.mountaincar, rng)
    actions = getActions(agent.mountaincar)
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        action = actions[policy[cont_state_to_idx(agent.mountaincar, state)]]
        next_state, cost = step(agent.mountaincar, state, action, true_params)
        next_predicted_state, _ = step(agent.model, state, action, true_params)
        if (abs(next_state.position - next_predicted_state.position) +
            abs(next_state.speed - next_predicted_state.speed)) > cmax_threshold
            # Discrepancy; inflate cost
            cost_matrix[cont_state_to_idx(agent.mountaincar, state),
                        policy[cont_state_to_idx(agent.mountaincar, state)]] =
                            1e5
            # Update policy
            policy, _, _ = value_iteration(agent.model, transition_matrix, cost_matrix)
        end
        state = next_state
    end
    num_steps
end
