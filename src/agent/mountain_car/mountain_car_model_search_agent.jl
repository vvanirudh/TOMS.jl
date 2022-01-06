include("model_search/mountain_car_model_search_utils.jl")

struct MountainCarModelSearchAgent
    mountaincar::MountainCar
    model::MountainCar
    data::Array{MountainCarContTransition}
    optimization_params::MountainCarOptimizationParameters
    horizon::Int64
end

function return_based_model_search(
    mountaincar::MountainCar,
    data::Array{MountainCarContTransition},
    optimization_params::MountainCarOptimizationParameters,
    horizon::Int64;
    ensemble::Bool = false,
    hardcoded::Bool = false,
    num_episodes_eval::Int64 = 1,
    debug::Bool = false,
)
    params = true_params
    least_squares_params = get_least_squares_fit(mountaincar, params, data)
    x_matrices_array, x_array, x_next_array, disp_array, cost_array = preprocess_data(mountaincar, data)
    ensembles = nothing
    if ensemble
        ensembles = fit_ensembles(x_array, disp_array)
    end

    function eval_fn(p)
        policy, _, converged = value_iteration(mountaincar, p)
        if converged
            return mfmc_evaluation(
                mountaincar,
                policy,
                horizon,
                x_matrices_array,
                x_next_array,
                cost_array,
                num_episodes_eval;
                ensembles = ensembles,
                hardcoded = hardcoded,
                debug = debug,
            )
        else
            if debug
                println("Value Iteration did not converge. Skipping parameters ", p)
            end
            return Inf
        end
    end

    params = hill_climb(
        eval_fn,
        optimization_params;
        least_squares_params = least_squares_params,
        debug = debug,
    )
    params
end

function planner_return_based_model_search(
    mountaincar::MountainCar,
    data::Array{MountainCarContTransition},
    optimization_params::MountainCarOptimizationParameters,
    horizon::Int64;
    num_episodes_eval::Int64 = 10,
)
    params = true_params
    least_squares_params = get_least_squares_fit(mountaincar, params, data)
    x_matrices_array, x_array, xnext_array, disp_array, cost_array = preprocess_data(mountaincar, data)

    function eval_fn(p)
        policy, _ = rtaa_planning(mountaincar, p)
        mfmc_evaluation(
            mountaincar,
            policy,
            horizon,
            x_matrices_array,
            xnext_array,
            cost_array,
            num_episodes_eval,
        )
    end
    params = hill_climb(
        eval_fn,
        optimization_params,
        least_squares_params = least_squares_params,
    )
    params
end

function bellman_based_model_search(
    mountaincar::MountainCar,
    data::Array{MountainCarContTransition},
    optimization_params::MountainCarOptimizationParameters,
    horizon::Int64;
    num_episodes_eval::Int64 = 3,
    debug::Bool = false,
)
    params = true_params
    least_squares_params = get_least_squares_fit(mountaincar, params, data)
    x_matrices_array, x_array, xnext_array, disp_array, cost_array = preprocess_data(mountaincar, data)
    gamma = 1.0
    function eval_fn(p)
        policy, values, converged = value_iteration(mountaincar, p; gamma = gamma)
        if converged
            return bellman_evaluation(
                mountaincar,
                p,
                policy,
                values,
                horizon,
                x_matrices_array,
                xnext_array,
                cost_array,
                num_episodes_eval;
                debug = debug,
            )
        else
            if debug
                println("Value Iteration did not converge. Skipping parameters ", p)
            end
            return Inf
        end
    end
    params = hill_climb(
        eval_fn,
        optimization_params;
        least_squares_params = least_squares_params,
        debug = debug,
    )
    params
end

function MountainCarModelSearchAgent(
    mountaincar::MountainCar,
    model::MountainCar,
    horizon::Int64,
    num_episodes_offline::Int64,
)
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon)
    # println("Generated ", length(data), " transitions offline")
    MountainCarModelSearchAgent(mountaincar, model, horizon, data)
end

function MountainCarModelSearchAgent(
    mountaincar::MountainCar,
    model::MountainCar,
    horizon::Int64,
    data::Array{MountainCarContTransition},
)
    maximum_num_evaluations = 120
    optimization_params =
        MountainCarOptimizationParameters([0.0024, 1], [-0.0025, 3], maximum_num_evaluations, false)
    MountainCarModelSearchAgent(mountaincar, model, data, optimization_params, horizon)
end

function run_return_based_model_search(
    agent::MountainCarModelSearchAgent;
    ensemble = false,
    hardcoded = false,
    max_steps = 1e4,
    debug = false,
    eval_distance = false,
)
    params = return_based_model_search(
        agent.model,
        agent.data,
        agent.optimization_params,
        agent.horizon;
        ensemble = ensemble,
        hardcoded = hardcoded,
        debug = debug,
    )
    run(agent, params; max_steps = max_steps, eval_distance = eval_distance)
end

function run_planner_return_based_model_search(
    agent::MountainCarModelSearchAgent;
    max_steps = 1e4,
    debug = false,
)
    params = planner_return_based_model_search(
        agent.model,
        agent.data,
        agent.optimization_params,
        agent.horizon,
    )
    run(agent, params, max_steps = max_steps, debug = debug)
end

function run_bellman_based_model_search(
    agent::MountainCarModelSearchAgent;
    max_steps = 1e4,
    debug = false,
)
    params = bellman_based_model_search(
        agent.model,
        agent.data,
        agent.optimization_params,
        agent.horizon;
        debug = debug,
    )
    run(agent, params, max_steps = max_steps)
end

function run(
    agent::MountainCarModelSearchAgent,
    params::Array{Float64};
    max_steps = 1e4,
    debug = false,
    eval_distance = false,
)
    policy, _, _ = value_iteration(agent.model, params)
    actions = getActions(agent.mountaincar)
    state = init(agent.mountaincar; cont = true)
    num_steps = 0
    if eval_distance
        timestep_distances = []
        x_matrices_array, _, _, _, _ = preprocess_data(agent.mountaincar,
                                                       agent.data)
        position_range = agent.mountaincar.max_position - agent.mountaincar.min_position
        speed_range = 2 * agent.mountaincar.max_speed
        normalization = permutedims([position_range, speed_range])
    end
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        a = policy[cont_state_to_idx(agent.mountaincar, state)]
        best_action = actions[a]
        if eval_distance
            distances = distance_fn(vec(state), x_matrices_array[a],
                                    normalization)
            push!(timestep_distances, minimum(distances))
        end
        state, cost =
            step(agent.mountaincar, state, best_action, true_params, debug = debug)
        if debug
            println(state.position, " ", state.speed, " ", best_action.id)
        end
    end
    if eval_distance
        println("Mean distance ", mean(timestep_distances), " Std distance ",
                std(timestep_distances), " Max distance ",
                maximum(timestep_distances), " Min distance ",
                minimum(timestep_distances))
    end
    num_steps
end
