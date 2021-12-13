include("mountain_car_model_search_utils.jl")

function return_based_model_search(
    mountaincar::MountainCar,
    data::Array{MountainCarContTransition},
    optimization_params::MountainCarOptimizationParameters,
    horizon::Int64,
)
    params = true_params
    least_squares_params = get_least_squares_fit(mountaincar, params, data)
    x_array, xnext_array, cost_array = preprocess_data(mountaincar, data)

    eval_fn(p) = mfmc_evaluation(
        mountaincar,
        value_iteration(mountaincar, p)[1],
        horizon,
        x_array,
        xnext_array,
        cost_array,
        10,
    )
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
    horizon::Int64,
)
    params = true_params
    least_squares_params = get_least_squares_fit(mountaincar, params, data)
    x_array, xnext_array, cost_array = preprocess_data(mountaincar, data)

    function eval_fn(p)
        policy, values = value_iteration(mountaincar, p)
        bellman_evaluation(
            mountaincar,
            p,
            policy,
            values,
            horizon,
            x_array,
            xnext_array,
            cost_array,
            10,
        )
    end
    params = hill_climb(
        eval_fn,
        optimization_params,
        least_squares_params = least_squares_params,
    )
    params
end

struct MountainCarModelSearchAgent
    mountaincar::MountainCar
    model::MountainCar
    data::Array{MountainCarContTransition}
    optimization_params::MountainCarOptimizationParameters
    horizon::Int64
end

function MountainCarModelSearchAgent(
    mountaincar::MountainCar,
    model::MountainCar,
    horizon::Int64,
    num_episodes_offline::Int64,
)
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon)
    println("Generated ", length(data), " transitions offline")
    optimization_params =
        MountainCarOptimizationParameters([0.0024, 1], [-0.0025, 3], 90, false)
    MountainCarModelSearchAgent(mountaincar, model, data, optimization_params, horizon)
end

function run_return_based_model_search(
    agent::MountainCarModelSearchAgent;
    max_steps = 1e5,
    debug = false,
)
    params = return_based_model_search(
        agent.model,
        agent.data,
        agent.optimization_params,
        agent.horizon,
    )
    run(agent, params, max_steps = max_steps, debug = debug)
end

function run_bellman_based_model_search(
    agent::MountainCarModelSearchAgent;
    max_steps = 1e5,
    debug = false,
)
    params = bellman_based_model_search(
        agent.model,
        agent.data,
        agent.optimization_params,
        agent.horizon,
    )
    run(agent, params, max_steps = max_steps, debug = debug)
end

function run(
    agent::MountainCarModelSearchAgent,
    params::Array{Float64};
    max_steps = 1e5,
    debug = false,
)
    planner = MountainCarRTAAPlanner(
        agent.model,
        1000,
        MountainCarParameters(params[1], params[2]),
    )
    generateHeuristic!(planner)
    state = init(agent.mountaincar)
    num_steps = 0
    while !checkGoal(agent.mountaincar, state) && num_steps < max_steps
        num_steps += 1
        # best_action = actions[policy[disc_state_to_idx(agent.mountaincar, state)]]
        best_action, info = act(planner, state)
        updateResiduals!(planner, info)
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
