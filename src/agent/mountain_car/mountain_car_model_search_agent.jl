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
    hardcoded::Bool = false,
    num_episodes_eval::Int64 = 1,
    debug::Bool = false,
    eval_distance::Bool = false,
    optimistic::Bool = false,
    model_evaluation::Bool = false,
)
    params = true_params
    least_squares_params = get_least_squares_fit(mountaincar, params, data)
    x_matrices_array, _, x_next_array, _, cost_array = preprocess_data(data)
    function eval_fn(p)
        policy, values, converged = value_iteration(mountaincar, p)
        if converged
            if optimistic
                return mfmc_optimistic_evaluation(
                    mountaincar,
                    policy,
                    p,
                    horizon,
                    x_matrices_array,
                    x_next_array,
                    cost_array,
                    num_episodes_eval,
                    0.015;
                    debug = debug,
                )
            elseif model_evaluation
                return mfmc_model_evaluation(
                    mountaincar,
                    policy,
                    p,
                    horizon,
                    x_matrices_array,
                    x_next_array,
                    cost_array,
                    num_episodes_eval,
                    0.01;
                    debug = debug,
                )
            else
                return mfmc_evaluation(
                    mountaincar,
                    policy,
                    values,
                    horizon,
                    x_matrices_array,
                    x_next_array,
                    cost_array,
                    num_episodes_eval;
                    hardcoded = hardcoded,
                    debug = debug,
                )
            end
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
    if eval_distance
        policy, values, _ = value_iteration(mountaincar, params)
        estimated_return, eval_distances = mfmc_evaluation(
            mountaincar, policy, values, horizon,
            x_matrices_array, x_next_array, cost_array,
            num_episodes_eval;
            eval_distance = eval_distance,
        )
        println("Mean distance ", mean(eval_distances), 
                " Std distance ", std(eval_distances),
                " Max distance ", maximum(eval_distances),
                " Min distance ", minimum(eval_distances))
        println("Estimated return is ", estimated_return)
    end
    params
end

function bellman_based_model_search(
    mountaincar::MountainCar,
    data::Array{MountainCarContTransition},
    optimization_params::MountainCarOptimizationParameters,
    horizon::Int64;
    num_episodes_eval::Int64 = 1,
    debug::Bool = false,
)
    params = true_params
    least_squares_params = get_least_squares_fit(mountaincar, params, data)
    x_matrices_array, _, xnext_array, _, _ = preprocess_data(data)
    function eval_fn(p)
        policy, values, converged = value_iteration(mountaincar, p)
        if converged
            return bellman_evaluation(
                mountaincar,
                p,
                policy,
                values,
                horizon,
                x_matrices_array,
                xnext_array,
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
    maximum_num_evaluations = 90
    optimization_params =
        MountainCarOptimizationParameters([0.0012, 0.5], [-0.0025, 3], maximum_num_evaluations, false)
    MountainCarModelSearchAgent(mountaincar, model, data, optimization_params, horizon)
end

function run_return_based_model_search(
    agent::MountainCarModelSearchAgent;
    ensemble = false,
    hardcoded = false,
    max_steps = 500,
    debug = false,
    eval_distance = false,
    optimistic = false,
    model_evaluation = false,
)
    params = return_based_model_search(
        agent.model,
        agent.data,
        agent.optimization_params,
        agent.horizon;
        hardcoded = hardcoded,
        debug = debug,
        eval_distance = eval_distance,
        optimistic = optimistic,
        model_evaluation = model_evaluation,
    )
    println("Params found is ", params)
    run(agent, params; max_steps = max_steps)
end

function run_bellman_based_model_search(
    agent::MountainCarModelSearchAgent;
    max_steps = 500,
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
    max_steps = 500,
    debug = false,
    num_eval = 10,
)
    policy, _, _ = value_iteration(agent.model, params)
    rng = MersenneTwister(0)
    num_steps = 0.0
    for _ in 1:num_eval
        num_steps += length(
            simulate_episode(
                agent.mountaincar,
                true_params,
                agent.horizon;
                policy = policy,
                rng = rng,
            )
        )
    end
    num_steps/num_eval
end
