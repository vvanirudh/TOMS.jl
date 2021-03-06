# RETURN FUNCTIONS
function mountaincar_return_based_model_search_main(num_episodes_offline::Int64)
    rng = MersenneTwister(0)
    model_search_steps = []
    model = MountainCar(0.0)
    horizon = 500
    for rock_c in range_of_values
        mountaincar = MountainCar(rock_c)
        data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon; rng = rng)
        agent =
            MountainCarModelSearchAgent(mountaincar, model, horizon, data)
        n_steps = run_return_based_model_search(agent; debug = true)
        push!(model_search_steps, n_steps)
    end
    model_search_steps
end

function mountaincar_return_based_model_search(rock_c::Float64, num_episodes_offline::Int64)
    rng = MersenneTwister(0)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon; rng = rng)
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_return_based_model_search(agent; debug = true)
    println("Reached goal in ", n_steps)
end

# BELLMAN ERROR FUNCTIONS
function mountaincar_bellman_based_model_search(rock_c::Float64, num_episodes_offline::Int64)
    rng = MersenneTwister(0)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon; rng = rng)
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_bellman_based_model_search(agent; debug = true)
    println(n_steps)
end

function mountaincar_bellman_based_model_search_main(num_episodes_offline::Int64)
    rng = MersenneTwister(0)
    model_search_steps = []
    model = MountainCar(0.0)
    horizon = 500
    for rock_c in range_of_values
        mountaincar = MountainCar(rock_c)
        data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon; rng = rng)
        agent =
            MountainCarModelSearchAgent(mountaincar, model, horizon, data)
        n_steps = run_bellman_based_model_search(agent, max_steps = 1e4)
        push!(model_search_steps, n_steps)
    end
    model_search_steps
end

function bellman_experiment(rock_c::Float64, num_episodes_offline::Int64; seed::Int64 = 0)
    rng = MersenneTwister(seed)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon, rng = rng)
    println("Generated ", length(data), " transitions")
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_return_based_model_search(agent; debug = true)
    n_bellman_steps = run_bellman_based_model_search(agent; debug = true)
    println("Return ", n_steps)
    println("Bellman ", n_bellman_steps)
    n_steps, n_bellman_steps
end

function bellman_experiment(
    rock_c::Float64,
    data::Array{MountainCarContTransition},
)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_return_based_model_search(agent; debug=false)
    n_bellman_steps = run_bellman_based_model_search(agent; debug=false)
    #println("Return ", n_steps)
    #println("Bellman ", n_bellman_steps)
    n_steps, n_bellman_steps
end

function bellman_experiment_episodes(rock_c::Float64)
    println()
    println("Experiment with rock_c ", rock_c)
    horizon = 500
    # num_episodes = [250, 500, 1000, 1500, 2000]
    num_episodes = [5, 10, 20]
    parent_rng = MersenneTwister(0)
    seeds = rand(parent_rng, UInt32, 10)
    experiment_data = get_experiment_data(
        MountainCar(rock_c),
        horizon,
        seeds,
        num_episodes[end],
    )
    n_steps = []
    n_bellman_steps = []
    for episodes in num_episodes
        n_sub_steps = []
        n_bellman_sub_steps = []
        for data in experiment_data
            sliced_data = data[1:min(episodes*horizon, length(data))]
            result = bellman_experiment(rock_c, sliced_data)
            push!(n_sub_steps, result[1])
            push!(n_bellman_sub_steps, result[2])
        end
        push!(n_steps, n_sub_steps)
        push!(n_bellman_steps, n_bellman_sub_steps)
    end
    n_matrix_steps = hcat(n_steps...)
    n_matrix_bellman_steps = hcat(n_bellman_steps...)
    mean_n_steps = mean(n_matrix_steps, dims=1)
    std_n_steps = std(n_matrix_steps, dims=1)
    mean_n_bellman_steps = mean(n_matrix_bellman_steps, dims=1)
    std_n_bellman_steps = std(n_matrix_bellman_steps, dims=1)
    println(num_episodes)
    println("Mean return steps ", mean_n_steps)
    println("Std return steps ", std_n_steps)
    println("Mean bellman steps ", mean_n_bellman_steps)
    println("Std bellman steps ", std_n_bellman_steps)
    println(n_steps)
    println(n_bellman_steps)
    n_steps, n_bellman_steps
end

function all_bellman_experiments()
    rock_c_values = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
    n_return_steps = []
    n_bellman_steps = []
    for rock_c in rock_c_values
        result = bellman_experiment_episodes(rock_c)
        push!(n_return_steps, result[1])
        push!(n_bellman_steps, result[2])
    end
    println()
    n_return_steps, n_bellman_steps
end

function bellman_experiment_rock_c()
    num_episodes = 1000
    rock_cs = [0, 0.01, 0.02, 0.03, 0.04]
    n_steps = []
    n_bellman_steps = []
    for rock_c in rock_cs
        result = bellman_experiment(rock_c, num_episodes; seed = 0)
        push!(n_steps, result[1])
        push!(n_bellman_steps, result[2])
    end
    println(num_episodes)
    println(n_steps)
    println(n_bellman_steps)
end

function get_experiment_data(
    mountaincar::MountainCar,
    horizon::Int64,
    seeds::Array{UInt32},
    max_episodes::Int64,
)::Array{Array{MountainCarContTransition}}
    experiment_data = []
    for seed in seeds
        rng = MersenneTwister(seed)
        push!(experiment_data, generate_batch_data(
            mountaincar,
            true_params,
            max_episodes,
            horizon;
            rng = rng
        ))
    end
    experiment_data
end

# PROFILING FUNCTIONS
function mfmc_evaluation_profile()
    mountaincar = MountainCar(0.0)
    data = generate_batch_data(mountaincar, true_params, 1000, 500)
    x_array, xnext_array, cost_array = preprocess_data(data)
    policy, values, _ = value_iteration(mountaincar, true_params)
    mfmc_evaluation(mountaincar, policy, values, 500, x_array, xnext_array, cost_array, 1)
end

function hill_climb_profile()
    mountaincar = MountainCar(0.0)
    data = generate_batch_data(mountaincar, true_params, 10, 10)
    optimization_params =
        MountainCarOptimizationParameters([0.0024, 1], [-0.0025, 3], 20, false)
    return_based_model_search(mountaincar, data, optimization_params, 10)
end

# HARDCODED DISTANCE FUNCTIONS
function hardcoded_experiment(rock_c::Float64, num_episodes_offline::Int64;
                              seed = 0)
    rng = MersenneTwister(seed)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon; rng = rng)
    println("Generated ", length(data), " transitions")
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_return_based_model_search(
        agent; debug = false, eval_distance = true
    )
    n_ensemble_steps = run_return_based_model_search(
        agent; hardcoded=true, debug = false, eval_distance = true
    )
    println("Without hardcoded distance ", n_steps)
    println("With hardcoded distance ", n_ensemble_steps)
    n_steps, n_ensemble_steps
end

function hardcoded_experiment_episodes(rock_c::Float64)
    # num_episodes = [250, 500, 750, 1000]
    num_episodes = [10, 25, 50]
    for episodes in num_episodes
        println("Running with ", episodes, " episodes")
        hardcoded_experiment(rock_c, episodes)
    end
end

function hardcoded_experiment_seeds(rock_c::Float64, num_episodes::Int64)
    parent_rng = MersenneTwister(0)
    seeds = rand(parent_rng, UInt32, 5)
    # num_episodes = 10
    n_steps = []
    n_hardcoded_steps = []
    for seed in seeds
        println()
        result = hardcoded_experiment(rock_c, num_episodes; seed=seed)
        push!(n_steps, result[1])
        push!(n_hardcoded_steps, result[2])
    end
    n_steps, n_hardcoded_steps
end

# OPTIMISTIC EXPERIMENTS
function optimistic_experiment(rock_c::Float64, num_episodes_offline::Int64;
                               seed = 0)
    rng = MersenneTwister(seed)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline,
                               horizon; rng = rng)
    println("Generated ", length(data), " transitions")
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_return_based_model_search(
        agent; debug = true,
    )
    n_optimistic_steps = run_return_based_model_search(
        agent; debug = true, optimistic = true,
    )
    println("Without optimism ", n_steps)
    println("With optimism ", n_optimistic_steps)
    n_steps, n_optimistic_steps
end
