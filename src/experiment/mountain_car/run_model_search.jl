# RETURN FUNCTIONS
function mountaincar_return_based_model_search_main()
    model_search_steps = []
    model = MountainCar(0.0)
    horizon = 500
    num_episodes_offline = 3000
    for rock_c in range_of_values
        mountaincar = MountainCar(rock_c)
        agent =
            MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
        n_steps = run_return_based_model_search(agent, max_steps = 1e4)
        push!(model_search_steps, n_steps)
    end
    model_search_steps
end

function mountaincar_return_based_model_search()
    Random.seed!(0)
    model = MountainCar(0.0)
    mountaincar = MountainCar(0.03)
    horizon = 500
    num_episodes_offline = 1000
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
    n_steps = run_return_based_model_search(agent)
    println(n_steps)
end

# PLANNER FUNCTIONS
function mountaincar_planner_return_based_model_search()
    model = MountainCar(0.0)
    mountaincar = MountainCar(0.03)
    horizon = 500
    num_episodes_offline = 1000
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
    n_steps = run_planner_return_based_model_search(agent)
    println(n_steps)
end

# ENSEMBLE FUNCTIONS
function ensemble_experiment(rock_c::Float64, num_episodes_offline::Int64)
    Random.seed!(0)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon)
    println("Generated ", length(data), " transitions")
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_return_based_model_search(agent)
    n_ensemble_steps = run_return_based_model_search(agent, ensemble=true)
    println("Without ensemble ", n_steps)
    println("With ensemble ", n_ensemble_steps)
    n_steps, n_ensemble_steps
end

function ensemble_experiment_episodes()
    rock_c = 0.03
    num_episodes = [100, 200, 400, 800, 1000, 2000]
    n_steps = []
    n_ensemble_steps = []
    for episodes in num_episodes
        result = ensemble_experiment(rock_c, episodes)
        push!(n_steps, result[1])
        push!(n_ensemble_steps, result[2])
    end
    println(num_episodes)
    println(n_steps)
    println(n_ensemble_steps)
end

function ensemble_experiment_rock_c()
    num_episodes = 1000
    rock_cs = [0, 0.01, 0.02, 0.03, 0.04]
    n_steps = []
    n_ensemble_steps = []
    for rock_c in rock_cs
        result = ensemble_experiment(rock_c, num_episodes)
        push!(n_steps, result[1])
        push!(n_ensemble_steps, result[2])
    end
    println(num_episodes)
    println(n_steps)
    println(n_ensemble_steps)
end

function mountaincar_return_ensemble_based_model_search()
    Random.seed!(0)
    model = MountainCar(0.0)
    mountaincar = MountainCar(0.03)
    horizon = 500
    num_episodes_offline = 100
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
    n_steps = run_return_based_model_search(agent, ensemble = true)
    println(n_steps)
end

# BELLMAN ERROR FUNCTIONS
function mountaincar_bellman_based_model_search()
    model = MountainCar(0.0)
    mountaincar = MountainCar(0.03)
    horizon = 500
    num_episodes_offline = 1000
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
    n_steps = run_bellman_based_model_search(agent)
    println(n_steps)
end

function mountaincar_bellman_based_model_search_main()
    model_search_steps = []
    model = MountainCar(0.0)
    horizon = 500
    num_episodes_offline = 3000
    for rock_c in range_of_values
        mountaincar = MountainCar(rock_c)
        agent =
            MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
        n_steps = run_bellman_based_model_search(agent, max_steps = 1e4)
        push!(model_search_steps, n_steps)
    end
    model_search_steps
end

function bellman_experiment(rock_c::Float64, num_episodes_offline::Int64, seed::Int64)
    rng = MersenneTwister(seed)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon, rng = rng)
    println("Generated ", length(data), " transitions")
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_return_based_model_search(agent)
    n_bellman_steps = run_bellman_based_model_search(agent)
    println("Return ", n_steps)
    println("Bellman ", n_bellman_steps)
    n_steps, n_bellman_steps
end

function bellman_experiment_episodes()
    rock_c = 0.035
    num_episodes = [100, 200, 400, 800, 1000]
    seeds = [1, 2, 3]
    n_steps = []
    n_bellman_steps = []
    for episodes in num_episodes
        return_steps = 0
        bellman_steps = 0
        for seed in seeds
            result = bellman_experiment(rock_c, episodes, seed)
            return_steps += result[1]
            bellman_steps += result[2]
        end
        push!(n_steps, return_steps/length(seeds))
        push!(n_bellman_steps, bellman_steps/length(seeds))
    end
    println(num_episodes)
    println(n_steps)
    println(n_bellman_steps)
end

function bellman_experiment_rock_c()
    num_episodes = 1000
    rock_cs = [0, 0.01, 0.02, 0.03, 0.04]
    n_steps = []
    n_bellman_steps = []
    for rock_c in rock_cs
        result = bellman_experiment(rock_c, num_episodes, 0)
        push!(n_steps, result[1])
        push!(n_bellman_steps, result[2])
    end
    println(num_episodes)
    println(n_steps)
    println(n_bellman_steps)
end

# PROFILING FUNCTIONS
function mfmc_evaluation_profile()
    mountaincar = MountainCar(0.0)
    data = generate_batch_data(mountaincar, true_params, 1000, 500)
    x_array, xnext_array, cost_array = preprocess_data(mountaincar, data)
    policy = random_policy(mountaincar)
    mfmc_evaluation(mountaincar, policy, 500, x_array, xnext_array, cost_array, 1)
end

function hill_climb_profile()
    mountaincar = MountainCar(0.0)
    data = generate_batch_data(mountaincar, true_params, 10, 10)
    optimization_params =
        MountainCarOptimizationParameters([0.0024, 1], [-0.0025, 3], 20, false)
    return_based_model_search(mountaincar, data, optimization_params, 10)
end

# HARDCODED DISTANCE FUNCTIONS
function hardcoded_experiment(rock_c::Float64, num_episodes_offline::Int64)
    Random.seed!(0)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon)
    println("Generated ", length(data), " transitions")
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_return_based_model_search(agent)
    n_ensemble_steps = run_return_based_model_search(agent; hardcoded=true)
    println("Without hardcoded distance ", n_steps)
    println("With hardcoded distance ", n_ensemble_steps)
    n_steps, n_ensemble_steps
end