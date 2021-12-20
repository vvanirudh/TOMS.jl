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

function mountaincar_return_based_model_search()
    Random.seed!(0)
    model = MountainCar(0.0)
    mountaincar = MountainCar(0.03)
    horizon = 500
    num_episodes_offline = 100
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
    n_steps = run_return_based_model_search(agent)
    println(n_steps)
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

function mountaincar_planner_return_based_model_search()
    model = MountainCar(0.0)
    mountaincar = MountainCar(0.03)
    horizon = 500
    num_episodes_offline = 1000
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
    n_steps = run_planner_return_based_model_search(agent)
    println(n_steps)
end

function ensemble_experiment(rock_c::Float64, num_episodes_offline::Int64)
    Random.seed!(0)
    model = MountainCar(0.0)
    mountaincar = MountainCar(rock_c)
    horizon = 500
    data = generate_batch_data(mountaincar, true_params, num_episodes_offline, horizon)
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, data)
    n_steps = run_return_based_model_search(agent)
    n_ensemble_steps = run_return_based_model_search(agent, ensemble=true)
    println("Without ensemble ", n_steps)
    println("With ensemble ", n_ensemble_steps)
end

function mountaincar_bellman_based_model_search()
    model = MountainCar(0.0)
    mountaincar = MountainCar(0.03)
    horizon = 500
    num_episodes_offline = 1000
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
    n_steps = run_bellman_based_model_search(agent)
    println(n_steps)
end

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