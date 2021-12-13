function mountaincar_model_search_main()
    model_search_steps = []
    model = MountainCar(0.0)
    horizon = 500
    num_episodes_offline = 1000
    for rock_c in range_of_values
        mountaincar = MountainCar(rock_c)
        agent =
            MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
        n_steps = run(agent, max_steps = 1e4)
        push!(model_search_steps, n_steps)
    end
    model_search_steps
end

function mountaincar_model_search()
    model = MountainCar(0.0)
    mountaincar = MountainCar(0.04)
    horizon = 500
    num_episodes_offline = 1000
    agent = MountainCarModelSearchAgent(mountaincar, model, horizon, num_episodes_offline)
    println("Return")
    n_steps = run_return_based_model_search(agent)
    println(n_steps)
    println("Bellman")
    n_steps = run_bellman_based_model_search(agent)
    println(n_steps)
end
