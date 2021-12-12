function mountaincar_rtaa_main(; params = true_params)::Vector{Int64}
    rtaa_steps = []
    model = MountainCar(0.0)
    planner = MountainCarRTAAPlanner(model, 1000, params)
    generateHeuristic!(planner)
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        agent = MountainCarRTAAAgent(mountaincar, planner)
        n_steps = run(agent, max_steps = 1e4)
        push!(rtaa_steps, n_steps)
        clearResiduals!(planner)
    end
    rtaa_steps
end

function mountaincar_true_main()
    true_steps = []
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        model = MountainCar(rock_c)
        planner = MountainCarRTAAPlanner(model, 1000, true_params)
        generateHeuristic!(planner)
        agent = MountainCarRTAAAgent(mountaincar, planner)
        n_steps = run(agent, max_steps = 1e4)
        push!(true_steps, n_steps)
    end
    true_steps
end

function mountaincar_rtaa_grid_search()
    data = fill(0, (length(theta1_values), length(theta2_values), length(range_of_values)))
    for idx1 = 1:length(theta1_values)
        theta1 = theta1_values[idx1]
        for idx2 = 1:length(theta2_values)
            theta2 = theta2_values[idx2]
            params = MountainCarParameters(theta1, theta2)
            rtaa_steps = mountaincar_rtaa_main(params = params)
            data[idx1, idx2, :] = rtaa_steps
        end
    end
    save(
        gridsearch_data_path,
        "data",
        data,
        "theta1",
        collect(theta1_values),
        "theta2",
        collect(theta2_values),
    )
end
