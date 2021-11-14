function mountaincar_cmax_main()
    cmax_steps = []
    model = MountainCar(0.0)
    cmax_planner = MountainCarCMAXPlanner(model, 1000, true_params)
    generateHeuristic!(cmax_planner)
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        cmax_agent = MountainCarCMAXAgent(mountaincar, cmax_planner)
        n_steps = run(cmax_agent, max_steps=1e4)
        push!(cmax_steps, n_steps)
        clearResiduals!(cmax_planner)
    end
    cmax_steps
end
