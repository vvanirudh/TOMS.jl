function mountaincar_finite_model_class_main(;true_model=false, local_agent=false)
    finite_model_class_steps = []
    model = MountainCar(0.0)
    planners = Vector{MountainCarRTAAPlanner}()
    push!(planners, MountainCarRTAAPlanner(model, 1000, true_params))
    push!(planners, MountainCarRTAAPlanner(model, 1000, MountainCarParameters(-0.0025, 3.2)))
    push!(planners, MountainCarRTAAPlanner(model, 1000, MountainCarParameters(-0.0027, 3)))
    push!(planners, MountainCarRTAAPlanner(model, 1000, MountainCarParameters(-0.0025, 2.7)))
    push!(planners, MountainCarRTAAPlanner(model, 1000, MountainCarParameters(-0.0023, 3)))
    generateHeuristic!(planners)
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        if true_model
            planner = MountainCarRTAAPlanner(mountaincar, 1000, true_params)
            push!(planners, planner)
            generateHeuristic!(planners)
        end
        agent = MountainCarFiniteModelClassAgent(mountaincar, planners)
        if local_agent
            agent = MountainCarFiniteModelClassLocalAgent(mountaincar, planners)
        end
        n_steps = run(agent, debug=false)
        push!(finite_model_class_steps, n_steps)
        if true_model
            pop!(planners)
        end
        clearResiduals!(planners)
    end
    finite_model_class_steps
end
