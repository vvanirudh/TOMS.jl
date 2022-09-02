function mountaincar_run_agnostic_sysid(
    rock_c::Float64, 
    position_sigma::Float64; 
    value_aware = true,
)
    mountaincar = MountainCar(rock_c; position_sigma = position_sigma)
    model = MountainCar(0.0)
    horizon = 500
    agent = MountainCarAgnosticSysIDAgent(
        mountaincar,
        model,
        horizon,
    )
    run(agent; debug = true, value_aware = value_aware)
end

function mountaincar_run_agnostic_sysid_all(
    rock_c::Float64, 
    position_sigma::Float64;
    num_iterations = 50,
)
    mountaincar = MountainCar(rock_c; position_sigma = position_sigma)
    model = MountainCar(0.0)
    horizon = 500
    agent = MountainCarAgnosticSysIDAgent(
        mountaincar,
        model,
        horizon,
    )
    value_aware_costs, value_aware_consistencies, optimal_cost = run(agent; debug = true, value_aware = true, num_iterations=num_iterations)
    # mle_costs, mle_consistencies, _ = run(agent; debug = true, value_aware=false, num_iterations=num_iterations)
    # opt_value_aware_costs, _, _ = run(agent; debug=true, value_aware=true, num_iterations=num_iterations, use_optimal_values=true)
    opt_policy_value_aware_costs, _, _ = run(agent; debug=true, value_aware=true, num_iterations=num_iterations, use_optimal_policy=true)

    # Plot both
    p1 = plot(1:length(value_aware_costs), value_aware_costs, lw=3, label="MA")
    # p1 = plot!(1:length(mle_costs), mle_costs, lw=3, label="MLE")
    # p1 = plot!(1:length(opt_value_aware_costs), opt_value_aware_costs, lw=3, label="MA Opt Val")
    p1 = plot!(1:length(opt_policy_value_aware_costs), opt_policy_value_aware_costs, lw=3, label="MA Opt Pol")
    p1 = plot!(1:length(value_aware_costs), ones(length(value_aware_costs)) * optimal_cost, lw=3, label="Opt")
    p1 = xlabel!("Iteration")
    p1 = ylabel!("Cost of rollout")
    p1 = title!("Cost vs Iteration on Mountain Car with c=" * string(rock_c))

    #= p2 = plot(1:length(value_aware_consistencies), value_aware_consistencies, lw=3, label="MA")
    p2 = plot!(1:length(mle_consistencies), mle_consistencies, lw=3, label="MLE")
    p2 = xlabel!("Iteration")
    p2 = ylabel!("Consistency")
    p2 = title!("Consistency vs Iteration on Mountain Car with c=" * string(rock_c))

    plot(p1, p2, layout=(2, 1)) =#

end