function mountaincar_run_online_model_search(rock_c::Float64,
                                             num_eval_samples::Int64;
                                             max_likelihood = false,
                                             optimistic = false,
                                             model_evaluation = false,
                                             debug = false,
                                             seed = 0)
    rng = MersenneTwister(seed)
    mountaincar = MountainCar(rock_c)
    model = MountainCar(0.0)
    agent = MountainCarOnlineModelSearchAgent(mountaincar, model,
                                              num_eval_samples)
    n_steps = run(agent, rng; debug = debug, max_likelihood = max_likelihood,
                  optimistic = optimistic, model_evaluation = model_evaluation,)
    n_steps
end

function mountaincar_run_cmax(rock_c::Float64;
                              debug = false,
                              seed = 0)
    rng = MersenneTwister(seed)
    mountaincar = MountainCar(rock_c)
    model = MountainCar(0.0)
    agent = MountainCarOnlineModelSearchAgent(mountaincar, model,
                                              10)
    n_steps = run_cmax(agent, rng; debug = debug)
    n_steps
end

function mountaincar_run_true(rock_c::Float64;
                              debug = false,
                              seed = 0)
    rng = MersenneTwister(seed)
    mountaincar = MountainCar(rock_c)
    model = MountainCar(0.0)
    agent = MountainCarOnlineModelSearchAgent(mountaincar, model, 10)
    n_steps = run_true(agent, rng; debug = debug)
    n_steps
end

function mountaincar_run_online_experiments()
    num_eval_samples = 50
    seeds = collect(1:10)
    # ml_steps = []
    # oms_steps = []
    # cmax_steps = []
    # true_steps = []
    # rbms_steps = []
    me_steps = []
    rock_c_values = collect(0.02:0.002:0.03)
    for rock_c in rock_c_values
        # ml_sub_steps = []
        # oms_sub_steps = []
        # cmax_sub_steps = []
        # true_sub_steps = []
        # rbms_sub_steps = []
        me_sub_steps = []
        for seed in seeds
            # println("MLE with rock_c ", rock_c, " seed ", seed)
            # push!(ml_sub_steps, mountaincar_run_online_model_search(
            #     rock_c, num_eval_samples; max_likelihood = true, seed = seed,
            #     debug = false,
            # ))
            # println("OMS with rock_c ", rock_c, " seed ", seed)
            # push!(oms_sub_steps, mountaincar_run_online_model_search(
            #     rock_c, num_eval_samples; optimistic = true, seed = seed,
            #     debug = false,
            # ))
            # println("CMAX with rock_c ", rock_c, " seed ", seed)
            # push!(cmax_sub_steps, mountaincar_run_cmax(
            #     rock_c; seed = seed, debug = false,
            # ))
            # println("True with rock_c ", rock_c, " seed ", seed)
            # push!(true_sub_steps, mountaincar_run_true(
            #     rock_c; seed = seed, debug = false,
            # ))
            # println("RBMS with rock_c ", rock_c, " seed ", seed)
            # push!(rbms_sub_steps, mountaincar_run_online_model_search(
            #     rock_c, num_eval_samples; optimistic = false, seed = seed,
            #     debug = false,
            # ))
            println("ME with rock_c ", rock_c, " seed ", seed)
            push!(me_sub_steps, mountaincar_run_online_model_search(
                rock_c, num_eval_samples; model_evaluation = true, seed = seed,
                debug = false,
            ))
        end
        # push!(ml_steps, ml_sub_steps)
        # push!(oms_steps, oms_sub_steps)
        # push!(cmax_steps, cmax_sub_steps)
        # push!(true_steps, true_sub_steps)
        # push!(rbms_steps, rbms_sub_steps)
        push!(me_steps, me_sub_steps)
    end
    # ml_mean_steps = [mean(x) for x in ml_steps]
    # ml_std_steps = [std(x) for x in ml_steps] ./ sqrt(length(seeds))
    # oms_mean_steps = [mean(x) for x in oms_steps]
    # oms_std_steps = [std(x) for x in oms_steps] ./ sqrt(length(seeds))
    # cmax_mean_steps = [mean(x) for x in cmax_steps]
    # cmax_std_steps = [std(x) for x in cmax_steps] ./ sqrt(length(seeds))
    # true_mean_steps = [mean(x) for x in true_steps]
    # true_std_steps = [std(x) for x in true_steps] ./ sqrt(length(seeds))
    # rbms_mean_steps = [mean(x) for x in rbms_steps]
    # rbms_std_steps = [std(x) for x in rbms_steps] ./ sqrt(length(seeds))
    me_mean_steps = [mean(x) for x in me_steps]
    me_std_steps = [std(x) for x in me_steps] ./ sqrt(length(seeds))

    # println("MLE")
    # println(ml_steps)
    # println("OMS")
    # println(oms_steps)
    # println("CMAX")
    # println(cmax_steps)
    # println("TRUE")
    # println(true_steps)
    # println("RBMS")
    # println(rbms_steps)
    println("ME")
    println(me_steps)

    # println("MLE")
    # println(ml_mean_steps)
    # println("OMS")
    # println(oms_mean_steps)
    # println("CMAX")
    # println(cmax_mean_steps)
    # println("TRUE")
    # println(true_mean_steps)
    # println("RBMS")
    # println(rbms_mean_steps)
    println("ME")
    println(me_mean_steps)

    # println("MLE")
    # println(ml_std_steps)
    # println("OMS")
    # println(oms_std_steps)
    # println("CMAX")
    # println(cmax_std_steps)
    # println("TRUE")
    # println(true_std_steps)
    # println("RBMS")
    # println(rbms_std_steps)
    println("ME")
    println(me_std_steps)

    # plot(rock_c_values, ml_mean_steps, grid=true, yerror=ml_std_steps, lw=3,
    #      label="MLE")
    # plot!(rock_c_values, oms_mean_steps, grid=true, yerror=oms_std_steps, lw=3,
    #      label="TOMS")
    # plot!(rock_c_values, cmax_mean_steps, grid=true, yerror=cmax_std_steps, lw=3,
    #       label="CMAX")
    # plot!(rock_c_values, true_mean_steps, grid=true, yerror=true_std_steps, lw=3,
    #      label="TRUE")
    # plot!(rock_c_values, rbms_mean_steps, grid=true, yerror=rbms_std_steps, lw=3,
    #      label="RBMS")
    # xlabel!("Misspecification")
    # ylabel!("Number of steps to reach goal")
end
