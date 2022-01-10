function mountaincar_run_online_model_search(rock_c::Float64,
                                             num_eval_samples::Int64;
                                             max_likelihood = false,
                                             optimistic = false,
                                             debug = false,
                                             seed = 0)
    rng = MersenneTwister(seed)
    mountaincar = MountainCar(rock_c)
    model = MountainCar(0.0)
    agent = MountainCarOnlineModelSearchAgent(mountaincar, model,
                                              num_eval_samples)
    n_steps = run(agent, rng; debug = debug, max_likelihood = max_likelihood,
                  optimistic = optimistic,)
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

function mountain_run_online_experiments()
    num_eval_samples = 10
    seeds = collect(1:10)
    ml_steps = []
    oms_steps = []
    cmax_steps = []
    rock_c_values = collect(0.02:0.001:0.03)
    for rock_c in rock_c_values
        ml_sub_steps = []
        oms_sub_steps = []
        cmax_sub_steps = []
        for seed in seeds
            println("MLE with rock_c ", rock_c, " seed ", seed)
            push!(ml_sub_steps, mountaincar_run_online_model_search(
                rock_c, num_eval_samples; max_likelihood = true, seed = seed,
                debug = false,
            ))
            println("OMS with rock_c ", rock_c, " seed ", seed)
            push!(oms_sub_steps, mountaincar_run_online_model_search(
                rock_c, num_eval_samples; optimistic = true, seed = seed,
                debug = false,
            ))
            println("CMAX with rock_c ", rock_c, " seed ", seed)
            push!(cmax_sub_steps, mountaincar_run_cmax(
                rock_c; seed = seed, debug = false,
            ))
        end
        push!(ml_steps, ml_sub_steps)
        push!(oms_steps, oms_sub_steps)
        push!(cmax_steps, cmax_sub_steps)
    end
    ml_mean_steps = [mean(x) for x in ml_steps]
    ml_std_steps = [std(x) for x in ml_steps] ./ sqrt(length(seeds))
    oms_mean_steps = [mean(x) for x in oms_steps]
    oms_std_steps = [std(x) for x in oms_steps] ./ sqrt(length(seeds))
    cmax_mean_steps = [mean(x) for x in cmax_steps]
    cmax_std_steps = [std(x) for x in cmax_steps] ./ sqrt(length(seeds))

    println("MLE")
    println(ml_steps)
    println("OMS")
    println(oms_steps)
    println("CMAX")
    println(cmax_steps)

    plot(rock_c_values, ml_mean_steps, grid=true, yerror=ml_std_steps, lw=3,
         label="MLE")
    plot!(rock_c_values, oms_mean_steps, grid=true, yerror=oms_std_steps, lw=3,
         label="OMS")
    plot!(rock_c_values, cmax_mean_steps, grid=true, yerror=cmax_std_steps, lw=3,
         label="CMAX")
    xlabel!("Misspecification")
    ylabel!("Number of steps to reach goal")
end
