include("params.jl")
include("run_rtaa.jl")
include("run_cmax.jl")
include("run_finite_model_class.jl")
include("run_model_search.jl")

function mountaincar_all_main()
    rtaa_steps = mountaincar_rtaa_main()
    cmax_steps = mountaincar_cmax_main()
    true_steps = mountaincar_true_main()
    # finite_model_class_steps = mountaincar_finite_model_class_main()
    # true_finite_model_class_steps = mountaincar_finite_model_class_main(true_model = true)
    # local_finite_model_class_steps = mountaincar_finite_model_class_main(local_agent = true)
    model_search_steps = mountaincar_model_search_main()


    plot(range_of_values, rtaa_steps, lw = 3, label = "RTAA*", legend = :topleft)
    plot!(range_of_values, cmax_steps, lw = 3, label = "CMAX")
    plot!(range_of_values, true_steps, lw = 3, label = "True")
    # plot!(range_of_values, finite_model_class_steps, lw = 3, label = "Finite Model Class")
    # plot!(
    #     range_of_values,
    #     true_finite_model_class_steps,
    #     lw = 3,
    #     label = "Finite Model Class with true model",
    # )
    # plot!(
    #     range_of_values,
    #     local_finite_model_class_steps,
    #     lw = 3,
    #     label = "Finite Model Class with Local Data",
    # )
    plot!(range_of_values, model_search_steps, lw = 3, label = "RBMS")
    xlabel!("Misspecification")
    ylabel!("Number of steps to reach goal")
end
