module Dalinar

using Plots
using JLD: save, load

abstract type State end
abstract type Action end
abstract type Planner end
abstract type Parameters end
abstract type Transition end

include("graph/astar.jl")
# --------- gridworld ------------------
include("env/gridworld/gridworld.jl")
include("planner/gridworld/gridworld_planner.jl")
include("agent/gridworld/gridworld_agent.jl")

function gridworld_main()
    gridworld = create_example_gridworld()
    planner = GridworldPlanner(gridworld, 100)
    agent = GridworldAgent(gridworld, planner)
    run(agent)
end

# --------- mountain car ---------------
include("env/mountain_car/mountain_car.jl")
include("planner/mountain_car/mountain_car_rtaa_planner.jl")
include("planner/mountain_car/mountain_car_cmax_planner.jl")
include("agent/mountain_car/mountain_car_rtaa_agent.jl")
include("agent/mountain_car/mountain_car_cmax_agent.jl")
include("agent/mountain_car/mountain_car_finite_model_class_agent.jl")

const true_params = MountainCarParameters(-0.0025, 3)
# const range_of_values = 0:0.005:0.045
const range_of_values = 0.03:0.001:0.045

const theta1_values = -0.0023:-0.00005:-0.0027
const theta2_values = 2.9:0.01:3.1
const gridsearch_data_path = "/Users/avemula1/Developer/TAML/Dalinar/data/gridsearch.jld"

function mountaincar_rtaa_main(;params=true_params)::Vector{Int64}    
    rtaa_steps = []
    model = MountainCar(0.0)
    planner = MountainCarRTAAPlanner(model, 1000, params)
    generateHeuristic!(planner)
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        agent = MountainCarRTAAAgent(mountaincar, planner)
        n_steps = run(agent, max_steps=1e4)
        push!(rtaa_steps, n_steps)
        clearResiduals!(planner)
    end
    rtaa_steps
end

function mountaincar_rtaa_grid_search()
    data = fill(0, (length(theta1_values), length(theta2_values), length(range_of_values)))
    for idx1 in 1:length(theta1_values)
        theta1 = theta1_values[idx1]
        for idx2 in 1:length(theta2_values)
            theta2 = theta2_values[idx2]
            params = MountainCarParameters(theta1, theta2)
            rtaa_steps = mountaincar_rtaa_main(params=params)
            data[idx1, idx2, :] = rtaa_steps
        end
    end
    save(gridsearch_data_path, "data", data, "theta1", collect(theta1_values), "theta2", collect(theta2_values))
end

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

function mountaincar_true_main()
    true_steps = []
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        model = MountainCar(rock_c)
        planner = MountainCarRTAAPlanner(model, 1000, true_params)
        generateHeuristic!(planner)
        agent = MountainCarRTAAAgent(mountaincar, planner)
        n_steps = run(agent, max_steps=1e4)
        push!(true_steps, n_steps)
    end
    true_steps
end

function mountaincar_all_main()
    rtaa_steps = mountaincar_rtaa_main()
    cmax_steps = mountaincar_cmax_main()
    true_steps = mountaincar_true_main()
    finite_model_class_steps = mountaincar_finite_model_class_main()
    true_finite_model_class_steps = mountaincar_finite_model_class_main(true_model=true)
    local_finite_model_class_steps = mountaincar_finite_model_class_main(local_agent=true)
    plot(range_of_values, rtaa_steps, lw=3, label="RTAA*", legend=:topleft)
    plot!(range_of_values, cmax_steps, lw=3, label="CMAX")
    plot!(range_of_values, true_steps, lw=3, label="True")
    plot!(range_of_values, finite_model_class_steps, lw=3, label="Finite Model Class")
    plot!(range_of_values, true_finite_model_class_steps, lw=3, label="Finite Model Class with true model")
    plot!(range_of_values, local_finite_model_class_steps, lw=3, label="Finite Model Class with Local Data")
    xlabel!("Misspecification")
    ylabel!("Number of steps to reach goal")
end


function mountaincar_single_run_main()
    mountaincar = MountainCar(0.045)
    model = MountainCar(0.0)
    planner = MountainCarCMAXPlanner(model, 1000, true_params)
    generateHeuristic!(planner)
    agent = MountainCarCMAXAgent(mountaincar, planner)
    run(agent)
end

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

end