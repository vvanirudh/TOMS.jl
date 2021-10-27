module Dalinar

using Plots

abstract type State end
abstract type Action end
abstract type Planner end
abstract type Parameters end
abstract type Transition end

include("graph/astar.jl")
# --------- gridworld ------------------
include("env/gridworld.jl")
include("planner/gridworld_planner.jl")
include("agent/gridworld_agent.jl")

function gridworld_main()
    gridworld = create_example_gridworld()
    planner = GridworldPlanner(gridworld, 100)
    agent = GridworldAgent(gridworld, planner)
    run(agent)
end

# --------- mountain car ---------------
include("env/mountain_car.jl")
include("planner/mountaincar_planner.jl")
include("agent/mountaincar_agent.jl")

const true_params = MountainCarParameters(-0.0025, 3)
const range_of_values = 0:0.005:0.045

function mountaincar_heuristic()
    rm(heuristic_path, force=true)
    mountaincar = MountainCar(0.0)
    model = MountainCar(0.0)
    planner = MountainCarRTAAPlanner(model, 1000, true_params)
    for _ in 1:200
        agent = MountainCarRTAAAgent(mountaincar, planner)
        run(agent, save_heuristic=true)
    end
end

function mountaincar_rtaa_main()
    rtaa_steps = []
    model = MountainCar(0.0)
    planner = MountainCarRTAAPlanner(model, 1000, true_params)
    generateHeuristic(planner)
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        agent = MountainCarRTAAAgent(mountaincar, planner)
        n_steps = run(agent, max_steps=1e4)
        push!(rtaa_steps, n_steps)
        clearResiduals(planner)
    end
    rtaa_steps
end

function mountaincar_cmax_main()
    cmax_steps = []
    model = MountainCar(0.0)
    cmax_planner = MountainCarCMAXPlanner(model, 1000, true_params)
    generateHeuristic(cmax_planner)
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        cmax_agent = MountainCarCMAXAgent(mountaincar, cmax_planner)
        n_steps = run(cmax_agent, max_steps=1e4)
        push!(cmax_steps, n_steps)
        clearResiduals(cmax_planner)
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
        generateHeuristic(planner)
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
    # finite_model_class_steps = mountaincar_finite_model_class_main()
    yticks = 0:1e2:1e4
    plot(range_of_values, rtaa_steps, lw=3, label="RTAA*", yaxis=:log, yticks=yticks)
    plot!(range_of_values, cmax_steps, lw=3, label="CMAX", yaxis=:log, yticks=yticks)
    plot!(range_of_values, true_steps, lw=3, label="True", yaxis=:log, yticks=yticks)
    # plot!(range_of_values, finite_model_class_steps, lw=3, label="Finite Model Class", yaxis=:log)
    xlabel!("Misspecification")
    ylabel!("Number of steps to reach goal")
end


function mountaincar_single_run_main()
    mountaincar = MountainCar(0.045)
    model = MountainCar(0.045)
    planner = MountainCarRTAAPlanner(model, 1000, true_params)
    generateHeuristic(planner)
    agent = MountainCarRTAAAgent(mountaincar, planner)
    run(agent, debug=false)
end

function mountaincar_finite_model_class_main()
    mountaincar = MountainCar(0.045)
    model = MountainCar(0.0)
    planners = Vector{MountainCarRTAAPlanner}()
    push!(planners, MountainCarRTAAPlanner(model, 1000, true_params))
    push!(planners, MountainCarRTAAPlanner(model, 1000, MountainCarParameters(-0.0025, 3.2)))
    push!(planners, MountainCarRTAAPlanner(model, 1000, MountainCarParameters(-0.0028, 3)))
    push!(planners, MountainCarRTAAPlanner(mountaincar, 1000, true_params))
    generateHeuristic(planners)
    agent = MountainCarFiniteModelClassAgent(mountaincar, planners)
    run(agent, debug=true)
end

end