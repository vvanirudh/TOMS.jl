module Dalinar

using Plots

abstract type State end
abstract type Action end
abstract type Planner end
abstract type Parameters end

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

const range_of_values = range(0.0, stop=0.1, length=20)

function mountaincar_heuristic()
    rm(heuristic_path, force=true)
    mountaincar = MountainCar(0.0)
    model = MountainCar(0.0)
    planner = MountainCarPlanner(model, 1000, model.true_params)
    for _ in 1:200
        agent = MountainCarRTAAAgent(mountaincar, planner)
        run(agent, save_heuristic=true)
    end
end

function mountaincar_rtaa_main()
    rtaa_steps = []
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        model = MountainCar(0.0)
        planner = MountainCarPlanner(model, 100, model.true_params)
        agent = MountainCarRTAAAgent(mountaincar, planner)
        n_steps = run(agent)
        push!(rtaa_steps, n_steps)
    end
    rtaa_steps
end

function mountaincar_cmax_main()
    cmax_steps = []
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        model = MountainCar(0.0)
        cmax_planner = MountainCarCMAXPlanner(model, 100, model.true_params)
        cmax_agent = MountainCarCMAXAgent(mountaincar, cmax_planner)
        n_steps = run(cmax_agent)
        push!(cmax_steps, n_steps)
    end
    cmax_steps
end

function mountaincar_true_main()
    true_steps = []
    for rock_c in range_of_values
        println("Rock_c is ", rock_c)
        mountaincar = MountainCar(rock_c)
        model = MountainCar(rock_c)
        planner = MountainCarPlanner(model, 100, model.true_params)
        agent = MountainCarRTAAAgent(mountaincar, planner)
        n_steps = run(agent)
        push!(true_steps, n_steps)
    end
    true_steps
end

function mountaincar_all_main()
    rtaa_steps = mountaincar_rtaa_main()
    cmax_steps = mountaincar_cmax_main()
    true_steps = mountaincar_true_main()
    plot(range_of_values, rtaa_steps, lw=3, label="RTAA*", yaxis=:log)
    plot!(range_of_values, cmax_steps, lw=3, label="CMAX", yaxis=:log)
    plot!(range_of_values, true_steps, lw=3, label="True", yaxis=:log)
    xlabel!("Misspecification")
    ylabel!("Number of steps to reach goal")
end


function mountaincar_single_run_main()
    mountaincar = MountainCar(0.1)
    model = MountainCar(0.1)
    planner = MountainCarPlanner(model, 100, model.true_params)
    agent = MountainCarRTAAAgent(mountaincar, planner)
    run(agent, debug=true)
end

end