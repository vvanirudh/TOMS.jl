const data_path = pwd() * "/data/"
const inflated_cost = 1e6
const number_of_runs_to_generate_heuristic = 1000

mutable struct MountainCarRTAAPlanner <: Planner
    mountaincar::MountainCar
    num_expansions::Int64
    actions::Vector{MountainCarAction}
    residuals::Matrix{Float64}
    heuristic::Matrix{Float64}
    params::MountainCarParameters
end

function MountainCarRTAAPlanner(
    mountaincar::MountainCar,
    num_expansions::Int64,
    params::MountainCarParameters,
)
    heuristic =
        fill(0.0, (mountaincar.position_discretization, mountaincar.speed_discretization))
    residuals =
        fill(0.0, (mountaincar.position_discretization, mountaincar.speed_discretization))
    MountainCarRTAAPlanner(
        mountaincar,
        num_expansions,
        getActions(mountaincar),
        residuals,
        heuristic,
        params,
    )
end

function generateHeuristic!(planner::MountainCarRTAAPlanner; max_steps = 1e4)
    heuristic_path = getHeuristicFilePath(planner)
    if isfile(heuristic_path)
        loadHeuristic!(planner)
    else
        println("Generating Heuristic")
        clearHeuristic!(planner)
        for i = 1:number_of_runs_to_generate_heuristic
            state = init(planner.mountaincar; random = true)
            num_steps = 0
            while !checkGoal(planner.mountaincar, state) && num_steps < max_steps
                num_steps += 1
                best_action, info = act(planner, state)
                updateResiduals!(planner, info)
                state, cost = step(planner.mountaincar, state, best_action, planner.params)
            end
        end
        planner.heuristic = deepcopy(planner.residuals)
        clearResiduals!(planner)
        saveHeuristic(planner)
        println("Generated Heuristic")
    end
end

function generateHeuristic!(planners::Vector{MountainCarRTAAPlanner}; max_steps = 1e4)
    for planner in planners
        generateHeuristic!(planner, max_steps = max_steps)
    end
end

function clearHeuristic!(planner::MountainCarRTAAPlanner)
    planner.heuristic = fill(
        0.0,
        (
            planner.mountaincar.position_discretization,
            planner.mountaincar.speed_discretization,
        ),
    )
end

function clearResiduals!(planner::MountainCarRTAAPlanner)
    planner.residuals = fill(
        0.0,
        (
            planner.mountaincar.position_discretization,
            planner.mountaincar.speed_discretization,
        ),
    )
end

function clearResiduals!(planners::Vector{MountainCarRTAAPlanner})
    for planner in planners
        clearResiduals!(planner)
    end
end

function getHeuristic(planner::MountainCarRTAAPlanner, state::MountainCarDiscState)
    heuristic = planner.heuristic[state.disc_position+1, state.disc_speed+1]
    # adding 1 for julia indexing
    residual = planner.residuals[state.disc_position+1, state.disc_speed+1]
    heuristic + residual
end

function getLookaheadValue(planner::MountainCarRTAAPlanner, state::MountainCarDiscState)
    _, info = act(planner, state)
    info["best_node_f"]
end

function getSuccessors(
    planner::MountainCarRTAAPlanner,
    state::MountainCarDiscState,
    action::MountainCarAction,
)
    next_disc_state, cost = step(planner.mountaincar, state, action, planner.params)
    next_disc_state, cost
end

function checkGoal(planner::MountainCarRTAAPlanner, state::MountainCarDiscState)
    checkGoal(planner.mountaincar, state)
end

function getActions(planner::MountainCarRTAAPlanner, state::MountainCarDiscState)
    planner.actions
end

function act(planner::MountainCarRTAAPlanner, state::MountainCarDiscState)
    best_action, info = astar(planner, planner.num_expansions, state)
    return best_action, info
end

function act(
    planner::MountainCarRTAAPlanner,
    state::MountainCarDiscState,
    num_expansions::Int64,
)
    best_action, info = astar(planner, num_expansions, state)
    return best_action, info
end

function updateResiduals!(planner::MountainCarRTAAPlanner, info::Dict)
    best_node_f = info["best_node_f"]
    for node in info["closed"]
        state = node.state
        g = node.g
        h = planner.heuristic[state.disc_position+1, state.disc_speed+1]
        # adding 1 for julia indexing
        planner.residuals[state.disc_position+1, state.disc_speed+1] = best_node_f - g - h
    end
end

function saveHeuristic(planner::MountainCarRTAAPlanner)
    heuristic_path = getHeuristicFilePath(planner)
    save(heuristic_path, "heuristic", planner.heuristic + planner.residuals)
end

function getHeuristicFilePath(planner::MountainCarRTAAPlanner)
    heuristic_file_name =
        "heuristic" *
        "_$(planner.params.theta1)_$(planner.params.theta2)_$(planner.mountaincar.rock_c)" *
        ".jld"
    data_path * heuristic_file_name
end

function loadHeuristic!(planner::MountainCarRTAAPlanner)
    heuristic_path = getHeuristicFilePath(planner)
    if isfile(heuristic_path)
        planner.heuristic = load(heuristic_path)["heuristic"]
    end
end
