using JLD: save, load

const heuristic_path = path = "/Users/avemula1/Developer/TAML/Dalinar/data/heuristic.jld"

mutable struct MountainCarPlanner <: Planner
    mountaincar::MountainCar
    num_expansions::Int64
    actions::Vector{MountainCarAction}
    residuals::Matrix{Float64}
    heuristic::Matrix{Float64}
    params::MountainCarParameters
end

function MountainCarPlanner(mountaincar::MountainCar, num_expansions::Int64, params::MountainCarParameters)
    heuristic = loadHeuristic(mountaincar.position_discretization, mountaincar.speed_discretization)
    residuals = fill(0.0, (mountaincar.position_discretization, mountaincar.speed_discretization))
    MountainCarPlanner(mountaincar, num_expansions, getActions(mountaincar), residuals, heuristic, params)
end

function getHeuristic(planner::MountainCarPlanner, state::MountainCarDiscState)
    heuristic = planner.heuristic[state.disc_position+1, state.disc_speed+1]
    # adding 1 for julia indexing
    residual = planner.residuals[state.disc_position+1, state.disc_speed+1]
    heuristic+residual
end

function getSuccessors(planner::MountainCarPlanner, state::MountainCarDiscState, action::MountainCarAction)
    next_disc_state, cost = step(planner.mountaincar, state, action, planner.params)
    next_disc_state, cost
end

function checkGoal(planner::MountainCarPlanner, state::MountainCarDiscState)
    checkGoal(planner.mountaincar, state)
end

function getActions(planner::MountainCarPlanner, state::MountainCarDiscState)
    planner.actions
end

function act(planner::MountainCarPlanner, state::MountainCarDiscState)
    best_action, info = astar(planner, planner.num_expansions, state)
    return best_action, info
end

function updateResiduals!(planner::MountainCarPlanner, info::Dict)
    best_node_f = info["best_node_f"]
    for node in info["closed"]
        state = node.state
        g = node.g
        h = planner.heuristic[state.disc_position+1, state.disc_speed+1]
        # adding 1 for julia indexing
        planner.residuals[state.disc_position+1, state.disc_speed+1] = best_node_f - g - h
    end
end

function saveHeuristic(planner::MountainCarPlanner)
    save(heuristic_path, "heuristic", planner.heuristic + planner.residuals)
end

function loadHeuristic(pd::Int64, sd::Int64)
    heuristic = fill(0.0, (pd, sd))
    if isfile(heuristic_path)
        heuristic = load(heuristic_path)["heuristic"]
    end
    heuristic
end