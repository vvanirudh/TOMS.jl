using JLD: save, load

const heuristic_path = "/Users/avemula1/Developer/TAML/Dalinar/data/heuristic.jld"
const inflated_cost = 1e6

mutable struct MountainCarPlanner <: Planner
    mountaincar::MountainCar
    num_expansions::Int64
    actions::Vector{MountainCarAction}
    residuals::Matrix{Float64}
    heuristic::Matrix{Float64}
    params::MountainCarParameters
end

function MountainCarPlanner(mountaincar::MountainCar, num_expansions::Int64, params::MountainCarParameters)
    heuristic = loadHeuristic(mountaincar)
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

function loadHeuristic(mountaincar::MountainCar)
    heuristic = fill(0.0, (mountaincar.position_discretization, mountaincar.speed_discretization))
    if isfile(heuristic_path)
        heuristic = load(heuristic_path)["heuristic"]
    end
    heuristic
end

mutable struct MountainCarCMAXPlanner <: Planner
    planner::MountainCarPlanner
    discrepancy_sets::Dict{MountainCarAction, Set{MountainCarDiscState}}
end

function MountainCarCMAXPlanner(mountaincar::MountainCar, num_expansions::Int64, params::MountainCarParameters)
    planner = MountainCarPlanner(mountaincar, num_expansions, params)
    discrepancy_sets = Dict{MountainCarAction, Set{MountainCarDiscState}}()
    for action in planner.actions
        discrepancy_sets[action] = Set{MountainCarDiscState}()
    end
    MountainCarCMAXPlanner(planner, discrepancy_sets)
end

function getHeuristic(cmax_planner::MountainCarCMAXPlanner, state::MountainCarDiscState)
    getHeuristic(cmax_planner.planner, state)
end

function getSuccessors(cmax_planner::MountainCarCMAXPlanner, state::MountainCarDiscState, action::MountainCarAction)
    next_disc_state, cost = getSuccessors(cmax_planner.planner, state, action)
    if state ∈ cmax_planner.discrepancy_sets[action]
        cost = inflated_cost
    end
    next_disc_state, cost
end

function checkGoal(cmax_planner::MountainCarCMAXPlanner, state::MountainCarDiscState)
    checkGoal(cmax_planner.planner, state)
end

function getActions(cmax_planner::MountainCarCMAXPlanner, state::MountainCarDiscState)
    getActions(cmax_planner.planner, state)
end

function act(cmax_planner::MountainCarCMAXPlanner, state::MountainCarDiscState)
    best_action, info = astar(cmax_planner, cmax_planner.planner.num_expansions, state)
    return best_action, info
end

function updateResiduals!(cmax_planner::MountainCarCMAXPlanner, info::Dict)
    updateResiduals!(cmax_planner.planner, info)
end

function addDiscrepancy!(cmax_planner::MountainCarCMAXPlanner, state::MountainCarDiscState, action::MountainCarAction)
    push!(cmax_planner.discrepancy_sets[action], state)
end