mutable struct MountainCarCMAXPlanner <: Planner
    planner::MountainCarRTAAPlanner
    discrepancy_sets::Dict{MountainCarAction,Set{MountainCarDiscState}}
end

function MountainCarCMAXPlanner(
    mountaincar::MountainCar,
    num_expansions::Int64,
    params::MountainCarParameters,
)
    planner = MountainCarRTAAPlanner(mountaincar, num_expansions, params)
    discrepancy_sets = Dict{MountainCarAction,Set{MountainCarDiscState}}()
    for action in planner.actions
        discrepancy_sets[action] = Set{MountainCarDiscState}()
    end
    MountainCarCMAXPlanner(planner, discrepancy_sets)
end

function generateHeuristic!(cmax_planner::MountainCarCMAXPlanner; max_steps = 1e4)
    generateHeuristic!(cmax_planner.planner, max_steps = max_steps)
end

function generateHeuristic!(planners::Vector{MountainCarCMAXPlanner}; max_steps = 1e4)
    for planner in planners
        generateHeuristic!(planner, max_steps = max_steps)
    end
end

function clearResiduals!(cmax_planner::MountainCarCMAXPlanner)
    clearResiduals!(cmax_planner.planner)
end

function clearResiduals!(planners::Vector{MountainCarCMAXPlanner})
    for planner in planners
        clearResiduals!(planner)
    end
end

function getHeuristic(cmax_planner::MountainCarCMAXPlanner, state::MountainCarDiscState)
    getHeuristic(cmax_planner.planner, state)
end

function getSuccessors(
    cmax_planner::MountainCarCMAXPlanner,
    state::MountainCarDiscState,
    action::MountainCarAction,
)
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

function addDiscrepancy!(
    cmax_planner::MountainCarCMAXPlanner,
    state::MountainCarDiscState,
    action::MountainCarAction,
)
    push!(cmax_planner.discrepancy_sets[action], state)
end
