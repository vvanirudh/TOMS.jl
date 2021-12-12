mutable struct GridworldPlanner <: Planner
    gridworld::Gridworld
    num_expansions::Int64
    actions::Vector{GridworldAction}
    residuals::Matrix{Float64}
end

function GridworldPlanner(gridworld::Gridworld, num_expansions::Int64 = 10)
    residuals = fill(0.0, (gridworld.size, gridworld.size))
    GridworldPlanner(gridworld, num_expansions, getActions(gridworld), residuals)
end

function getHeuristic(planner::GridworldPlanner, state::GridworldState)
    manhattan_dist = manhattanHeuristic(planner, state)
    return manhattan_dist + planner.residuals[state.x, state.y]
end

function manhattanHeuristic(planner::GridworldPlanner, state::GridworldState)
    abs(planner.gridworld.goal_state.x - state.x) +
    abs(planner.gridworld.goal_state.y - state.y)
end

function getSuccessors(
    planner::GridworldPlanner,
    state::GridworldState,
    action::GridworldAction,
)
    next_state, cost = step(planner.gridworld, state, action)
    return next_state, cost
end

function checkGoal(planner::GridworldPlanner, state::GridworldState)
    checkGoal(planner.gridworld, state)
end

function getActions(planner::GridworldPlanner, state::GridworldState)
    planner.actions
end

function act(planner::GridworldPlanner, state::GridworldState)
    best_action, info = astar(planner, planner.num_expansions, state)
    return best_action, info
end

function updateResiduals!(planner::GridworldPlanner, info::Dict)
    best_node_f = info["best_node_f"]
    for node in info["closed"]
        state = node.state
        g = node.g
        planner.residuals[state.x, state.y] =
            best_node_f - g - manhattanHeuristic(planner, state)
    end
end
