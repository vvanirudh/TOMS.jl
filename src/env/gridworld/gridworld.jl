struct GridworldState <: State
    x::Int64
    y::Int64
end

struct GridworldAction <: Action
    id::Int64
end

struct Gridworld
    size::Int64
    grid::Matrix{Int64}
    render::Bool
    start_state::GridworldState
    goal_state::GridworldState
end

function Gridworld(
    grid::Matrix{Int64},
    render::Bool,
    start_state::GridworldState,
    goal_state::GridworldState,
)
    gridsize = size(grid, 1)
    Gridworld(gridsize, grid, render, start_state, goal_state)
end

function init(gridworld::Gridworld)
    return gridworld.start_state
end

function getDisplacement(
    gridworld::Gridworld,
    state::GridworldState,
    action::GridworldAction,
)
    x, y = state.x, state.y
    if gridworld.grid[x, y] == 0
        # Free state
        if action.id == 1
            return (-1, 0)
        elseif action.id == 2
            return (0, 1)
        elseif action.id == 3
            return (0, -1)
        elseif action.id == 4
            return (1, 0)
        else
            error("Invalid action")
        end
    elseif gridworld.grid[x, y] == 2
        # slip state
        if action.id == 1
            return (1, 0)
        elseif action.id == 2
            return (0, 1)
        elseif action.id == 3
            return (0, -1)
        elseif action.id == 4
            return (-1, 0)
        else
            error("Invalid action")
        end
    else
        error("agent is inside obstacle")
    end
end

function getActions(gridworld::Gridworld)
    return [GridworldAction(1), GridworldAction(2), GridworldAction(3), GridworldAction(4)]
end

function checkGoal(gridworld::Gridworld, state::GridworldState)
    return state.x == gridworld.goal_state.x && state.y == gridworld.goal_state.y
end

function step(gridworld::Gridworld, state::GridworldState, action::GridworldAction)
    displacement = getDisplacement(gridworld, state, action)
    next_state = GridworldState(state.x + displacement[1], state.y + displacement[2])
    if outOfBounds(gridworld, next_state) || inCollision(gridworld, next_state)
        # No displacement
        next_state = state
    end
    cost = getCost(gridworld, next_state)
    return next_state, cost
end

function outOfBounds(gridworld::Gridworld, state::GridworldState)
    state.x < 1 || state.y < 1 || state.x > gridworld.size || state.y > gridworld.size
end

function inCollision(gridworld::Gridworld, state::GridworldState)
    return gridworld.grid[state.x, state.y] == 1
end

function getCost(gridworld::Gridworld, state::GridworldState)
    if checkGoal(gridworld, state)
        return 0
    end
    return 1
end

function render(gridworld::Gridworld, state::GridworldState)
    if gridworld.render
        current_grid = fill("x", (gridworld.size, gridworld.size))
        for i = 1:gridworld.size
            for j = 1:gridworld.size
                if i == state.x && j == state.y
                    current_grid[i, j] = "A"
                elseif i == gridworld.goal_state.x && j == gridworld.goal_state.y
                    current_grid[i, j] = "G"
                else
                    current_grid[i, j] = string(gridworld.grid[i, j])
                end
            end
        end
        display(current_grid)
    end
end

# ----------------------------------------------- #
function create_example_gridworld()
    size = 1000
    grid = fill(0, (size, size))
    grid[Int(floor(size / 4)):size, Int(floor(size / 2))] .= 1
    grid[Int(floor(size / 2)), Int(floor(size / 2))+2:end] .= 1

    start_state = GridworldState(1, 1)
    goal_state = GridworldState(size, size)

    Gridworld(grid, false, start_state, goal_state)
end
