struct MountainCarState <: State
    position::Float64
    speed::Float64
end

struct MountainCarDiscState <: State 
    disc_position::Int64
    disc_speed::Int64
end

struct MountainCarAction <: Action
    id::Int64
end

struct MountainCarParameters <: Parameters 
    theta1::Float64
    theta2::Float64
end

struct MountainCar
    min_position::Float64
    max_position::Float64
    max_speed::Float64
    goal_position::Float64
    goal_speed::Float64
    true_params::MountainCarParameters
    force::Float64
    position_discretization::Float64
    speed_discretization::Float64
    position_grid_cell_size::Float64
    speed_grid_cell_size::Float64
    start_state::MountainCarState
    rock_c::Float64
    rock_position::Float64
end

function MountainCar(rock_c::Float64)
    true_params = MountainCarParameters(-0.0025, 3)
    start_state = MountainCarState(-0.5, 0)
    position_discretization = 500
    speed_discretization = 250
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5
    goal_speed = 0
    force = 0.001
    rock_position = 0.25
    position_grid_cell_size = (max_position - min_position) / position_discretization
    speed_grid_cell_size = (max_speed * 2) / speed_discretization
    MountainCar(min_position, max_position, max_speed, goal_position,
        goal_speed, true_params, force, position_discretization,
        speed_discretization, position_grid_cell_size,
        speed_grid_cell_size, start_state, 
        rock_c, rock_position)
end

function init(mountaincar::MountainCar)
    mountaincar.start_state
end

function getActions(mountaincar::MountainCar)
    [MountainCarAction(0), MountainCarAction(1)]
end

function checkGoal(mountaincar::MountainCar, state::MountainCarState)
    state.position >= mountaincar.goal_position && state.speed >= mountaincar.goal_speed
end

function step(mountaincar::MountainCar, disc_state::MountainCarDiscState, action::MountainCarAction, params::MountainCarParameters)
    state = disc_state_to_cont(mountaincar, disc_state)
    next_disc_state, cost = getSuccessor(mountaincar, disc_state, action, params)
    next_state = disc_state_to_cont(mountaincar, next_disc_state)

    slip = 0
    if mountaincar.rock_c != 0
        if sign(state.position - mountaincar.rock_position) != sign(next_state.position - mountaincar.rock_position)
            if state.speed > 0
                slip = max(-mountaincar.rock_c, -state.speed)
            else
                slip = min(mountaincar.rock_c, state.speed)
            end
        end
    end
    next_state.speed += slip

    next_disc_state = cont_state_to_disc(mountaincar, next_state)
    return next_disc_state, cost
end

function getSuccessor(mountaincar::MountainCar, disc_state::MountainCarDiscState, action::MountainCarAction, params::MountainCarParameters)
    state = disc_state_to_cont(mountaincar, disc_state)
    # IF already at goal; return absorbing state
    if checkGoal(mountaincar, state)
        return absorbing_state(mountaincar), 0.0
    end

    position = position_dynamics(mountaincar, state)
    speed = speed_dynamics(mountaincar, state, action, params)
    cost = getCost(mountaincar, state)
    new_state = MountainCarState(position, speed)
    new_disc_state = cont_state_to_disc(mountaincar, new_state)
    return new_disc_state, cost
end

function position_dynamics(mountaincar::MountainCar, state::MountainCarState)
    new_position = state.position + state.speed
    clamp(new_position, mountaincar.min_position, mountaincar.max_position)
end

function speed_dynamics(mountaincar::MountainCar, state::MountainCarState, action::MountainCarAction, params::MountainCarParameters)
    new_speed = speed + (2*action.id - 1) * mountaincar.force + cos(params.theta2 * state.position) * params.theta1
    clamp(new_speed, -mountaincar.max_speed, mountaincar.max_speed)
end

function getCost(mountaincar::MountainCar, state::MountainCarState)
    if state.position < mountaincar.goal_position
        return 1.0
    else
        return 0.0
    end
end

function cont_state_to_disc(mountaincar::MountainCar, state::MountainCarState)
    position = state.position - mountaincar.min_position
    speed = state.speed + mountaincar.max_speed
    disc_position = max(min(cont_to_disc(position, mountaincar.position_grid_cell_size),
        mountaincar.position_discretization-1), 0)
    disc_speed = max(min(cont_to_disc(speed, mountaincar.speed_grid_cell_size),
        mountaincar.speed_discretization-1), 0)
    MountainCarDiscState(disc_position, disc_speed)
end

function absorbing_state(mountaincar::MountainCar)
    MountainCarDiscState(mountaincar.position_discretization-1, mountaincar.speed_discretization-1)
end

function cont_to_disc(x::Float64, cell_size::Float64)
    disc_x = Int64((x + 0.5 * cell_size) / cell_size)
    if x >=0
        return disc_x
    else
        return disc_x - 1
    end
end

function disc_state_to_cont(mountaincar::MountainCar, state::MountainCarDiscState)
    disc_position = state.disc_position
    disc_speed = state.disc_speed
    position = disc_to_cont(disc_position, mountaincar.position_grid_cell_size)
    speed = disc_to_cont(disc_speed, mountaincar.speed_grid_cell_size)
    position = max(min(position + mountaincar.min_position, mountaincar.max_position),
        mountaincar.min_position)
    speed = max(min(speed - mountaincar.max_speed, mountaincar.max_speed),
        -mountaincar.max_speed)
    MountainCarState(position, speed)
end

function disc_to_cont(x::Float64, cell_size::Float64)
    x * cell_size
end