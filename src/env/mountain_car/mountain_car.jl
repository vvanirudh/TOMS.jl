const car_width = 0.05
const car_height = car_width / 2.0
const clearance = 0.2 * car_height
const flag_height = 0.05

struct MountainCarState <: State
    position::Float64
    speed::Float64
end

function vec(state::MountainCarState)::Array{Float64}
    [state.position, state.speed]
end

struct MountainCarDiscState <: State
    disc_position::Int64
    disc_speed::Int64
end

function vec(disc_state::MountainCarDiscState)::Array{Int64}
    [disc_state.disc_position, disc_state.disc_speed]
end

function unvec(state::Array{Float64}; cont::Bool)
    if cont
        return MountainCarState(state[1], state[2])
    else
        return MountainCarDiscState(state[1], state[2])
    end
end

struct MountainCarAction <: Action
    id::Int64
end

struct MountainCarParameters <: Parameters
    theta1::Float64
    theta2::Float64
end

function vec(params::MountainCarParameters)::Array{Float64}
    [params.theta1, params.theta2]
end

function unvec_params(params::Array{Float64})::MountainCarParameters
    MountainCarParameters(params[1], params[2])
end

struct MountainCarTransition <: Transition
    initial_state::MountainCarDiscState
    action::MountainCarAction
    cost::Float64
    final_state::MountainCarDiscState
end

struct MountainCarContTransition <: Transition
    initial_state::MountainCarState
    action::MountainCarAction
    cost::Float64
    final_state::MountainCarState
end

struct MountainCar
    min_position::Float64
    max_position::Float64
    max_speed::Float64
    goal_position::Float64
    goal_speed::Float64
    force::Float64
    position_discretization::Int64
    speed_discretization::Int64
    position_grid_cell_size::Float64
    speed_grid_cell_size::Float64
    start_state::MountainCarState
    rock_c::Float64
    rock_position::Float64
    noise::Normal
end

function MountainCar(rock_c::Float64; position_sigma::Float64 = 0.0)
    start_state = MountainCarState(-π / 6, 0)
    position_discretization = 300 # 150  # 500 # 150 # 300 # 500
    speed_discretization = 300 # 150  # 250 # 150 # 500 # 250 # new param
    min_position = -1.2
    max_position = 0.5 # 0.6
    max_speed = 0.07  # 0.2
    goal_position = 0.5
    goal_speed = 0
    force = 0.001
    rock_position = 0.25
    position_grid_cell_size = (max_position - min_position) / (position_discretization - 1)
    speed_grid_cell_size = (max_speed * 2) / (speed_discretization - 1)
    noise = Normal(0.0, position_sigma)
    MountainCar(
        min_position,
        max_position,
        max_speed,
        goal_position,
        goal_speed,
        force,
        position_discretization,
        speed_discretization,
        position_grid_cell_size,
        speed_grid_cell_size,
        start_state,
        rock_c,
        rock_position,
        noise,
    )
end

function init(mountaincar::MountainCar; rng = nothing, cont = false)
    start_state = mountaincar.start_state
    if !isnothing(rng)
        random_position = rand(rng, Uniform(mountaincar.min_position, mountaincar.max_position))
        random_speed = rand(rng, Uniform(-mountaincar.max_speed, mountaincar.max_speed))
        start_state = MountainCarState(random_position, random_speed)
    end
    if cont
        return start_state
    end
    cont_state_to_disc(mountaincar, start_state)
end

function getActions(mountaincar::MountainCar)
    [MountainCarAction(0), MountainCarAction(1)]
end

function checkGoal(mountaincar::MountainCar, state::MountainCarState)
    state.position >= mountaincar.goal_position
end

function checkGoal(mountaincar::MountainCar, state::MountainCarDiscState)
    checkGoal(mountaincar, disc_state_to_cont(mountaincar, state))
end

function step(
    mountaincar::MountainCar,
    disc_state::MountainCarDiscState,
    action::MountainCarAction,
    params::Array{Float64};
    debug = false,
    rng = nothing,
)
    step(
        mountaincar,
        disc_state,
        action,
        MountainCarParameters(params[1], params[2]),
        debug = debug,
        rng = rng,
    )
end

function step(
    mountaincar::MountainCar,
    state::MountainCarState,
    action::MountainCarAction,
    params::Array{Float64};
    debug = false,
    rng = nothing,
)
    step(
        mountaincar,
        state,
        action,
        MountainCarParameters(params[1], params[2]),
        debug = debug,
        rng = rng,
    )
end

function step(
    mountaincar::MountainCar,
    disc_state::MountainCarDiscState,
    action::MountainCarAction,
    params::MountainCarParameters;
    debug = false,
    rng = nothing,
)
    state = disc_state_to_cont(mountaincar, disc_state)
    next_slipped_state, cost = step(
        mountaincar, 
        state, 
        action, 
        params;
        debug = debug,
        rng = rng,
    )
    next_slipped_disc_state = cont_state_to_disc(mountaincar, next_slipped_state)
    next_slipped_disc_state, cost
end

function step_discrete(
    mountaincar::MountainCar,
    state::MountainCarState,
    action::MountainCarAction,
    params::MountainCarParameters;
    debug = false,
    rng = nothing,
)
    cost = getCost(mountaincar, state)
    new_position = position_dynamics(mountaincar, state; rng = rng)
    new_speed = speed_dynamics(mountaincar, state, action, params)
    slip = 0

    disc_state = cont_state_to_disc(mountaincar, state)
    disc_rock_position =
        cont_state_to_disc(
            mountaincar,
            MountainCarState(mountaincar.rock_position, 0),
        ).disc_position
    new_disc_position =
        cont_state_to_disc(mountaincar, MountainCarState(new_position, 0)).disc_position
    if mountaincar.rock_c > 0
        # Check for rock in discrete grid to ensure that no discretization errors happen
        if sign(disc_state.disc_position - disc_rock_position) !=
           sign(new_disc_position - disc_rock_position)
            if debug
                println("Hit rock")
            end
            if new_speed > 0
                slip = max(-mountaincar.rock_c, -new_speed)
            else
                slip = min(mountaincar.rock_c, -new_speed)
            end
        end
    end

    slipped_speed = clamp(new_speed + slip, -mountaincar.max_speed, mountaincar.max_speed)
    @assert abs(slipped_speed) <= abs(new_speed)
    MountainCarState(new_position, slipped_speed), cost
end

function step(
    mountaincar::MountainCar,
    state::MountainCarState,
    action::MountainCarAction,
    params::MountainCarParameters;
    debug = false,
    rng = nothing,
)
    cost = getCost(mountaincar, state)
    new_position = position_dynamics(mountaincar, state; rng = rng)
    new_speed = speed_dynamics(mountaincar, state, action, params)
    slip = 0

    if mountaincar.rock_c > 0
        if sign(state.position - mountaincar.rock_position) !=
           sign(new_position - mountaincar.rock_position)
            if debug
                println("Hit rock")
            end
            if new_speed > 0
                slip = max(-mountaincar.rock_c, -new_speed)
            else
                slip = min(mountaincar.rock_c, -new_speed)
            end
        end
    end

    slipped_speed = clamp(new_speed + slip, -mountaincar.max_speed, mountaincar.max_speed)
    @assert abs(slipped_speed) <= abs(new_speed)
    MountainCarState(new_position, slipped_speed), cost
end

function position_dynamics(
    mountaincar::MountainCar, 
    state::MountainCarState;
    rng = nothing,
)
    new_position = state.position + state.speed
    if !isnothing(rng)
        new_position += rand(rng, mountaincar.noise)
    end
    clamp(new_position, mountaincar.min_position, mountaincar.max_position)
end

function speed_dynamics(
    mountaincar::MountainCar,
    state::MountainCarState,
    action::MountainCarAction,
    params::MountainCarParameters,
)
    new_speed =
        state.speed +
        (2 * action.id - 1) * mountaincar.force +
        cos(params.theta2 * state.position) * params.theta1
    # clamp(new_speed, -mountaincar.max_speed, mountaincar.max_speed)
    new_speed
end

function getCost(mountaincar::MountainCar, state::MountainCarState)
    if state.position < mountaincar.goal_position
        # return abs(state.position - mountaincar.goal_position)
        return 1.0
    else
        return 0.0
    end
end

function getCost(mountaincar::MountainCar, disc_state::MountainCarDiscState)
    getCost(mountaincar, disc_state_to_cont(mountaincar, disc_state))
end

function cont_state_to_disc(mountaincar::MountainCar, state::MountainCarState)
    position = state.position - mountaincar.min_position
    speed = state.speed + mountaincar.max_speed
    disc_position = clamp(
        cont_to_disc(position, mountaincar.position_grid_cell_size),
        0,
        mountaincar.position_discretization - 1,
    )
    disc_speed = clamp(
        cont_to_disc(speed, mountaincar.speed_grid_cell_size),
        0,
        mountaincar.speed_discretization - 1,
    )
    MountainCarDiscState(disc_position, disc_speed)
end

function absorbing_state(mountaincar::MountainCar)
    MountainCarDiscState(Inf, Inf)
end

function absorbing_state_idx(mountaincar::MountainCar)
    mountaincar.position_discretization * mountaincar.speed_discretization + 1
end

function disc_state_to_idx(mountaincar::MountainCar, disc_state::MountainCarDiscState)
    disc_position = disc_state.disc_position
    disc_speed = disc_state.disc_speed
    # Adding 1 for julia indexing
    idx = disc_position * mountaincar.speed_discretization + disc_speed + 1
    @assert idx >= 1 &&
            idx <= mountaincar.position_discretization * mountaincar.speed_discretization
    idx
end

function cont_state_to_idx(mountaincar::MountainCar, cont_state::MountainCarState)
    disc_state = cont_state_to_disc(mountaincar, cont_state)
    disc_state_to_idx(mountaincar, disc_state)
end

function idx_to_disc_state(mountaincar::MountainCar, idx::Int64)
    disc_position = (idx - 1) ÷ mountaincar.speed_discretization
    disc_speed = (idx - 1) % mountaincar.speed_discretization
    @assert disc_position >= 0 && disc_position < mountaincar.position_discretization
    @assert disc_speed >= 0 && disc_speed < mountaincar.speed_discretization
    MountainCarDiscState(disc_position, disc_speed)
end

function idx_to_cont_state(mountaincar::MountainCar, idx::Int64)
    disc_state = idx_to_disc_state(mountaincar, idx)
    disc_state_to_cont(mountaincar, disc_state)
end


function cont_to_disc(x::Float64, cell_size::Float64)
    disc_x = floor(Int64, (x + 0.5 * cell_size) / cell_size)
    if x >= 0
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
    position = clamp(
        position + mountaincar.min_position,
        mountaincar.min_position,
        mountaincar.max_position,
    )
    speed =
        clamp(speed - mountaincar.max_speed, -mountaincar.max_speed, mountaincar.max_speed)
    MountainCarState(position, speed)
end

function disc_to_cont(x::Int64, cell_size::Float64)
    x * cell_size
end
