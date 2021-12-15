scipy_optimize = pyimport("scipy.optimize")

function generate_batch_data(
    mountaincar::MountainCar,
    params::MountainCarParameters,
    num_episodes::Int64,
    horizon::Int64;
    policy = nothing,
)::Array{MountainCarContTransition}
    data = []
    # TODO: Can be parallelized
    for episode = 1:num_episodes
        data = vcat(data, simulate_episode(mountaincar, params, horizon, policy = policy))
    end
    data
end

function simulate_episode(
    mountaincar::MountainCar,
    params::MountainCarParameters,
    horizon::Int64;
    policy = nothing,
)
    episode_data = []
    cont_state = init(mountaincar, cont = true)
    if isnothing(policy)
        policy = random_policy(mountaincar)
        cont_state = init(mountaincar, random = true, cont = true)
    end
    actions = getActions(mountaincar)
    for t = 1:horizon-1
        disc_state = cont_state_to_disc(mountaincar, cont_state)
        s = disc_state_to_idx(mountaincar, disc_state)
        u = policy[s]
        action = actions[u]
        cont_state_next, cost = step(mountaincar, cont_state, action, params)
        push!(
            episode_data,
            MountainCarContTransition(cont_state, action, cost, cont_state_next),
        )
        if checkGoal(mountaincar, cont_state_next)
            break
        end
        cont_state = cont_state_next
    end
    episode_data
end

function fit_kdtree(mountaincar::MountainCar, data::Array{MountainCarContTransition})
    Δposition = (mountaincar.max_position - mountaincar.min_position)
    Δspeed = 2 * mountaincar.max_speed
    data_matrix = zeros(2, length(data))
    for idx = 1:length(data)
        transition = data[idx]
        data_matrix[:, idx] = [
            transition.initial_state.position / Δposition,
            transition.initial_state.speed / Δspeed,
        ]
    end
    println("Creating kdtree with generated data")
    KDTree(data_matrix)
end

function random_policy(mountaincar::MountainCar)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization
    n_actions = 2
    rand(1:n_actions, n_states)
end

function prediction_error(
    mountaincar::MountainCar,
    params::Array{Float64},
    data::Array{MountainCarContTransition},
)
    err = []
    for i = 1:length(data)
        transition = data[i]
        predicted_state =
            step(mountaincar, transition.initial_state, transition.action, params)[1]
        push!(err, norm(vec(predicted_state) - vec(transition.final_state)))
    end
    err
end

function get_least_squares_fit(
    mountaincar::MountainCar,
    params::MountainCarParameters,
    data::Array{MountainCarContTransition},
)
    fn(p) = prediction_error(mountaincar, p, data)
    least_squares_params = scipy_optimize.leastsq(fn, vec(params))[1]
    println(
        "Least squares fit gives ",
        least_squares_params[1],
        " ",
        least_squares_params[2],
    )
    least_squares_params
end

struct MountainCarOptimizationParameters
    initial_step_size::Array{Float64}
    initial_params::Array{Float64}
    maximum_evaluations::Int64
    only_positive::Bool
end

function hill_climb(
    eval_fn::Function,
    optimization_params::MountainCarOptimizationParameters;
    least_squares_params::Array{Float64} = nothing,
)
    step = copy(optimization_params.initial_step_size)
    inputs = []
    outputs = []
    println("Evaluating initial params ", optimization_params.initial_params)
    push!(inputs, copy(optimization_params.initial_params))
    push!(outputs, eval_fn(inputs[1]))
    best_params = copy(optimization_params.initial_params)
    if !isnothing(least_squares_params)
        println("Evaluating least squares params ", least_squares_params)
        push!(inputs, least_squares_params)
        push!(outputs, eval_fn(inputs[2]))
        if outputs[2] < outputs[1]
            best_params = least_squares_params
        end
    end

    while length(outputs) < optimization_params.maximum_evaluations && maximum(step) > 1e-6
        new_inputs = []
        for i = 1:length(step)
            if optimization_params.only_positive
                push!(
                    new_inputs,
                    [
                        max.(best_params[i] - step[i], 1e-6),
                        max.(best_params[i], 1e-6),
                        max.(best_params[i] + step[i], 1e-6),
                    ],
                )
            else
                push!(
                    new_inputs,
                    [best_params[i] - step[i], best_params[i], best_params[i] + step[i]],
                )
            end
        end

        new_inputs = product(new_inputs[1], new_inputs[2])
        new_inputs = [[x for x in input] for input in new_inputs]
        # TODO: Can be parallelized
        # new_outputs = [eval_fn(input) for input in new_inputs]
        new_outputs = ThreadsX.map(eval_fn, new_inputs)
        for input in new_inputs
            push!(inputs, input)
        end
        for output in new_outputs
            push!(outputs, output)
        end

        new_best_params = inputs[argmin(outputs)]
        if new_best_params == best_params
            println("Decreasing step size")
            step = step ./ 2
        else
            println("Moving to params ", new_best_params)
            best_params = new_best_params
        end
    end
    inputs[argmin(outputs)]
end

function preprocess_data(mountaincar::MountainCar, data::Array{MountainCarContTransition})
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization
    n_actions = 2
    actions = getActions(mountaincar)
    x_array::Array{Array{Array{Float64}}} = []
    xnext_array::Array{Array{MountainCarState}} = []
    cost_array::Array{Array{Float64}} = []
    for a = 1:n_actions
        push!(x_array, [])
        push!(xnext_array, MountainCarState[])
        push!(cost_array, Float64[])
    end

    for i = 1:length(data)
        transition = data[i]
        a = transition.action.id + 1
        push!(x_array[a], vec(transition.initial_state))
        push!(xnext_array[a], transition.final_state)
        push!(cost_array[a], transition.cost)
    end
    x_array, xnext_array, cost_array
end

function mfmc_evaluation(
    mountaincar::MountainCar,
    policy::Array{Int64},
    horizon::Int64,
    x_array::Array{Array{Array{Float64}}},
    xnext_array::Array{Array{MountainCarState}},
    cost_array::Array{Array{Float64}},
    num_episodes_eval::Int64,
)
    x_array_copy = deepcopy(x_array)
    position_range = mountaincar.max_position - mountaincar.min_position
    speed_range = 2 * mountaincar.max_speed
    total_return = 0.0
    for i = 1:num_episodes_eval
        x = init(mountaincar, cont = true)
        c = 1.0
        for t = 1:horizon
            a = policy[cont_state_to_idx(mountaincar, x)]
            total_return += c
            manual_data_index =
                argmin(distance_fn(vec(x), x_array_copy[a], position_range, speed_range))
            x_array_copy[a][manual_data_index] = [Inf, Inf]
            x = xnext_array[a][manual_data_index]
            c = cost_array[a][manual_data_index]
            if checkGoal(mountaincar, x)
                break
            end
        end
    end
    avg_return = total_return / num_episodes_eval
    println("MFMC return computed as ", avg_return)
    return avg_return
end


function bellman_evaluation(
    mountaincar::MountainCar,
    params::Array{Float64},
    policy::Array{Int64},
    values::Array{Float64},
    horizon::Int64,
    x_array::Array{Array{Array{Float64}}},
    xnext_array::Array{Array{MountainCarState}},
    cost_array::Array{Array{Float64}},
    num_episodes_eval::Int64;
    gamma::Float64 = 0.99,
)
    # Evaluate return in the model
    model_return = 0.0
    x = init(mountaincar, cont = true)
    actions = getActions(mountaincar)
    for t = 1:horizon
        a = policy[cont_state_to_idx(mountaincar, x)]
        action = actions[a]
        x, c = step(mountaincar, x, action, params)
        model_return += c
        if checkGoal(mountaincar, x)
            break
        end
    end
    # Evaluate bellman error
    bellman_error = 0.0
    x_array_copy = deepcopy(x_array)
    position_range = mountaincar.max_position - mountaincar.min_position
    speed_range = 2 * mountaincar.max_speed
    for i = 1:num_episodes_eval
        x = init(mountaincar, cont = true)
        for t = 1:horizon
            a = policy[cont_state_to_idx(mountaincar, x)]
            action = actions[a]
            manual_data_index =
                argmin(distance_fn(vec(x), x_array_copy[a], position_range, speed_range))
            x_array_copy[a][manual_data_index] = [Inf, Inf]
            xnext = xnext_array[a][manual_data_index]
            xprednext, _ = step(mountaincar, x, action, params)
            # TODO: Any effect of gamma here? Since values are computing using gamma
            bellman_error += abs(
                values[cont_state_to_idx(mountaincar, xnext)] -
                values[cont_state_to_idx(mountaincar, xprednext)],
            )
            x = xnext
            if checkGoal(mountaincar, x)
                break
            end
        end
    end
    bellman_error = bellman_error / num_episodes_eval
    println("Bellman evaluation computed as ", model_return + bellman_error)
    model_return + bellman_error
end

function distance_fn(
    x::Array{Float64},
    xs::Array{Array{Float64}},
    position_range::Float64,
    speed_range::Float64,
)::Array{Float64}
    # TODO: Can be parallelized
    [distance_fn(x, x_other, position_range, speed_range) for x_other in xs]
    # ThreadsX.map(x_other -> distance_fn(x, x_other, position_range, speed_range), xs)
end


function distance_fn(
    x::Array{Float64},
    x_other::Array{Float64},
    position_range::Float64,
    speed_range::Float64,
)::Float64
    position_distance = (x[1] - x_other[1]) / position_range
    speed_distance = (x[2] - x_other[2]) / speed_range
    position_distance^2 + speed_distance^2
end


function get_nearest_data_index(
    x::MountainCarState,
    tree::KDTree,
    used_transitions::Set{Int64};
    query_num::Int64 = 10,
)
    println("NOT WORKING. NEED TO DEBUG")
    multiplier = 1
    while true
        idxs, _ = knn(tree, [x.position, x.speed], multiplier * query_num, true)
        for idx in idxs[(multiplier-1)*query_num+1:multiplier*query_num]
            if idx ∈ used_transitions
                continue
            else
                return idx
            end
        end
        multiplier += 1
    end
end
