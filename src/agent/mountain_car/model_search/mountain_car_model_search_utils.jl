include("mountain_car_model_search_mlp.jl")
scipy_optimize = pyimport("scipy.optimize")

struct MountainCarOptimizationParameters
    initial_step_size::Array{Float64}
    initial_params::Array{Float64}
    maximum_evaluations::Int64
    only_positive::Bool
end

function generate_batch_data(
    mountaincar::MountainCar,
    params::MountainCarParameters,
    num_episodes::Int64,
    horizon::Int64;
    policy = nothing,
)::Array{MountainCarContTransition}
    data = @distributed (vcat) for episode = 1:num_episodes
        simulate_episode(mountaincar, params, horizon, policy=policy)
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

function random_policy(mountaincar::MountainCar)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization
    n_actions = 2
    rand(1:n_actions, n_states)
end

function preprocess_data(mountaincar::MountainCar, data::Array{MountainCarContTransition})
    n_actions = 2
    x_array::Array{Array{Array{Float64}}} = []
    x_next_array::Array{Array{Array{Float64}}} = []
    disp_array::Array{Array{Array{Float64}}} = []
    cost_array::Array{Array{Float64}} = []
    for a = 1:n_actions
        push!(x_array, [])
        push!(x_next_array, [])
        push!(disp_array, [])
        push!(cost_array, [])
    end

    for i = 1:length(data)
        transition = data[i]
        a = transition.action.id + 1
        push!(x_array[a], vec(transition.initial_state))
        push!(x_next_array[a], vec(transition.final_state))
        push!(disp_array[a], vec(transition.final_state) - vec(transition.initial_state))
        push!(cost_array[a], transition.cost)
    end
    x_array_matrices = [permutedims(hcat(x_subarray...)) for x_subarray in x_array]
    x_array_matrices, x_array, x_next_array, disp_array, cost_array
end

function fit_ensembles(
    x_array::Array{Array{Array{Float64}}},
    disp_array::Array{Array{Array{Float64}}}, 
)
    ensembles = []
    n_actions = length(x_array)
    for a=1:n_actions
        println("Fitting ensembles for action ", a)
        push!(ensembles, fit_ensemble(x_array[a], disp_array[a]))
    end
    ensembles
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
    scipy_optimize.leastsq(fn, vec(params))[1]
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
        new_outputs = pmap(eval_fn, new_inputs)
        for input in new_inputs
            push!(inputs, input)
        end
        for output in new_outputs
            push!(outputs, output)
        end

        new_best_params = inputs[argmin(outputs)]
        if new_best_params == best_params
            println("Decreasing step size at params ", best_params)
            step = step ./ 2
        else
            println("Moving to params ", new_best_params)
            best_params = new_best_params
        end
    end
    inputs[argmin(outputs)]
end

function mfmc_evaluation(
    mountaincar::MountainCar,
    policy::Array{Int64},
    horizon::Int64,
    x_array::Array{Matrix{Float64}},
    xnext_array::Array{Array{Array{Float64}}},
    cost_array::Array{Array{Float64}},
    num_episodes_eval::Int64;
    ensembles = nothing,
    max_inflation = 2.0,
    scale = 10.0,
)
    x_array_copy = deepcopy(x_array)
    position_range = mountaincar.max_position - mountaincar.min_position
    speed_range = 2 * mountaincar.max_speed
    normalization = permutedims([position_range, speed_range])
    total_return = 0.0
    for i = 1:num_episodes_eval
        x = init(mountaincar, cont = true)
        c = 1.0
        for t = 1:horizon
            a = policy[cont_state_to_idx(mountaincar, x)]
            total_return += c
            manual_data_index =
                argmin(distance_fn(vec(x), x_array_copy[a], normalization))
            x_array_copy[a][manual_data_index, :] = [Inf, Inf]
            max_distance = 0.0
            if !isnothing(ensembles)
                predictions = predict_ensemble(ensembles[a], vec(x))
                max_distance = find_max_distance(predictions)
            end
            # println("Distance is ", max_distance)
            inflation = min(1 + scale * max_distance, max_inflation)
            x = unvec(xnext_array[a][manual_data_index], cont = true)
            c = inflation * cost_array[a][manual_data_index]
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
    x_array::Array{Matrix{Float64}},
    xnext_array::Array{Array{Array{Float64}}},
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
        model_return += gamma^(t-1) * c
        if checkGoal(mountaincar, x)
            break
        end
    end
    println("Return in model is ", model_return)
    # Evaluate bellman error
    bellman_error = 0.0
    x_array_copy = deepcopy(x_array)
    position_range = mountaincar.max_position - mountaincar.min_position
    speed_range = 2 * mountaincar.max_speed
    normalization = permutedims([position_range, speed_range])
    for i = 1:num_episodes_eval
        x = init(mountaincar, cont = true)
        for t = 1:horizon
            a = policy[cont_state_to_idx(mountaincar, x)]
            action = actions[a]
            manual_data_index =
                argmin(distance_fn(vec(x), x_array_copy[a], normalization))
            x_array_copy[a][manual_data_index, :] = [Inf, Inf]
            xnext = unvec(xnext_array[a][manual_data_index], cont = true)
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
    println("Bellman error is ", bellman_error)
    println("Bellman evaluation computed as ", model_return + bellman_error)
    model_return + bellman_error
end

function distance_fn(
    x::Array{Float64},
    xs::Matrix{Float64},
    normalization::Matrix{Float64}
)::Array{Float64}
    x_row = permutedims(x)
    sum(((x_row .- xs) ./ normalization).^2, dims=2)[:, 1]
end