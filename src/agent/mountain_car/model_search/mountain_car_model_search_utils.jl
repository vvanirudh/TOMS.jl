include("mountain_car_model_search_mlp.jl")
include("mountain_car_model_search_data.jl")
include("mountain_car_model_search_least_squares.jl")

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
    threshold::Float64 = 1e-6,
    debug::Bool = false,
)
    step = copy(optimization_params.initial_step_size)
    inputs = []
    outputs = []
    if debug
        println("Evaluating initial params ", optimization_params.initial_params)
    end
    push!(inputs, copy(optimization_params.initial_params))
    push!(outputs, eval_fn(inputs[1]))
    best_params = copy(optimization_params.initial_params)
    if !isnothing(least_squares_params)
        if debug
            println("Evaluating least squares params ", least_squares_params)
        end
        push!(inputs, least_squares_params)
        push!(outputs, eval_fn(inputs[2]))
        if outputs[2] < outputs[1]
            best_params = least_squares_params
        end
    end

    while length(outputs) < optimization_params.maximum_evaluations && maximum(step) > threshold
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
            if debug
                println("Decreasing step size at params ", best_params)
            end
            step = step ./ 2
        else
            if debug
                println("Moving to params ", new_best_params)
            end
            best_params = new_best_params
        end
    end
    if length(outputs) >= optimization_params.maximum_evaluations && debug
        println("Exhausted maximum number of evaluations")
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
    hardcoded::Bool = false,
    max_inflation::Float64 = 2.0,
    scale::Float64 = 50.0,
    debug::Bool = false,
    eval_distance::Bool = false,
)
    x_array_copy = deepcopy(x_array)
    position_range = mountaincar.max_position - mountaincar.min_position
    speed_range = 2 * mountaincar.max_speed
    normalization = permutedims([position_range, speed_range])
    total_return = 0.0
    if hardcoded
        actual_return = 0.0
    end
    if eval_distance
        eval_distances = []
    end
    for i = 1:num_episodes_eval
        x = init(mountaincar; cont = true)
        c = 1.0
        actual_c = 1.0
        for t = 1:horizon
            a = policy[cont_state_to_idx(mountaincar, x)]
            total_return += c
            distances = distance_fn(vec(x), x_array_copy[a], normalization)
            manual_data_index = argmin(distances)
            distance = 0.0
            if hardcoded
                actual_return += actual_c
                distance = distances[manual_data_index]
                actual_c = cost_array[a][manual_data_index]
            end
            inflation = min(1 + scale * distance, max_inflation)
            x_array_copy[a][manual_data_index, :] = [Inf, Inf]
            x = unvec(xnext_array[a][manual_data_index]; cont = true)
            c = inflation * cost_array[a][manual_data_index]
            if checkGoal(mountaincar, x)
                break
            end
            if eval_distance
                push!(eval_distances, distances[manual_data_index])
            end
        end
    end
    avg_return = total_return / num_episodes_eval
    if debug
        println("MFMC return computed as ", avg_return)
        if hardcoded
            println("Uninflated MFMC return computed as ", actual_return / num_episodes_eval)
        end
    end
    if eval_distance
        return avg_return, eval_distances
    end
    avg_return
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
    debug::Bool = false,
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
    if debug
        println("Return in model is ", model_return)
    end
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
            xnext = unvec(xnext_array[a][manual_data_index]; cont = true)
            xprednext, _ = step(mountaincar, x, action, params)
            # ABSOLUTE MODEL ADVANTAGE
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
    if debug
        println("Bellman error is ", bellman_error)
        println("Bellman evaluation computed as ", model_return + bellman_error)
    end
    model_return + bellman_error
end

function distance_fn(
    x::Array{Float64},
    xs::Matrix{Float64},
    normalization::Matrix{Float64}
)::Array{Float64}
    x_row = permutedims(x)
    sum(abs.((x_row .- xs) ./ normalization), dims=2)[:, 1]
end
