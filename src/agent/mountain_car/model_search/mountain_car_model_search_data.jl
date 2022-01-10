function generate_batch_data(
    mountaincar::MountainCar,
    params::MountainCarParameters,
    num_episodes::Int64,
    horizon::Int64;
    policy = nothing,
    rng = nothing,
)::Array{MountainCarContTransition}
    data = []
    for episode = 1:num_episodes
        data = vcat(data, simulate_episode(mountaincar, params, horizon; policy=policy, rng = rng))
    end
    data = vcat(data, simulate_episode(mountaincar, params, horizon; policy = good_policy(mountaincar), rng = rng))
    data
end

function simulate_episode(
    mountaincar::MountainCar,
    params::MountainCarParameters,
    horizon::Int64;
    policy = nothing,
    rng = nothing,
)::Array{MountainCarContTransition}
    episode_data = []
    cont_state = init(mountaincar; cont = true)
    if isnothing(policy)
        policy = random_policy(mountaincar; rng = rng)
        cont_state = init(mountaincar; rng = rng, cont = true)
    end
    actions = getActions(mountaincar)
    for t = 1:horizon
        action = actions[policy[cont_state_to_idx(mountaincar, cont_state)]]
        cont_state_next, cost = step(mountaincar, cont_state, action, params; rng = rng)
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

function random_policy(mountaincar::MountainCar; rng = nothing)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization
    n_actions = 2
    if isnothing(rng)
        rand(1:n_actions, n_states)
    else
        rand(rng, 1:n_actions, n_states)
    end
end

function good_policy(mountaincar::MountainCar)
    value_iteration(mountaincar, vec(true_params))[1]
end

function preprocess_data(data::Array{MountainCarContTransition})
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
    x_array_matrices::Array{Matrix{Float64}} = [permutedims(hcat(x_subarray...)) for x_subarray in x_array]
    x_array_matrices, x_array, x_next_array, disp_array, cost_array
end

function eval_params(
    mountaincar::MountainCar,
    model::MountainCar,
    params::Array{Float64},
    horizon::Int64;
    num_eval = 10,
    rng = nothing,
)
    policy, _, _ = value_iteration(model, params)
    num_steps = 0.0
    for _ in 1:num_eval
        num_steps += length(
            simulate_episode(
                mountaincar,
                true_params,
                horizon;
                policy = policy,
                rng = rng,
            )
        )
    end
    num_steps/num_eval
end

function eval_params_bellman(
    mountaincar::MountainCar,
    model::MountainCar,
    params::Array{Float64},
    horizon::Int64;
    num_eval = 10,
    rng = nothing,
)
    policy, values, _ = value_iteration(model, params)
    num_steps = 0.0
    actions = getActions(mountaincar)
    for _ in 1:num_eval
        episode = simulate_episode(
            mountaincar,
            true_params,
            horizon;
            policy = policy,
            rng = rng,
        )
        model_steps = length(simulate_episode(
            model,
            unvec_params(params),
            horizon;
            policy = policy,
            rng = rng,
        ))
        println(model_steps, " ", values[cont_state_to_idx(model, init(mountaincar; cont = true))])
        for transition in episode
            successor = cont_state_to_idx(model, transition.final_state)
            action = actions[policy[cont_state_to_idx(model, transition.initial_state)]]
            predicted_successor = cont_state_to_idx(
                model, step(model, transition.initial_state, action, params)[1]
            )
            model_steps += values[successor] - values[predicted_successor]
        end
        model_steps += -values[cont_state_to_idx(model, episode[end].final_state)]
        num_steps += model_steps
    end
    num_steps/num_eval
end
