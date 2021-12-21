function generate_batch_data(
    mountaincar::MountainCar,
    params::MountainCarParameters,
    num_episodes::Int64,
    horizon::Int64;
    policy = nothing,
)::Array{MountainCarContTransition}
    data = []
    for episode = 1:num_episodes
        data = vcat(data, simulate_episode(mountaincar, params, horizon, policy=policy))
    end
    data = vcat(data, simulate_episode(mountaincar, params, horizon, policy = good_policy(mountaincar)))
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

function good_policy(mountaincar::MountainCar)
    value_iteration(mountaincar, vec(true_params))[1]
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