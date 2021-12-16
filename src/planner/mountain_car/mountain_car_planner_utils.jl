function value_iteration(
    mountaincar::MountainCar,
    params::Array{Float64};
    threshold = 1e-5,
    gamma = 0.99,
)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization + 1
    n_actions = 2
    T = generate_transition_matrix(mountaincar, params)
    R = generate_reward_vector(mountaincar)
    V_old = copy(R)
    pi = zeros(Int64, n_states)
    V = 2 * V_old
    Q = zeros((n_states, n_actions))
    error_vec = abs.(V - V_old) ./ (V_old .+ 1e-50)
    criterion = maximum(error_vec)
    while criterion >= threshold
        # @enter pi, V
        # println("Value iteration criterion ", criterion)
        V_old = copy(V)
        for a = 1:n_actions
            Q[:, a] = R .+ gamma .* V_old[T[:, a]]
        end
        idxs = argmin(Q, dims = 2)
        for s = 1:n_states
            pi[s] = idxs[s][2]
        end
        V = Q[idxs]
        error_vec = abs.(V - V_old) ./ (V_old .+ 1e-50)
        criterion = maximum(error_vec)
    end
    # println("Value iteration finished with criterion ", criterion)
    # removing absorbing state
    pi[1:n_states-1], V[1:n_states-1]
end

function generate_transition_matrix(mountaincar::MountainCar, params::Array{Float64})
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization + 1
    n_actions = 2
    T = zeros(Int64, (n_states, n_actions))
    actions = getActions(mountaincar)
    for a = 1:n_actions
        u = actions[a]
        for s = 1:n_states-1
            disc_state = idx_to_disc_state(mountaincar, s)
            if checkGoal(mountaincar, disc_state)
                T[s, a] = n_states  # absorbing state
            else
                s_next = disc_state_to_idx(
                    mountaincar,
                    step(mountaincar, disc_state, u, params)[1],
                )
                T[s, a] = s_next
            end
        end
        T[n_states, a] = n_states  # absorbing state
    end
    T
end

function generate_reward_vector(mountaincar::MountainCar)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization + 1
    R = zeros(n_states)
    for s = 1:n_states-1
        R[s] = getCost(mountaincar, idx_to_disc_state(mountaincar, s))
    end
    R[n_states] = 0
    R
end

function rtaa_planning(
    mountaincar::MountainCar,
    params::Vector{Float64};
    num_expansions::Int64 = 1000
)
    rtaa_planning(
        mountaincar,
        MountainCarParameters(params[1], params[2]),
        num_expansions = num_expansions
    )
end

function rtaa_planning(
    mountaincar::MountainCar,
    params::MountainCarParameters;
    num_expansions::Int64 = 1000
)
    planner = MountainCarRTAAPlanner(mountaincar, num_expansions, params)
    generateHeuristic!(planner)
    convert_planner_to_policy_and_values(mountaincar, planner)
end

function convert_planner_to_policy_and_values(
    mountaincar::MountainCar,
    planner::MountainCarRTAAPlanner,
)
    n_states = mountaincar.position_discretization * mountaincar.speed_discretization
    n_actions = 2
    pi = zeros(Int64, n_states)
    V = zeros(n_states)

    for s = 1:n_states
        if checkGoal(mountaincar, idx_to_disc_state(mountaincar, s))
            pi[s] = 1
            V[s] = 0
        else
            action, info = act(planner, idx_to_disc_state(mountaincar, s))
            pi[s] = action.id + 1
            V[s] = info["best_node_f"]
        end
    end
    pi, V
end