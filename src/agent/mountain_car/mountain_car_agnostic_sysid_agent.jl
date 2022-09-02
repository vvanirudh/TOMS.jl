struct MountainCarAgnosticSysIDAgent
    mountaincar::MountainCar
    model::MountainCar
    model_class::Array{Tuple{MountainCar, MountainCarParameters}}
    horizon::Int64
end

function MountainCarAgnosticSysIDAgent(
    mountaincar::MountainCar,
    model::MountainCar,
    horizon::Int64
)
    model_class = generate_model_class(model)
    MountainCarAgnosticSysIDAgent(mountaincar, model, model_class, horizon)
end

function generate_model_class(model::MountainCar)::Array{Tuple{MountainCar, MountainCarParameters}}
    theta1s = -0.002:-0.00001:-0.003
    theta2s = 2.5:0.01:3.5
    models = []
    for theta1 in theta1s
        for theta2 in theta2s
            push!(models, (model, MountainCarParameters(theta1, theta2)))
        end
    end
    #= # The second parameter below is good for 0.03 rock_c
    models = [
        (MountainCar(0.0), MountainCarParameters(-0.0025, 3)),
        (MountainCar(0.03), MountainCarParameters(-0.003, 3.5)),
    ] =#
    models
end

function run(
    agent::MountainCarAgnosticSysIDAgent;
    debug = false,
    value_aware = true,
    value_aware_both = false,
    off_policy = false,
    on_policy = true,
    use_optimal_values = false,
    use_optimal_policy = false,
    num_iterations = 100,
)
    rng = MersenneTwister(0)
    # rng = nothing
    # Choose an initial model
    model, params = agent.model_class[1]
    # Choose an initial policy
    policy, values, _ = value_iteration(model, params)
    collect_policy = copy(policy)
    optimal_policy, optimal_values, _ = value_iteration(agent.mountaincar, true_params)
    optimal_cost = cost_episode(simulate_episode(agent.mountaincar, true_params, agent.horizon; policy=optimal_policy, rng=nothing))
    if use_optimal_values
        values = copy(optimal_values)
    end
    if use_optimal_policy
        collect_policy = copy(optimal_policy)
    end
    # Initialize dataset
    all_transitions::Array{Array{MountainCarContTransition}} = []
    # Initialize value functions
    all_values::Array{Array{Float64}} = []
    # Initialize losses
    losses = [0.0 for _ in 1:length(agent.model_class)]
    # All costs 
    total_costs::Array{Float64} = []
    # Consistencies
    consistencies::Array{Float64} = []

    # Start iterations
    n = num_iterations  # number of iterations
    m = 2  # number of rollouts using policy
    if !on_policy
        m = 0
    end
    p = 0  # number of rollouts using random policy with random start
    if off_policy
        p = 2
    end
    for i in 1:n
        println("Running iteration ", i)
        push!(all_transitions, [])
        # Do rollouts in the real world using policy to collect data
        total_cost = 0.0
        for _ in 1:m
            episode = simulate_episode(
                agent.mountaincar,
                true_params,
                agent.horizon;
                policy = collect_policy,
                rng = rng,
            )
            all_transitions[i] = vcat(all_transitions[i], episode)
        end
        # println("Return is ", total_cost)
        total_cost = cost_episode(simulate_episode(agent.mountaincar, true_params, agent.horizon; policy=policy, rng=nothing))
        total_cost_in_model = cost_episode(simulate_episode(model, params, agent.horizon; policy=policy, rng=nothing))
        push!(total_costs, total_cost)
        push!(consistencies, total_cost - total_cost_in_model)
        # Do rollouts in the real world using random policy to collect data
        for _ in 1:p
            episode = simulate_episode(
                agent.mountaincar,
                true_params,
                agent.horizon;
                rng = rng,
            )
            all_transitions[i] = vcat(all_transitions[i], episode)
        end
        # Store value function
        push!(all_values, values)
        if value_aware_both
            opt_policy_values = iterative_policy_evaluation(model, params, optimal_policy)
        end
        # Collected data, now update model
        for j in 1:length(agent.model_class)
            # Compute loss
            loss = 0.0
            if value_aware
                loss = compute_model_advantage_loss(
                    agent,
                    all_transitions[i],
                    all_values[i],
                    agent.model_class[j],
                )
                if value_aware_both
                    loss += compute_model_advantage_loss(
                        agent,
                        all_transitions[i],
                        opt_policy_values,
                        agent.model_class[j]
                    )
                end
            else
                loss = compute_l2_loss(
                    all_transitions[i],
                    agent.model_class[j],
                )
            end
            loss = loss / (m + p)
            # println(vec(model_params), " ", loss)
            losses[j] += loss
        end
        # Find best model
        # TODO: Doing FTL now, but need to do better
        model, params = agent.model_class[argmin(losses)]
        println("Best model so far has params ", vec(params), " and loss ", minimum(losses))
        # Compute corresponding policy and values
        policy, values, _ = value_iteration(model, params)
        if use_optimal_policy
            collect_policy = copy(optimal_policy)
        end
        if use_optimal_values
            values = copy(optimal_values)
        end
    end
    total_costs, consistencies, optimal_cost
end

function compute_model_advantage_loss(
    agent::MountainCarAgnosticSysIDAgent,
    transitions::Array{MountainCarContTransition},
    values::Array{Float64},
    model_params::Tuple{MountainCar, MountainCarParameters},
)
    # _, values, _ = value_iteration(agent.mountaincar, true_params)
    loss = 0.0
    model, params = model_params
    for transition in transitions
        predicted_state, _ = step(
            model,
            transition.initial_state,
            transition.action,
            params,
        )
        loss += abs(
            values[
                cont_state_to_idx(
                    model,
                    transition.final_state,
                )
            ] -
            values[
                cont_state_to_idx(
                    model,
                    predicted_state,
                )
            ]
        )
    end
    loss
end

function compute_l2_loss(
    transitions::Array{MountainCarContTransition},
    model_params::Tuple{MountainCar, MountainCarParameters},
)
    model, params = model_params
    loss = 0.0
    for transition in transitions
        predicted_state, _ = step(
            model,
            transition.initial_state,
            transition.action,
            params,
        )
        loss += (predicted_state.position - transition.final_state.position)^2 + (predicted_state.speed - transition.final_state.speed)^2
    end
    loss
end

function cost_episode(episode::Array{MountainCarContTransition})
    cost_val = 0.0
    for transition in episode
        cost_val += transition.cost
    end
    cost_val
end