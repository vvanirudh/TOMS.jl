struct WideTreeAgnosticSysIDAgent
    widetree::WideTreeEnv
    model_class::Array{WideTreeEnv}
end

function WideTreeAgnosticSysIDAgent(widetree::WideTreeEnv, rng::MersenneTwister)
    model_class::Array{WideTreeEnv} = []
    # Add bad model
    push!(model_class, WideTreeEnv(widetree.n_leaves, "bad", rng))
    # Add several good models
    for _ in 1:5
        push!(model_class, WideTreeEnv(widetree.n_leaves, "good", rng))
    end
    WideTreeAgnosticSysIDAgent(widetree, model_class)
end

function run(
    agent::WideTreeAgnosticSysIDAgent,
    rng::MersenneTwister;
    debug = false,
    value_aware = true,
)
    # Choose an initial model
    model = agent.model_class[1]
    # Compute policy and values
    policy, values = optimal_policy_and_values(model, rng)
    # Initialize losses
    losses = [0.0 for _ in 1:length(agent.model_class)]

    # Start iterations
    n = 100  # number of iterations
    m = 10  # number of rollouts using policy
    p = 10  # number of rollouts using random policy with random start

    for _ in 1:n
        all_transitions::Array{Tuple{Int64, Int64, Int64, Int64}} = []
        performance = 0.
        # Do rollouts in the real world using policy to collect data
        for _ in 1:m
            transitions, total_return = simulate_episode(
                agent.widetree, policy, rng,
            )
            performance += total_return
            all_transitions = vcat(all_transitions, transitions)
        end
        println("Cost to go ", performance/m)
        # Do rollouts in the real world using random policy to collect data
        for _ in 1:p
            random_policy, _ = random_policy_and_values(
                model, rng,
            )
            transitions, _ = simulate_episode(
                agent.widetree, random_policy, rng,
            )
            all_transitions = vcat(all_transitions, transitions)
        end
        # Collected data, now update model
        for j in 1:length(agent.model_class)
            if value_aware
                losses[j] += compute_model_advantage_loss(
                    agent, all_transitions, values, agent.model_class[j], rng,
                ) / (m + p)
            else
                losses[j] += compute_classification_loss(
                    agent, all_transitions, agent.model_class[j], rng,
                ) / (m + p)
            end
        end
        # Find best model
        # TODO: Doing FTL now, but need to do better
        model = agent.model_class[argmin(losses)]
        println(losses)
        println("Selected ", argmin(losses))
        # Compute policy and values
        policy, values = optimal_policy_and_values(model, rng)
    end
end

function simulate_episode(
    widetree::WideTreeEnv,
    policy::Array{Int64},
    rng::MersenneTwister,
)
    state = init(widetree)
    transitions::Array{Tuple{Int64, Int64, Int64, Int64}} = []
    total_return = 0
    while !checkTerminal(widetree, state)
        action = policy[state]
        next_state, cost = step(widetree, state, action, rng)
        push!(transitions, (state, action, cost, next_state))
        total_return += cost
        state = next_state
    end

    transitions, total_return
end

function compute_model_advantage_loss(
    agent::WideTreeAgnosticSysIDAgent,
    transitions::Array{Tuple{Int64, Int64, Int64, Int64}},
    values::Array{Int64},
    model::WideTreeEnv,
    rng::MersenneTwister,
)
    loss = 0.
    for transition in transitions
        predicted_state, _ = step(model, transition[1], transition[2], rng)
        loss += abs(values[transition[4]] - values[predicted_state])
    end
    loss
end

function compute_classification_loss(
    agent::WideTreeAgnosticSysIDAgent,
    transitions::Array{Tuple{Int64, Int64, Int64, Int64}},
    model::WideTreeEnv,
    rng::MersenneTwister,
)
    loss = 0.
    for transition in transitions
        predicted_state, _ = step(model, transition[1], transition[2], rng)
        loss += 1 - Float64(predicted_state == transition[4])
    end
    loss
end