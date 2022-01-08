struct MountainCarAgnosticSysIDAgent
    mountaincar::MountainCar
    model::MountainCar
    model_class::Array{MountainCarParameters}
    horizon::Int64
end

function MountainCarAgnosticSysIDAgent(
    mountaincar::MountainCar,
    model::MountainCar,
    horizon::Int64
)
    model_class = generate_model_class()
    MountainCarAgnosticSysIDAgent(mountaincar, model, model_class, horizon)
end

function generate_model_class()::Array{MountainCarParameters}
    # theta1s = -0.0024:-0.00001:-0.0026
    # theta2s = 2.9:0.01:3.1
    # models = []
    # for theta1 in theta1s
    #     for theta2 in theta2s
    #         push!(models, MountainCarParameters(theta1, theta2))
    #     end
    # end
    models = [MountainCarParameters(-0.0025, 3), MountainCarParameters(-0.002640625, 2.93359375)]
    models
end

function run(
    agent::MountainCarAgnosticSysIDAgent;
    debug = false,
    value_aware = true,
)
    rng = MersenneTwister(0)
    # Choose an initial model
    params = true_params
    # Choose an initial policy
    policy, values, _ = value_iteration(agent.model, params)
    # Initialize dataset
    all_transitions::Array{Array{MountainCarContTransition}} = []
    # Initialize value functions
    all_values::Array{Array{Float64}} = []
    # Initialize losses
    losses = [0.0 for _ in 1:length(agent.model_class)]

    # Start iterations
    n = 1000  # number of iterations
    m = 10  # number of rollouts using policy
    p = 10  # number of rollouts using random policy with random start
    for i in 1:n
        push!(all_transitions, [])
        # Do rollouts in the real world using policy to collect data
        for _ in 1:m
            episode = simulate_episode(
                agent.mountaincar,
                true_params,
                agent.horizon;
                policy = policy,
            )
            all_transitions[i] = vcat(all_transitions[i], episode)
        end
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
            else
                loss = compute_l2_loss(
                    agent,
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
        params = agent.model_class[argmin(losses)]
        # println(vec(params), " ", minimum(losses))
        println([(vec(agent.model_class[j]), losses[j]) for j in 1:length(agent.model_class)])
        # Compute corresponding policy and values
        policy, values, _ = value_iteration(agent.model, params)
    end
end

function compute_model_advantage_loss(
    agent::MountainCarAgnosticSysIDAgent,
    transitions::Array{MountainCarContTransition},
    values::Array{Float64},
    model_params::MountainCarParameters,
)
    loss = 0.0
    for transition in transitions
        predicted_state, _ = step(
            agent.model,
            transition.initial_state,
            transition.action,
            model_params,
        )
        loss += abs(
            values[
                cont_state_to_idx(
                    agent.model,
                    transition.final_state,
                )
            ] -
            values[
                cont_state_to_idx(
                    agent.model,
                    predicted_state,
                )
            ]
        )
    end
    loss
end

function compute_l2_loss(
    agent::MountainCarAgnosticSysIDAgent,
    transitions::Array{MountainCarContTransition},
    model_params::MountainCarParameters,
)
    loss = 0.0
    for transition in transitions
        predicted_state, _ = step(
            agent.model,
            transition.initial_state,
            transition.action,
            model_params,
        )
        loss += (predicted_state.position - transition.final_state.position)^2 + (predicted_state.speed - transition.final_state.speed)^2
    end
    loss
end