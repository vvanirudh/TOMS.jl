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
    theta1s = -0.0024:-0.00001:-0.0026
    theta2s = 2.9:0.01:3.1
    models = []
    for theta1 in theta1s
        for theta2 in theta2s
            push!(models, MountainCarParameters(theta1, theta2))
        end
    end
    models
end

function run(
    agent::MountainCarAgnosticSysIDAgent;
    debug = false,
)
    rng = MersenneTwister(0)
    # Choose an initial model
    params = MountainCarParameters(-0.002, 3.5)
    # Choose an initial policy
    policy, values, _ = value_iteration(agent.model, params)
    # Initialize dataset
    all_transitions::Array{Array{MountainCarContTransition}} = []
    # Initialize value functions
    all_values::Array{Array{Float64}} = []

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
        losses = []
        for model_params in agent.model_class
            # TODO: Need to do something more FTRL like
            # Compute loss
            loss = compute_model_advantage_loss(
                agent,
                all_transitions,
                all_values,
                model_params
            )
            loss = loss / (i * (m + p))
            # println(vec(model_params), " ", loss)
            push!(losses, loss)
        end
        # Find best model
        params = agent.model_class[argmin(losses)]
        println(vec(params), " ", minimum(losses))
        # Compute corresponding policy and values
        policy, values, _ = value_iteration(agent.model, params)
    end
end

function compute_model_advantage_loss(
    agent::MountainCarAgnosticSysIDAgent,
    all_transitions::Array{Array{MountainCarContTransition}},
    all_values::Array{Array{Float64}},
    model_params::MountainCarParameters,
)
    loss = 0.0
    for j in 1:length(all_transitions)
        iteration_transitions = all_transitions[j]
        iteration_values = all_values[j]
        for transition in iteration_transitions
            predicted_state, _ = step(
                agent.model,
                transition.initial_state,
                transition.action,
                model_params,
            )
            loss += abs(
                iteration_values[
                    cont_state_to_idx(
                        agent.model,
                        transition.final_state,
                    )
                ] -
                iteration_values[
                    cont_state_to_idx(
                        agent.model,
                        predicted_state,
                    )
                ]
            )
        end
    end
    loss
end
