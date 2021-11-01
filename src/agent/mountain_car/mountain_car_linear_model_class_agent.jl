struct MountainCarLinearModelClassAgent
    mountaincar::MountainCar
    planners::Vector{MountainCarRTAAPlanner}
    transitions::Queue{MountainCarTransition}
    horizon::Int64
    weights::Vector{Float64}
end

function MountainCarLinearModelClassAgent(mountaincar::MountainCar, planners::Vector{MountainCarRTAAPlanner})
    transitions = Queue{MountainCarTransition}()
    horizon = 100
    # uniform weights at start
    weights = fill(1/length(planners), length(planners))
    MountainCarLinearModelClassAgent(mountaincar, planners, transitions, horizon, weights)
end

function run(agent::MountainCarLinearModelClassAgent)
    # TODO: Implement
end