scipy_optimize = pyimport("scipy.optimize")

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
        # push!(err, norm(vec(predicted_state) - vec(transition.final_state)))
        push!(err, (predicted_state.position -
                    transition.final_state.position)^2 + (predicted_state.speed -
                                                          transition.final_state.speed)^2)

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

function get_least_squares_fit(
    mountaincar::MountainCar,
    params::Array{Float64},
    data::Array{MountainCarContTransition},
)
    get_least_squares_fit(
        mountaincar,
        unvec_params(params),
        data,
    )
end
