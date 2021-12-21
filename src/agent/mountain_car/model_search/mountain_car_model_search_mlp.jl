using Flux
using Flux.Losses: mse
using Flux.Data: DataLoader


function build_model(; hidden_layer_size::Int64 = 32)
    return Chain(Dense(2, hidden_layer_size, relu), Dense(hidden_layer_size, 2))
end

function get_data(
    x_array::Array{Array{Float64}},
    disp_array::Array{Array{Float64}};
    batchsize::Int64 = 256,
)
    Xtrain = hcat(x_array...)
    Ytrain = hcat(disp_array...)

    DataLoader((Xtrain, Ytrain), batchsize = batchsize, shuffle = true)
end

function loss_fn(data, model)
    loss = 0.0
    num = 0
    for (x, y) in data
        ŷ = model(x)
        loss += mse(ŷ, y)
        num += size(x)[end]
    end
    loss / num
end

function train(
    x_array::Array{Array{Float64}},
    disp_array::Array{Array{Float64}};
    lr::Float64 = 1e-3,
    batchsize = 256,
    epochs::Int64 = 10,
)
    data = get_data(x_array, disp_array, batchsize = batchsize)
    model = build_model() |> cpu
    ps = Flux.params(model)
    opt = ADAM(lr)

    for epoch = 1:epochs
        for (x, y) in data
            gs = gradient(() -> mse(model(x), y), ps)
            Flux.Optimise.update!(opt, ps, gs)
        end
        loss = loss_fn(data, model)
        println("Epoch = ", epoch, " Loss = ", loss)
    end
    model
end

function predict(model, x::Array{Float64})::Array{Float64}
    x + model(x)
end

function fit_ensemble(
    x_array::Array{Array{Float64}},
    disp_array::Array{Array{Float64}};
    ensemble_size::Int64 = 10,
)
    models = []
    for ensemble=1:ensemble_size
        println("Fitting network ", ensemble, " in the ensemble")
        push!(models, train(x_array, disp_array))
    end
    models
end

function predict_ensemble(
    models,
    x::Array{Float64},
)::Array{Array{Float64}}
    predictions = []
    for model in models
        push!(predictions, predict(model, x))
    end
    predictions
end

function fit_ensembles(
    x_array::Array{Array{Array{Float64}}},
    disp_array::Array{Array{Array{Float64}}}, 
)
    ensembles = []
    n_actions = length(x_array)
    for a=1:n_actions
        println("Fitting ensembles for action ", a)
        push!(ensembles, fit_ensemble(x_array[a], disp_array[a]))
    end
    ensembles
end

function find_max_distance(
    predictions::Array{Array{Float64}}
)
    max_distance = 0.0
    for i=1:length(predictions)
        for j=i+1:length(predictions)
            x = predictions[i]
            y = predictions[j]
            max_distance = max(max_distance, norm(x - y))
        end
    end
    max_distance
end