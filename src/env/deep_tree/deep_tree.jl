struct DeepTreeEnv
    depth::Int64
    adj_list::Dict{Int64, Vector{Int64}}
end

function DeepTreeEnv(depth::Int64, model::String, rng::MersenneTwister)
    @assert depth > 3
    num_nodes = compute_num_nodes(depth)
    list_of_nodes = collect(1:num_nodes)
    if model == "true"

    elseif model == "optimistic"

    else

    end
end

function compute_num_nodes(depth::Int64)
    sum([2^i for i in 0:depth])
end