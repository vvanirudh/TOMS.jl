struct WideTreeEnv
    n_leaves::Int64
    left::Int64
    right::Int64
    first_parent::Int64
    second_parent::Int64
    first_bucket::Array{Int64}
    second_bucket::Array{Int64}
    third_bucket::Array{Int64}
    fourth_bucket::Array{Int64}
end

function WideTreeEnv(n_leaves::Int64, model::String, rng::MersenneTwister)
    @assert n_leaves >= 4 && n_leaves % 4 == 0
    @assert model in ["real", "good", "bad"]
    bucket_size = n_leaves÷4
    if model == "real"
        left = 2
        right = 3
        first_parent = 4
        second_parent = 5
        first_bucket = collect(5+1:5+bucket_size)
        second_bucket = collect(5+bucket_size+1:5+2*bucket_size)
        third_bucket = collect(5+2*bucket_size+1:5+3*bucket_size)
        fourth_bucket = collect(5+3*bucket_size+1:5+4*bucket_size)
        return WideTreeEnv(
            n_leaves, left, right, first_parent, second_parent, 
            first_bucket, second_bucket, third_bucket, 
            fourth_bucket,
        )
    elseif model == "good"
        left = 2
        right = 3
        first_parent = 5
        second_parent = 4
        # Generate a random permutation of leaves
        leaves = collect(5+1:5+n_leaves)
        leaves = leaves[randperm(rng, n_leaves)]
        first_bucket = leaves[1:bucket_size]
        second_bucket = leaves[bucket_size+1:2*bucket_size]
        third_bucket = leaves[2*bucket_size+1:3*bucket_size]
        fourth_bucket = leaves[3*bucket_size+1:4*bucket_size]
        return WideTreeEnv(
            n_leaves, left, right, first_parent, second_parent,
            first_bucket, second_bucket, third_bucket,
            fourth_bucket,
        )
    elseif model == "bad"
        # Just switch dynamics at the root
        left = 3
        right = 2
        first_parent = 4
        second_parent = 5
        first_bucket = collect(5+1:5+bucket_size)
        second_bucket = collect(5+bucket_size+1:5+2*bucket_size)
        third_bucket = collect(5+2*bucket_size+1:5+3*bucket_size)
        fourth_bucket = collect(5+3*bucket_size+1:5+4*bucket_size)
        return WideTreeEnv(
            n_leaves, left, right, first_parent, second_parent, 
            first_bucket, second_bucket, third_bucket, 
            fourth_bucket,
        )
    end
end

function init(widetree::WideTreeEnv)
    1
end

function num_nodes(widetree::WideTreeEnv)
    5 + widetree.n_leaves
end

function num_non_leaves(widetree::WideTreeEnv)
    5
end

function step(widetree::WideTreeEnv, state::Int64, action::Int64, rng::MersenneTwister)
    @assert state >= 1 && state <= 5 + widetree.n_leaves
    @assert action in [0, 1]
    if state == 1
        if action == 0
            return widetree.left, 0
        else
            return widetree.right, 0
        end
    elseif state == 2
        return widetree.first_parent, 1
    elseif state == 3
        return widetree.second_parent, 0
    elseif state == widetree.first_parent
        if action == 0
            return rand(rng, widetree.first_bucket), 0
        else
            return rand(rng, widetree.second_bucket), 0
        end
    elseif state == widetree.second_parent
        if action == 0
            return rand(rng, widetree.third_bucket), 0
        else
            return rand(rng, widetree.fourth_bucket), 0
        end
    else
        println("Should never be queried")
    end
end

function optimal_policy_and_values(widetree::WideTreeEnv, rng::MersenneTwister)
    policy = rand(rng, 0:1, 5+widetree.n_leaves)
    if widetree.left == 2
        policy[1] = 1  # Take right
    else
        policy[1] = 0  # Take left
    end
    values = zeros(Int64, 5+widetree.n_leaves)
    values[2] = 1
    policy, values
end

function random_policy_and_values(widetree::WideTreeEnv, rng::MersenneTwister)
    policy = rand(rng, 0:1, 5+widetree.n_leaves)
    values = zeros(Int64, 5+widetree.n_leaves)
    if policy[1] == 0 && widetree.left == 2
        values[1] = 1
    elseif policy[1] == 1 && widetree.right == 2
        values[1] = 1
    end
    values[2] = 1
    policy, values
end

function checkTerminal(widetree::WideTreeEnv, state::Int64)
    state > 5
end