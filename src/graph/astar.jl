import Base: ==
import Base: hash

struct Node
    state::State
    g::Float64
    h::Float64
    came_from::Union{Node,Nothing}
    action::Union{Action,Nothing}
end

function ==(node1::Node, node2::Node)
    node1.state == node2.state
end

function hash(node::Node)
    hash(node.state)
end

function astar(planner::Planner, num_expansions::Int64, start_state::State)
    closed_set = Set{Node}()
    open_list = MutableBinaryMinHeap{Tuple{Float64,Float64,Int64}}()
    open_dict = Dict{Node,Int64}()
    open_dict_reverse = Dict{Int64,Node}()

    reached_goal = false
    count = 0
    best_node = nothing

    h = getHeuristic(planner, start_state)
    g = 0.0
    f = h
    updated_start_node = Node(start_state, g, h, nothing, nothing)
    handle = push!(open_list, (f, h, count))
    count += 1
    open_dict[updated_start_node] = handle
    open_dict_reverse[handle] = updated_start_node

    for exp = 1:num_expansions
        triplet, handle = top_with_handle(open_list)
        f, h, _ = triplet
        node = open_dict_reverse[handle]
        pop!(open_list)
        delete!(open_dict, node)
        delete!(open_dict_reverse, handle)
        push!(closed_set, node)

        if checkGoal(planner, node.state)
            reached_goal = true
            best_node = node
            break
        end

        for action in getActions(planner, node.state)
            neighbor_state, cost = getSuccessors(planner, node.state, action)
            neighbor_node = Node(neighbor_state, 0, 0, nothing, nothing)
            if neighbor_node in closed_set
                continue
            end

            tentative_g = node.g + cost
            if !haskey(open_dict, neighbor_node)
                h = getHeuristic(planner, neighbor_state)
                f = tentative_g + h
                updated_neighbor_node = Node(neighbor_state, tentative_g, h, node, action)
                handle = push!(open_list, (f, h, count))
                count += 1
                open_dict[updated_neighbor_node] = handle
                open_dict_reverse[handle] = updated_neighbor_node
            else
                handle = open_dict[neighbor_node]
                old_neighbor_node = open_dict_reverse[handle]
                if tentative_g < old_neighbor_node.g
                    h = old_neighbor_node.h
                    f = tentative_g + h
                    updated_neighbor_node =
                        Node(old_neighbor_node.state, tentative_g, h, node, action)
                    delete!(open_dict, neighbor_node)
                    update!(open_list, handle, (f, h, count))
                    count += 1
                    open_dict[updated_neighbor_node] = handle
                    open_dict_reverse[handle] = updated_neighbor_node
                end
            end
        end
        if length(open_list) == 0
            best_node = node
            break
        end
    end

    if !reached_goal && length(open_list) > 0
        _, handle = top_with_handle(open_list)
        best_node = open_dict_reverse[handle]
    end

    info = Dict()
    info["best_node_f"] = best_node.g + best_node.h
    info["reached_goal"] = reached_goal
    info["closed"] = closed_set

    best_action, path = getBestAction(updated_start_node, best_node)
    info["path"] = path

    return best_action, info
end

function getBestAction(start_node::Node, best_node::Node)
    if start_node == best_node
        println("Should not reach this!")
        return nothing, [start_node.state, start_node.state]
    end

    node = best_node.came_from
    action = best_node.action
    path = [best_node.state]
    while true
        if !isnothing(node.came_from)
            # not the start node
            push!(path, node.state)
            next_node = node.came_from
            action = node.action
            if isnothing(next_node.came_from)
                # next node is start node
                break
            end
            node = next_node
        else
            # start node
            break
        end
    end
    push!(path, start_node.state)
    reverse!(path)
    return action, path
end
