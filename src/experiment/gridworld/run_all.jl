function gridworld_main()
    gridworld = create_example_gridworld()
    planner = GridworldPlanner(gridworld, 100)
    agent = GridworldAgent(gridworld, planner)
    run(agent)
end
