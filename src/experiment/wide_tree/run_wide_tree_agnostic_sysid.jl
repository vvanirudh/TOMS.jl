function run_wide_tree_agnostic_sysid(n_leaves::Int64; value_aware=true)
    rng = MersenneTwister(1)
    widetree = WideTreeEnv(n_leaves, "real", rng)
    agent = WideTreeAgnosticSysIDAgent(widetree, rng)
    run(agent, rng; value_aware=value_aware)
end