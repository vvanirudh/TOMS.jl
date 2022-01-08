function mountaincar_run_agnostic_sysid(rock_c::Float64; value_aware = true)
    mountaincar = MountainCar(rock_c)
    model = MountainCar(0.0)
    horizon = 500
    agent = MountainCarAgnosticSysIDAgent(
        mountaincar,
        model,
        horizon,
    )
    run(agent; debug = true, value_aware = value_aware)
end
