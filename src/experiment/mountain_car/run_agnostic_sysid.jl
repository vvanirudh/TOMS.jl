function mountaincar_run_agnostic_sysid(rock_c::Float64)
    mountaincar = MountainCar(rock_c)
    model = MountainCar(0.0)
    horizon = 500
    agent = MountainCarAgnosticSysIDAgent(
        mountaincar,
        model,
        horizon,
    )
    run(agent; debug = true)
end
