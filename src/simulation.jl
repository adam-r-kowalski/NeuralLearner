struct Simulation{Environment, Agent}
    env::Environment
    agent::Agent
    render::Bool

    Simulation(env::AbstractEnvironment{ObservationSpace, ActionSpace},
               agent::AbstractAgent{Observation, Action, ObservationSpace, ActionSpace},
               render::Bool=false
               ) where {ObservationSpace<:AbstractSpace,
                        ActionSpace<:AbstractSpace,
                        Observation,
                        Action} =
        new{typeof(env), typeof(agent)}(env, agent, render)
end


Base.IteratorSize(::Simulation) = Base.IsInfinite()


Base.eltype(::Simulation{Environment{Box{T, N}, Box{T2, N2}}}
           ) where {T, N, T2, N2} =
    Transition{Array{T, N}, Array{T2, N2}}


function Base.iterate(sim::Simulation, observation=reset(sim.env))
    action = select_action(sim.agent, observation)
    next_observation, reward, done = step(sim.env, action)
    sim.render && render(sim.environment)
    transition = Transition(observation, action, reward, next_observation, done)
    remember!(sim.agent, transition)
    transition, done ? reset(sim.env) : next_observation
end
