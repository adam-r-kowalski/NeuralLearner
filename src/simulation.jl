struct Simulation{
        ObservationSpace <: AbstractSpace,
        ActionSpace <: AbstractSpace,
        Observation,
        Action,
        Environment <: AbstractEnvironment{ObservationSpace, ActionSpace},
        Agent <: AbstractAgent{Observation, Action, ObservationSpace, ActionSpace},
        }
    env::Environment
    agent::Agent
end


Base.IteratorSize(::Simulation) = Base.IsInfinite()


Base.eltype(::Simulation{Box{T, N}, Box{T2, N2}}
           ) where {T, N, T2, N2} =
    Transition{Array{T, N}, Array{T2, N2}}


function Base.iterate(sim::Simulation, observation=reset(sim.env))
    action = select_action(sim.agent, observation)
    next_observation, reward, done = step(sim.env, action)
    transition = Transition(observation, action, reward, next_observation, done)
    remember!(sim.agent, transition)
    transition, done ? reset(sim.env) : next_observation
end
