struct Transition{Observation, Action}
    observation::Observation
    action::Action
    reward::Float32
    next_observation::Observation
end


struct Agent{Observation,
             Action,
             ObservationSpace<:AbstractSpace,
             ActionSpace<:AbstractSpace,
             Encoder,
             Decoder,
             Policy}
    observation_space::ObservationSpace
    action_space::ActionSpace
    encoder::Encoder
    decoder::Decoder
    policy::Policy
    transitions::CircularBuffer{Transition{Observation, Action}}
end


construct_encoder(box::Box{<:AbstractFloat, 1}, latent=2^5, hidden=2^6) =
    Chain(Dense(size(box)[1], hidden, relu), Dense(hidden, latent, relu))


construct_decoder(box::Box{<:AbstractFloat, 1}, latent=2^5, hidden=2^6) =
    Chain(Dense(latent, hidden, relu), Dense(hidden, size(box)[1]))


construct_policy(box::Box{<:AbstractFloat, 1}, latent=2^5, hidden=2^6) =
    Chain(Dense(latent, hidden, relu), Dense(hidden, size(box)[1] * 2))


construct_transition_buffer(::Box{T1, N1}, ::Box{T2, N2},
                            capacity=10_000) where {T1, T2, N1, N2} =
    CircularBuffer{Transition{Array{T1, N1}, Array{T2, N2}}}(capacity)


function Agent(env::Environment)
    os = observation_space(env)
    as = action_space(env)
    encoder = construct_encoder(os)
    decoder = construct_decoder(os)
    policy = construct_policy(as)
    transitions = construct_transition_buffer(os, as)
    Agent(os, as, encoder, decoder, policy, transitions)
end


function select_action(agent::Agent{Observation, <:AbstractArray{T, 1}},
                       observation::Observation) where {Observation, T}
    logits = reshape(agent.policy(agent.encoder(observation)), 2, :)
    μ, σ = logits[1, :], logits[2, :]
    action = rand(MvNormal(data(μ), data(σ)))
    T.(clamp.(action, low(agent.action_space), high(agent.action_space)))
end


remember!(agent::Agent{Observation, Action},
          transition::Transition{Observation, Action}
          ) where {Observation, Action} =
    push!(agent.transitions, transition)
