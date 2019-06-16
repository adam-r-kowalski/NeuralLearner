struct Transition{Observation, Action}
    observation::Observation
    action::Action
    reward::Float32
    next_observation::Observation
    done::Bool
end


abstract type AbstractAgent{Observation,
                            Action,
                            ObservationSpace<:AbstractSpace,
                            ActionSpace<:AbstractSpace
                            }
end


struct Agent{Observation,
             Action,
             ObservationSpace<:AbstractSpace,
             ActionSpace<:AbstractSpace,
             Encoder,
             Decoder,
             Policy,
             Optimizer
             } <: AbstractAgent{Observation, Action, ObservationSpace, ActionSpace}
    observation_space::ObservationSpace
    action_space::ActionSpace
    encoder::Encoder
    decoder::Decoder
    π::Policy
    optimizer::Optimizer
    transitions::CircularBuffer{Transition{Observation, Action}}
    batch_size::Int
    γ::Float32
end


@treelike Agent


abstract type CustomizationToken end


struct DefaultToken <: CustomizationToken end


function encoder(::CustomizationToken, box::Box{<:AbstractFloat, 1})
    latent = 2^5
    hidden = 2^6
    Chain(Dense(size(box)[1], hidden, relu),
          Dense(hidden, latent, relu))
end


function decoder(::CustomizationToken, box::Box{<:AbstractFloat, 1})
    latent=2^5
    hidden=2^6
    Chain(Dense(latent, hidden, relu),
          Dense(hidden, size(box)[1]))
end


function policy(::CustomizationToken, box::Box{<:AbstractFloat, 1})
    latent=2^5
    hidden=2^6
    Chain(Dense(latent, hidden, relu),
          Dense(hidden, size(box)[1] * 2))
end


optimizer(::CustomizationToken) = ADAM()


transitions(::CustomizationToken, ::Box{T1, N1}, ::Box{T2, N2}
           ) where {T1, T2, N1, N2} =
    CircularBuffer{Transition{Array{T1, N1}, Array{T2, N2}}}(10_000)


batch_size(::CustomizationToken) = 200


discount_factor(::CustomizationToken) = Float32(0.9)


function Agent(env::AbstractEnvironment,
               token::CustomizationToken=DefaultToken())
    os = observation_space(env)
    as = action_space(env)
    Agent(os,
          as,
          encoder(token, os),
          decoder(token, os),
          policy(token, as),
          optimizer(token),
          transitions(token, os, as),
          batch_size(token),
          discount_factor(token))
end


function probability_distribution(agent::Agent{Observation, <:AbstractVector},
                                  encoded::AbstractMatrix) where Observation
    actions = size(agent.action_space)[1]
    logits = reshape(agent.π(encoded), 2, actions, :)
    map(1:agent.batch_size) do i
        μ, Σ = logits[1, :, i], exp.(logits[2, :, i])
        MvNormal(μ, Σ)
    end
end


function probability_distribution(agent::Agent{Observation, <:AbstractVector},
                                  observation::Observation) where Observation
    logits = reshape(agent.π(agent.encoder(observation)), 2, :)
    μ, Σ = logits[1, :], exp.(logits[2, :])
    MvNormal(μ, Σ)
end


Distributions.MvNormal(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}) =
    MvNormal(μ, PDiagMat(abs2.(σ)))


PDiagMat(v::AbstractVector) = Distributions.PDiagMat(v, inv.(v))


Base.eltype(::MvNormal{T}) where T = T


function select_action(agent::Agent{Observation, <:AbstractVector{T}},
                       observation::Observation) where {Observation, T}
    action = data.(rand(probability_distribution(agent, observation)))
    T.(clamp.(action, low(agent.action_space), high(agent.action_space)))
end


struct TransitionBatch{Observations, Action}
    observations::Observations
    actions::Vector{Action}
    rewards::Vector{Float32}
    next_observations::Observations
    dones::Vector{Bool}
end


function TransitionBatch(agent::Agent{Observation, Action}
                        ) where {Observation, Action}
    batch_size = agent.batch_size - 1
    start_index = rand(1:(length(agent.transitions) - batch_size))
    batch = agent.transitions[start_index:start_index + batch_size]
    observations = Observation[]
    actions = Action[]
    rewards = Float32[]
    next_observations = Observation[]
    dones = Bool[]
    for transition ∈ batch
        push!(observations, transition.observation)
        push!(actions, transition.action)
        push!(rewards, transition.reward)
        push!(next_observations, transition.next_observation)
        push!(dones, transition.done)
    end
    TransitionBatch(reduce(hcat, observations),
                    actions,
                    rewards,
                    reduce(hcat, next_observations),
                    dones)
end


function discount(batch::TransitionBatch, γ::AbstractFloat)
    rewards = batch.rewards
    dones = batch.dones
    discounted = similar(rewards)
    running_sum = zero(γ)
    for i ∈ length(rewards):-1:1
        dones[i] && (running_sum = zero(γ))
        running_sum = rewards[i] + γ * running_sum
        discounted[i] = running_sum
    end
    discounted
end


normalize(xs::AbstractVector{<:AbstractFloat}) =
    (xs .- mean(xs)) / (std(xs) + eps(eltype(xs)))


function improve!(agent::Agent)
    length(agent.transitions) < agent.batch_size && return nothing
    batch = TransitionBatch(agent)
    encoded = agent.encoder(batch.observations)
    decoded = agent.decoder(encoded)
    reconstruction_loss = mse(decoded, batch.observations)
    normals = probability_distribution(agent, encoded)
    log_probabilities = map(normals, batch.actions) do normal, action
        loglikelihood(normal, reshape(action, :, 1))
    end
    returns = discount(batch, agent.γ)
    π_loss = mean(-log_probabilities .* returns)
    θ = params(agent)
    Δ = gradient(θ) do
        reconstruction_loss + π_loss
    end
    update!(agent.optimizer, θ, Δ)
end


function remember!(agent::Agent{Observation, Action},
          transition::Transition{Observation, Action}
          ) where {Observation, Action}
    push!(agent.transitions, transition)
end
