gym = pyimport("gym")

abstract type AbstractSpace end

struct Box{T<:AbstractFloat, N} <: AbstractSpace
    low::AbstractArray{T, N}
    high::AbstractArray{T, N}
end

Box(low::T, high::T, shape::NTuple{N, Integer}) where {T<:AbstractFloat, N} =
    Box(fill(low, shape), fill(high, shape))

function Box(p::PyObject)
    @assert pytypeof(p) == NeuralLearner.gym.spaces.Box
    Box(p.low, p.high)
end

Base.size(box::Box) = size(box.low)

low(box::Box) = box.low
high(box::Box) = box.high

struct Discrete{T<:Integer} <: AbstractSpace
    n::T
end

function Discrete(p::PyObject)
    @assert pytypeof(p) == NeuralLearner.gym.spaces.Discrete
    Discrete(p.n)
end

n(discrete::Discrete) = discrete.n

function AbstractSpace(p::PyObject)
    t = pytypeof(p)
    t == gym.spaces.Discrete && return Discrete(p)
    t == gym.spaces.Box && return Box(p)
end

struct Environment{ObservationSpace<:AbstractSpace, ActionSpace<:AbstractSpace}
    p::PyObject
    observation_space::ObservationSpace
    action_space::ActionSpace
end

function Environment(name::AbstractString)
    p = gym.make(name)
    observation_space = AbstractSpace(p.observation_space)
    action_space = AbstractSpace(p.action_space)
    Environment(p, observation_space, action_space)
end

observation_space(env::Environment) = env.observation_space
action_space(env::Environment) = env.action_space

preprocess_observation(env::Environment{Box{T, N}},
                       observation::AbstractArray{<:AbstractFloat, N}) where {T, N} =
    T.(observation)

preprocess_observation(env::Environment{Discrete{T}}, observation::T) where T =
    onehot(env.p.reset() + 1, 1:n(observation_space(env)))

Base.reset(env::Environment) = preprocess_observation(env, env.p.reset())

function Base.step(env::Environment{ObservationSpace, Discrete{T}},
                   action::T) where {ObservationSpace, T}
    observation, reward, done, _ = env.p.step(action - 1)
    preprocess_observation(env, observation), Float32(reward), done
end

function Base.step(env::Environment{ObservationSpace, Box{T, N}},
                   action::AbstractArray{T, N}) where {ObservationSpace, T, N}
    observation, reward, done, _ = env.p.step(action)
    preprocess_observation(env, observation), Float32(reward), done
end

Base.close(env::Environment) = env.p.close()
