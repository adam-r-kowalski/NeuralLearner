module NeuralLearner

export Box, low, high, Discrete, n, Environment,
       observation_space, action_space, Agent, select_action,
       remember!, Transition, Simulation

using PyCall, Flux, Distributions, DataStructures
using Flux: onehot, data

include("environment.jl")
include("agent.jl")
include("simulation.jl")

end
