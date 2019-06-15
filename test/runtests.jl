module NeuralLearnerTest

include("../src/NeuralLearner.jl")
using .NeuralLearner, Test, Flux
using Flux: OneHotVector

@testset "neural learner tests" begin

include("test_environment.jl")
include("test_agent.jl")
include("test_simulation.jl")

end

end
