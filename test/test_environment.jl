@testset "environment tests" begin

@testset "create box using low and high" begin
    box = Box([-1., -2], [1., 2])
    @test low(box) == [-1., -2]
    @test high(box) == [1., 2]
    @test size(box) == (2,)
end

@testset "create box using low, high and shape" begin
    box = Box(-1., 1., (3, 2))
    @test low(box) == [-1. -1
                      -1 -1
                      -1 -1]
    @test high(box) == [1. 1
                       1 1
                       1 1]
    @test size(box) == (3, 2)
end

@testset "create box from python using low, high and shape" begin
    python_box = NeuralLearner.gym.spaces.Box(low=[-1, -2], high=[1, 2])
    box = Box(python_box)
    @test low(box) == [-1., -2]
    @test high(box) == [1., 2]
    @test size(box) == (2,)
end

@testset "create box from python using low, high and shape" begin
    python_box = NeuralLearner.gym.spaces.Box(low=-1, high=1, shape=(3, 2))
    box = Box(python_box)
    @test low(box) == [-1. -1
                      -1 -1
                      -1 -1]
    @test high(box) == [1. 1
                       1 1
                       1 1]
    @test size(box) == (3, 2)
end

@testset "create discrete" begin
    discrete = Discrete(5)
    @test n(discrete) == 5
end

@testset "create discrete from python" begin
    python_discrete = NeuralLearner.gym.spaces.Discrete(5)
    discrete = Discrete(python_discrete)
    @test n(discrete) == 5
end

@testset "abstract space from python box" begin
    python_box = NeuralLearner.gym.spaces.Box(low=-1, high=1, shape=(3, 2))
    box = NeuralLearner.AbstractSpace(python_box)
    @test box isa Box{Float32, 2}
end

@testset "abstract space from python discrete" begin
    python_discrete = NeuralLearner.gym.spaces.Discrete(5)
    discrete = NeuralLearner.AbstractSpace(python_discrete)
    @test discrete isa Discrete{Int64}
end

@testset "cartpole environment" begin
    env = Environment("CartPole-v0")
    @test observation_space(env) isa Box{Float32, 1}
    @test action_space(env) isa Discrete{Int64}
    observation = reset(env)
    @test observation isa Vector{Float32}
    next_observation, reward, done = step(env, 1)
    @test next_observation isa Vector{Float32}
    @test reward isa Float32
    @test done isa Bool
    close(env)
end

@testset "frozen lake environment" begin
    env = Environment("FrozenLake-v0")
    @test observation_space(env) isa Discrete{Int}
    @test action_space(env) isa Discrete{Int}
    observation = reset(env)
    @test observation isa OneHotVector
    next_observation, reward, done = step(env, 1)
    @test next_observation isa OneHotVector
    @test reward isa Float32
    @test done isa Bool
    close(env)
end

@testset "ant environment" begin
    env = Environment("Ant-v2")
    @test observation_space(env) isa Box{Float32, 1}
    @test action_space(env) isa Box{Float32, 1}
    observation = reset(env)
    @test observation isa Vector{Float32}
    next_observation, reward, done = step(env, randn(Float32, size(action_space(env))))
    @test next_observation isa Vector{Float32}
    @test reward isa Float32
    @test done isa Bool
    close(env)
end

end
