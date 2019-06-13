@testset "agent tests" begin


@testset "construct encoder" begin
    input, latent = 2^7, 2^5
    observation = randn(input)
    os = Box(-1., 1., (input,))
    encoder = NeuralLearner.construct_encoder(os)
    @test length(encoder(observation)) == latent
end


@testset "construct encoder custom latent size" begin
    input, latent = 2^7, 2^3
    observation = randn(input)
    os = Box(-1., 1., (input,))
    encoder = NeuralLearner.construct_encoder(os, latent)
    @test length(encoder(observation)) == latent
end


@testset "construct decoder" begin
    input, latent = 2^7, 2^5
    encoded = randn(latent)
    os = Box(-1., 1., (input,))
    decoder = NeuralLearner.construct_decoder(os)
    @test length(decoder(encoded)) == input
end


@testset "construct decoder custom latent size" begin
    input, latent = 2^7, 2^3
    encoded = randn(latent)
    os = Box(-1., 1., (input,))
    decoder = NeuralLearner.construct_decoder(os, latent)
    @test length(decoder(encoded)) == input
end


@testset "construct policy" begin
    actions, latent = 8, 2^5
    encoded = randn(latent)
    as = Box(-1., 1., (actions,))
    policy = NeuralLearner.construct_policy(as)
    @test length(policy(encoded)) == 2*actions
end


@testset "agent encodes and decodes observation" begin
    env = Environment("Ant-v2")
    agent = Agent(env)
    observation = reset(env)
    encoded = agent.encoder(observation)
    decoded = agent.decoder(encoded)
    @test length(encoded) == 2^5
    @test length(decoded) == length(observation)
end


@testset "agent can select action" begin
    env = Environment("Ant-v2")
    agent = Agent(env)
    observation = reset(env)
    as = action_space(env)
    actions = size(as)[1]
    action = select_action(agent, observation)
    @test length(action) == actions
    @test all(action .> low(as))
    @test all(action .< high(as))
end


@testset "agent can remember transitions" begin
    env = Environment("Ant-v2")
    agent = Agent(env)
    @test length(agent.transitions) == 0
    observation = reset(env)
    action = select_action(agent, observation)
    next_observation, reward, done = step(env, action)
    remember!(agent, Transition(observation, action, reward, next_observation))
    @test length(agent.transitions) == 1
end


end
