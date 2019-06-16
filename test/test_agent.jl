@testset "agent tests" begin


@testset "construct encoder" begin
    input, latent = 2^7, 2^5
    observation = randn(input)
    os = Box(-1., 1., (input,))
    encoder = NeuralLearner.encoder(NeuralLearner.DefaultToken(), os)
    @test length(encoder(observation)) == latent
end


@testset "construct decoder" begin
    input, latent = 2^7, 2^5
    encoded = randn(latent)
    os = Box(-1., 1., (input,))
    decoder = NeuralLearner.decoder(NeuralLearner.DefaultToken(), os)
    @test length(decoder(encoded)) == input
end


@testset "construct policy" begin
    actions, latent = 8, 2^5
    encoded = randn(latent)
    as = Box(-1., 1., (actions,))
    policy = NeuralLearner.policy(NeuralLearner.DefaultToken(), as)
    @test length(policy(encoded)) == 2actions
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
    select_action(agent, observation)
    @test length(action) == actions
    @test all(action .>= low(as))
    @test all(action .<= high(as))
end


@testset "agent can remember transitions" begin
    env = Environment("Ant-v2")
    agent = Agent(env)
    @test length(agent.transitions) == 0
    observation = reset(env)
    action = select_action(agent, observation)
    next_observation, reward, done = step(env, action)
    transition = Transition(observation, action, reward, next_observation, done)
    remember!(agent, transition)
    @test length(agent.transitions) == 1
end


end
