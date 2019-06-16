@testset "simulation tests" begin


@testset "simulation can be iterated" begin
    env = Environment("Ant-v2")
    agent = Agent(env)
    sim = Simulation(env, agent)
    transition, state = iterate(sim)
    @test transition isa Transition{Vector{Float32}, Vector{Float32}}
    @test transition.observation isa Vector{Float32}
    @test transition.action isa Vector{Float32}
    @test transition.reward isa Float32
    @test transition.next_observation isa Vector{Float32}
    @test transition.done isa Bool
    @test state === transition.next_observation
end


@testset "simulation is an infinite iterator" begin
    env = Environment("Ant-v2")
    agent = Agent(env)
    sim = Simulation(env, agent)
    @test Base.IteratorSize(sim) == Base.IsInfinite()
    @test eltype(sim) == Transition{Vector{Float32}, Vector{Float32}}
end


@testset "simulation resets environment when episode is finished" begin
    env = Environment("Ant-v2")
    agent = Agent(env)
    sim = Simulation(env, agent)
    transition, state = iterate(sim)
    while !transition.done
        next_transition, state = iterate(sim, state)
        @test transition.next_observation == next_transition.observation
        transition = next_transition
    end
    next_transition, state = iterate(sim, state)
    @test next_transition.done == false
    @test transition.next_observation != next_transition.observation
end


@testset "iterating simulation will cause agent to remember transition" begin
    env = Environment("Ant-v2")
    agent = Agent(env)
    sim = Simulation(env, agent)
    @test length(agent.transitions) == 0
    transition, state = iterate(sim)
    @test length(agent.transitions) == 1
    @test agent.transitions[1] === transition
end


@testset "agent will not improve if there are not enough transitions" begin
    env = Environment("Ant-v2")
    agent = Agent(env)
    sim = Simulation(env, agent)
    for _ ∈ 1:agent.batch_size - 1
        iterate(sim)
        @test isnothing(NeuralLearner.improve!(agent))
    end
    iterate(sim)
    NeuralLearner.improve!(agent)
end

end
