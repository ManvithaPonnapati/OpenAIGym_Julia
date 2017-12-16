
 __precompile__()

module OpenAIGym

using PyCall
using Reexport
@reexport using Reinforce
import Reinforce:
    MouseAction, MouseActionSet,
    KeyboardAction, KeyboardActionSet

abstract type AbstractGymEnv<:AbstractEnvironment end
export
    gym,
    GymEnv

const _py_envs = Dict{String,Any}()

abstract type AbstractGymEnv<:AbstractEnvironment end


"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
type GymEnv <: AbstractGymEnv
    name::String
    pyenv 
    state
    reward::Float64
    actions::AbstractSet
    done::Bool
    info::Dict
    GymEnv(name,pyenv) = new(name,pyenv)
end
GymEnv(name) = gym(name)

function Reinforce.reset!(env::GymEnv)
    env.state = env.pyenv[:reset]()
    env.reward = 0.0
    env.actions = actions(env, nothing)
    env.done = false
end

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
type UniverseEnv <: AbstractGymEnv
    name::String
    pyenv  # the python "env" object
    state
    reward
    actions::AbstractSet
    done
    info::Dict
    UniverseEnv(name,pyenv) = new(name,pyenv)
end
UniverseEnv(name) = gym(name)

function Reinforce.reset!(env::UniverseEnv)
    env.state = env.pyenv[:reset]()
    env.reward = [0.0]
    env.actions = actions(env, nothing)
    env.done = [false]
end

function gym(name::AbstractString)
    env = if name in ("Soccer-v0", "SoccerEmptyGoal-v0")
        @pyimport gym_soccer
        get!(_py_envs, name) do
            GymEnv(name, pygym[:make](name))
        end
    elseif split(name, ".")[1] in ("flashgames", "wob")
        @pyimport universe
        @pyimport universe.wrappers as wrappers
        if !isdefined(OpenAIGym, :vnc_event)
            global const vnc_event = PyCall.pywrap(PyCall.pyimport("universe.spaces.vnc_event"))
        end
        get!(_py_envs, name) do
            pyenv = wrappers.SafeActionSpace(pygym[:make](name))
            pyenv[:configure](remotes=1) 
            o = UniverseEnv(name, pyenv)
            sleep(2)
            o
        end
    else
        GymEnv(name, pygym[:make](name))
    end
    reset!(env)
    env
end

function actionset(A::PyObject)
    if haskey(A, :n)
        DiscreteSet(0:A[:n]-1)
    elseif haskey(A, :spaces)
        sets = [actionset(a) for a in A[:spaces]]
        TupleSet(sets...)
    elseif haskey(A, :high)
        IntervalSet{Vector{Float64}}(A[:low], A[:high])
    elseif haskey(A, :buttonmasks)
        keyboard = KeyboardActionSet(A[:keys])
        buttons = DiscreteSet(Int[bm for bm in A[:buttonmasks]])
        width,height = A[:screen_shape]
        mouse = MouseActionSet(width, height, buttons)
        TupleSet(keyboard, mouse)
    elseif haskey(A, :actions)
        TupleSet(DiscreteSet(A[:actions]))
    else
        @show A
        @show keys(A)
        error("Unknown actionset type: $A")
    end
end


function Reinforce.actions(env::AbstractGymEnv, s′)
    actionset(env.pyenv[:action_space])
end

pyaction(a::Vector) = Any[pyaction(ai) for ai=a]
pyaction(a::KeyboardAction) = Any[a.key]
pyaction(a::MouseAction) = Any[vnc_event.PointerEvent(a.x, a.y, a.button)]
pyaction(a) = a

function Reinforce.step!(env::GymEnv, s, a)
    pyact = pyaction(a)
    s′, r, env.done, env.info = env.pyenv[:step](pyact)
    env.reward, env.state = r, s′
end

function Reinforce.step!(env::UniverseEnv, s, a)
    pyact = Any[pyaction(a)]
    s′, r, env.done, env.info = env.pyenv[:step](pyact)
    env.reward, env.state = r, s′
end

Reinforce.finished(env::GymEnv, s′) = env.done
Reinforce.finished(env::UniverseEnv, s′) = all(env.done)

function main()
    @static if is_linux()
        condadir = Pkg.dir("Conda","deps","usr","lib")
        Libdl.dlopen(joinpath(condadir, "libssl.so"))
        Libdl.dlopen(joinpath(condadir, "python2.7", "lib-dynload", "_ssl.so"))
    end

    global const pygym = pyimport("gym")
end

main()

end 


