# OpenAIGym_Julia


This is a working/uptodate with new julia implementation of the OpenAIGym.jl that's already around. To run this successfully

Pkg.build("PyCall") - Install this dependency.

You will need a proper python environment setup to use this. 

- Create a new viritualenvironment for python
- Install python version of the gym by using pip install gym
- Download any other games you want to use and make sure you install into them this virutalenv (eg atari). 
- Replace your env path with mine in the example provided