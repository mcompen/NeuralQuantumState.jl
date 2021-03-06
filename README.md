# NeuralQuantumState.jl

[![Build Status](https://travis-ci.com/mcompen/NeuralQuantumState.jl.svg?branch=master)](https://travis-ci.com/mcompen/NeuralQuantumState.jl)
[![Coveralls](https://coveralls.io/repos/github/mcompen/NeuralQuantumState.jl/badge.svg?branch=master)](https://coveralls.io/github/mcompen/NeuralQuantumState.jl?branch=master)
[![Docs](https://img.shields.io/badge/docs-dev-red.svg)](https://mcompen.github.io/NeuralQuantumState.jl/dev)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://mcompen.github.io/NeuralQuantumState.jl/stable)


## Description
Solving quantum many-body problems with a Neural Quantum State was first proposed in [1]. This package implements parallel sampling and optimization of many-body wavefunctions of arbitrary Hamiltonians.

[1] Carleo, Giuseppe, and Matthias Troyer. "Solving the quantum many-body problem with artificial neural networks." Science 355.6325 (2017): 602-606.

## Installation
Press `]` in the REPL and simply add the package by typing:
```julia
(v1.X) pkg> add NeuralQuantumState
```

## Example
```julia
julia > using Distributed
julia > addprocs(2)  # Add no. of desired worker processes.
julia > @everywhere using Random
julia > @everywhere using NeuralQuantumState
julia > @sync for i in workers()
            @async remotecall_wait(Random.seed!, i, i * 99999)
        end
julia > NetSettings = NETSETTINGS(
            modelname = "U_afh", # Marshall transformed AFH.
            repetitions =1000,
            n = 6,
            α = 3,
            mag0 = true,
            γ_decay = 0.997,
            mc_trials = 500,
            writetofile=true,
            init_therm_steps = 100,
            therm_steps = 50,
            stat_samples = 2000)
julia > energy = run_NQS(NetSettings)
```
## Info
The author of this package is not affiliated with the authors of the original publication. See [NetKet](https://netket.org) for the official C++ implementation.
