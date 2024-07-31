"""
ths file provides funciton that define an age structured popualtion model 
that tracks he full age dependent distribution of a trait in the population. 

The code is built around a centeral mutable struct, 'population'. Popualtion stores 
a vector with abundcaes of each age class, and an array that stores the density of the 
trait distribution for each cohort at a set of nodes. In additon, to these states the object
stores paramters that describe the fecudndity and survival of each age class, a stock - recruit
relationship, selection gradient and trait variance. 

Around this basic structure, several methods are defined to update the state of the popuatlion
as as differnt processes occur, such as immigraiton, reproduction, aging and recruitment. 
"""
module AgeTraitStructuredModels

include("AgeStructuredModels.jl")

using Distributions
using DSP
using Plots
using Roots
using FFTW
using Roots


mutable struct population
    # states
    abundanceN::AbstractVector{Float64}
    abundanceH::AbstractVector{Float64}
    traitN::AbstractMatrix{Float64} # columns are ages 
    traitH::AbstractMatrix{Float64}
    grid::AbstractVector{Float64} # nodes where trait is defined
    
    # parameters
    ageStructure
    gradient::AbstractVector{Float64} # age dependent seleciton gradient columsn are ages
    m::Int64 # length of convolution kernel 
    Vle::AbstractVector{Float64} # convolution kernel 
    correction::Float64
    
    Vle_::Float64
    s::Float64
    theta::Float64
end 


mutable struct immigrants
    # states
    N::Float64
    trait::AbstractVector{Float64} # columns are ages 
    grid::AbstractVector{Float64} # nodes where trait is defined
end 



"""
     V_star(Vle, s)

Computes equilibrium variance given severgation varinace Vle/2 and
slection strength s 
"""
function V_star(Vle, s)
    #sigma_s = 1/s
    V_prime = V -> (1/V + s)^(-1)/2 + Vle/2#V -> (V*sigma_s^2/(V + sigma_s^2))/2 + Vle/2
    V0 = Vle
    V1 = V_prime(Vle)
    while abs(V0-V1) > 10^-6
        V0 = V1
        V1 = V_prime(V0)
    end 
    return (1/V1 + s)^(-1)
end 

"""
     V_star_prime(Vle, s)

Computes equilibrium variance given severgation varinace Vle/2 and
slection strength s. measurement after reproduciton but before selection
"""
function V_star_prime(Vle, s)
    #sigma_s = 1/s
    V_prime = V -> (1/V + s)^(-1)/2 + Vle/2#V -> (V*sigma_s^2/(V + sigma_s^2))/2 + Vle/2
    V0 = Vle
    V1 = V_prime(Vle)
    while abs(V0-V1) > 10^-6
        V0 = V1
        V1 = V_prime(V0)
    end 
    return V1
end 

function init(ageStructure, Vle, theta, s, min, max, dx)

    
    # set trait at optimum value and variance = Vle
    grid = collect(min:dx:max)
    d = Distributions.Normal(theta, sqrt(V_star(Vle, s)))
    trait = pdf.(d, grid)
    trait = trait ./sum(trait)
    trait_A = transpose(repeat(transpose(trait),ageStructure.Amax))
    
    traitH = pdf.(d, grid)
    traitH = traitH ./sum(traitH)
    traitH = transpose(repeat(transpose(traitH),ageStructure.Amax))
    
    # Vle
    de = Distributions.Normal(theta, sqrt(Vle/2))
    Vle_ = Vle
    grid_Vle = collect((theta-3*Vle):dx:(theta+3*Vle))
    Vle = pdf.(de,grid_Vle)
    m = length(Vle)
    
    # gradient
    d = Distributions.Normal(theta, sqrt(V_star_prime(Vle_, s)))
    trait = pdf.(d, grid)
    trait = trait ./ sum(trait)
    gradient = exp.(-s/2*(grid .- theta).^2)
    correction = 1/sum(trait .* gradient)
    
    
    #### initialize populaiton state ####

    abundance = AgeStructuredModels.stable_age_structure(ageStructure)
    abundanceH = zeros(ageStructure.Amax)
    #return trait
    return population(abundance, abundanceH, trait_A, traitH, grid, ageStructure, gradient, m, Vle, correction, Vle_, s, theta)
end 




function init_imigrants(population, N, mean)
    grid = population.grid
    d = Distributions.Normal(mean, sqrt(population.Vle_))
    trait = pdf.(d, grid)
    trait = trait ./sum(trait)
    return immigrants(N,trait,grid)
end 


function update_im!(im, pop, N, mu_im)
    d = Distributions.Normal(mu_im, sqrt(pop.Vle_)) 
    trait = pdf.(d, im.grid)
    im.trait = trait ./sum(trait)
    im.N = N
end 


"""
    reset!(population)

resets the popualtion to the initial caonditions specificed by init
"""
function reset!(population)
    # set trait at optimum value and variance = Vle
    d = Distributions.Normal(population.theta, 
                           sqrt( V_star(population.Vle_, population.s)))
    trait = pdf.(d, population.grid)
    trait = trait ./sum(trait)
    trait_A = transpose(repeat(transpose(trait),population.ageStructure.Amax))
    traitH = transpose(repeat(transpose(trait),population.ageStructure.Amax))
    population.traitN = trait_A
    population.traitH = traitH
    populations.abundanceN = AgeStructuredModels.stable_age_structure(population.ageStructure)
    populations.abundanceH = zeros(ageStructure.Amax)
end 


"""
    reset!(population, s)

resets the popualtion to the same initial caonditions specificed by init
and updates the value for the strength of selection 
"""
function reset!(population,s)
    # set trait at optimum value and variance = Vle
    d = Distributions.Normal(population.theta, 
                            sqrt(V_star(population.Vle_,s)))
    #println(sqrt(V_star(population.Vle_,s)))
    trait = pdf.(d, population.grid)
    trait = trait ./sum(trait)
    trait_A = transpose(repeat(transpose(trait),population.ageStructure.Amax))
    traitH = transpose(repeat(transpose(trait),population.ageStructure.Amax))
    population.traitN = trait_A
    population.traitH = traitH
    
    # gradient
    d = Distributions.Normal(population.theta, 
                            sqrt(V_star_prime(population.Vle_,s)))
    trait = pdf.(d, population.grid)
    trait = trait ./ sum(trait)
    trait = trait ./sum(trait)
    
    population.gradient = exp.(-s/2*(population.grid .- population.theta).^2)
    population.correction =  1/sum(trait .* population.gradient)

    population.abundanceN = AgeStructuredModels.stable_age_structure(population.ageStructure)
    population.abundanceH = zeros(population.ageStructure.Amax)
end 


function selection_N(s, theta, mu, V)
    V_prime = (1/V + s)^(-1)
    mu_prime = (mu/V .+ s*theta)*V_prime
    
    p = exp(-1/2*((mu/sqrt(V))^2 + s*theta^2 -(mu_prime/sqrt(V_prime))^2))
    p *= sqrt(V_prime/V)
    return p
end 

function RRS(mu,s)
    V1 = V_star_prime(1, s)
    W1 = selection_N(s, 0, 0, V1)
    W2 = selection_N(s, 0, mu, 1)
    return W2/W1
end 

function solve_trait_difference(RRS_,s)
    return Roots.find_zeros(x -> RRS(x,s) - RRS_,[0,50.0])[1]
end



"""
    reset_immigrants!(im, N, mean)

updates immigants object with new mena vlaue and m
"""
function reset_immigrants!(immigrants,population, N, mean)
    d = Distributions.Normal(mean, sqrt(population.Vle_))
    immigrants.trait = Distributions.pdf.(d, population.grid)
    immigrants.trait= immigrants.trait ./sum(immigrants.trait)
    immigrants.N = N
end 


function reset_immigrants_RRS!(immigrants,population, N, RRS)
    mean = solve_trait_difference(RRS,population.s)
    d = Distributions.Normal(mean, sqrt(population.Vle_))
    immigrants.trait = Distributions.pdf.(d, population.grid)
    immigrants.trait= immigrants.trait ./sum(immigrants.trait)
    immigrants.N = N
end 


"""
returns trait distribution for new age class and 
the total spawning stock fecundity.
"""
function reproduction_fft(population)

    f_totalN = sum(population.abundanceN .* population.ageStructure.Fecundity)
    f_totalH = sum(population.abundanceH .* population.ageStructure.Fecundity)
    
    dsnN = population.traitN * (population.abundanceN .*  population.ageStructure.Fecundity)# ./ f_totalN
    dsnH = population.traitH * (population.abundanceH .*  population.ageStructure.Fecundity) #./ f_totalH
    
    dsn = (dsnN + dsnH)./ (f_totalN+f_totalH)
    
    # convolution - random mating 
    N = length(population.grid)-1
    dsn_zs = vcat(dsn,zeros(length(dsn)))
    dsn = real(ifft(fft(dsn_zs).*fft(dsn_zs)))[1:2:(2*N+1)]#DSP.conv(dsn, dsn)[1:2:(2*N+1)]
    dsn = dsn./sum(dsn)
    


    # convolution inperfect inheritance 
    m = convert(Int64,floor(population.m/2+1))
    dsn = DSP.conv(dsn, population.Vle)
 
    dsn= dsn[m:(N+m)]
    dsn = dsn ./ sum(dsn)

 
    return dsn, f_total
end 

function reproduction(population)

    f_totalN = sum(population.abundanceN .* population.ageStructure.Fecundity)
    f_totalH = sum(population.abundanceH .* population.ageStructure.Fecundity)
    
    dsnN = population.traitN * (population.abundanceN .*  population.ageStructure.Fecundity)# ./ f_totalN
    dsnH = population.traitH * (population.abundanceH .*  population.ageStructure.Fecundity) #./ f_totalH
    
    dsn = (dsnN + dsnH)./ (f_totalN+f_totalH)

    #Plots.plot!(population.grid,dsn )
    # convolution - random mating 
    N = length(population.grid)-1
    dsn = DSP.conv(dsn, dsn)[1:2:(2*N+1)]
    dsn = dsn./sum(dsn)
    

    #Plots.plot!(population.grid,dsn )

    # convolution inperfect inheritance 
    m = convert(Int64,floor(population.m/2+1))
    dsn = DSP.conv(dsn, population.Vle)
 
    dsn= dsn[m:(N+m)]
    dsn = dsn ./ sum(dsn)

 
    return dsn, f_totalN + f_totalH
end 


"""
selection on juviniles 
"""
function selection(dsn, N, population)
    dsn = dsn .* population.gradient
    survival = sum(dsn)
    N = N * survival
    dsn = dsn ./ survival
    return dsn, N
end 


"""
Beverton Holt recruitment 
assumes that selection occurs after recruitment 
"""
function recruitment(f_total, population )
    R = population.correction * population.ageStructure.SRCurve(f_total) #
    return R 
end 

"""
Beverton Holt recruitment 

f_total - number of eggs
populations - populaiton objct for parameters
before - boolian does selection occur before or after selection 
"""
function recruitment(f_total, population, before)
    if before 
        f_total *= population.correction 
        R = population.ageStructure.SRCurve(f_total)
    else
        R = population.correction * population.ageStructure.SRCurve(f_total)
    end 
    return R 
end 



# """
#     immigration(dsn,N, immigrants)

# Immigration of juviniles. This function is designed for populations 
# """
# function immigration(dsn,immigrants)
#     N_total = N + immigrants.N
#     p = N/N_total
#     dsn = p.*dsn .+ (1-p).* immigrants.trait
#     return dsnN, dsnH,NN, NH
# end 




"""
returns trait distribution for new age class and 
the total spawning stock fecundity.
"""
function selection_and_reproduction(population)
    
    S = transpose(repeat(transpose(population.gradient),population.A_max))
    
    dsn = population.trait .* S * (population.abundance .*  population.fecundity) 
    f_total = sum(dsn)
    dsn = dsn./f_total
    
    
    # convolution - random mating 
    N = length(population.grid)-1
    dsn = DSP.conv(dsn, dsn)[1:2:(2*N+1)]
    dsn = dsn./sum(dsn)
    
    # convolution inperfect inheritance 
    m = convert(Int64,floor(population.m/2+1))
    dsn = DSP.conv(dsn, population.Vle)[m:(N+m)]
    dsn = dsn ./ sum(dsn)
    
    return dsn, f_total
end 



 

"""
Updates the age structure of the popuatlion and adds recruits
"""
function ageing!(population, R, dsn_R)
    
    population.abundanceN = population.abundanceN .* population.ageStructure.Survival
    population.abundanceH = population.abundanceH .* population.ageStructure.Survival

    new_N = zeros(population.ageStructure.Amax)
    new_N[1] = R 
    new_N[2:end] = population.abundanceN[1:end-1]
    
    
    new_H = zeros(population.ageStructure.Amax)
    new_H[1] = 0 
    new_H[2:end] = population.abundanceH[1:end-1]
    
    N = length(dsn_R)
    new_dsnN = zeros(N,population.ageStructure.Amax)
    new_dsnN[:,2:end] = population.traitN[:,1:end-1]
    new_dsnN[:,1] = dsn_R
    
    new_dsnH = zeros(N,population.ageStructure.Amax)
    new_dsnH[:,2:end] = population.traitH[:,1:end-1]
    new_dsnH[:,1] = zeros(length(dsn_R))
    
    population.abundanceN = new_N
    population.traitN = new_dsnN
    
    population.abundanceH = new_H
    population.traitH = new_dsnH
    
end 

"""
Updates the age structure of the popuatlion and adds recruits
"""
function ageing!(population, R, dsn_R, NH, dsnH)
    
    population.abundanceN = population.abundanceN .* population.ageStructure.Survival
    population.abundanceH = population.abundanceH .* population.ageStructure.Survival

    new_N = zeros(population.ageStructure.Amax)
    new_N[1] = R 
    new_N[2:end] = population.abundanceN[1:end-1]
    
    new_H = zeros(population.ageStructure.Amax)
    new_H[1] = NH
    new_H[2:end] = population.abundanceH[1:end-1]
    
    
    N = length(dsn_R)
    new_dsnN = zeros(N,population.ageStructure.Amax)
    new_dsnN[:,2:end] = population.traitN[:,1:end-1]
    new_dsnN[:,1] = dsn_R
    
    new_dsnH = zeros(N,population.ageStructure.Amax)
    new_dsnH[:,2:end] = population.traitH[:,1:end-1]
    new_dsnH[:,1] = dsnH

    
    population.abundanceN = new_N
    population.traitN = new_dsnN
    
    population.abundanceH = new_H
    population.traitH = new_dsnH
end 

"""
    simulates dynamics and updates the correction factor once trait dsn has equilibrated 
"""
function update_correction!(population)
    for i in 1:1000
        dsn, R = reproduction(population)
        R = recruitment(R, population )
        dsn, R = selection(dsn, R, population)
        ageing!(population, R, dsn)
    end 
    dsn, R = reproduction(population)
    dsn, R = selection(dsn, 1, population)
    population.correction = 1/R
    population.abundance = AgeStructuredModels.stable_age_structure(population.ageStructure)
end 

###################################
### Demographic stochasticity   ###
###################################
using Distributions

"""
selection on juviniles 
"""
function selection_S(dsn, N, population)
    dsn = dsn .* population.gradient
    survival = sum(dsn)
    N = rand(Distributions.Binomial(N, survival),1)[1]
    dsn = dsn ./ survival
    return dsn, N
end 


"""
Beverton Holt recruitment 

f_total - number of eggs
populations - populaiton objct for parameters
before - boolian does selection occur before or after selection 
sigma - optional amount of environmetnal variability 
"""
function recruitment_S(f_total, population, before)
    if before 
        f_total *= population.correction 
        R = population.ageStructure.SRCurve(f_total)
    else
        R = population.correction * population.ageStructure.SRCurve(f_total)
    end 

    R = rand(Distributions.Poisson(R),1)[1]
    return R 
end 

function recruitment_S(f_total, population, before, sigma)
    if before 
        f_total *= population.correction 
        R = population.ageStructure.SRCurve(f_total)
    else
        R = population.correction * population.ageStructure.SRCurve(f_total)
    end 
    epsilon_t = rand(Distributions.Normal(-0.5*sigma^2,sigma),1)[1]
    R = rand(Distributions.Poisson(R*exp(epsilon_t)) ,1)[1]
    return R 
end 



"""
    immigration(dsn,N, immigrants)

Immigration of juviniles. This function is designed for populations 
"""
function immigration_S(dsn,N, immigrants)
    N_total = N + immigrants.N
    p = N/N_total
    dsn = p.*dsn .+ (1-p).* immigrants.trait
    return dsn, rand(Distributions.Poisson(N_total),1)[1]
end 

"""
Updates the age structure of the popuatlion and adds recruits
"""
function ageing_S!(population, R, dsn_R)
    
    for i in 1:length(population.abundance)
        Na = rand(Distributions.Binomial(floor(Int,population.abundance[i]), population.ageStructure.Survival[i]))[1]
        population.abundance[i] = Na
    end 

    new_A = zeros(population.ageStructure.Amax)
    new_A[1] = R 
    

    
    new_A[2:end] = population.abundance[1:end-1]
    
    N = length(population.grid)
    new_dsn = zeros(N,population.ageStructure.Amax)
    new_dsn[:,1] = dsn_R

    
    new_dsn[:,2:end] = population.trait[:,1:end-1]
    
    population.abundance = new_A
    population.trait = new_dsn
    
end 


###################################
### time step/ update functions ###
###################################

"""
    time_step_DSI!(populations; immigrants)
updates the popualtion with density dependnece before selection. If immigrants are included
immigration occurs last
"""
function time_step_DSI!(population)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment(R, population )
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 


function time_step_DSI!(population, immigrants)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment(R, population )
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    AgeTraitStructuredModels.ageing!(population, R, dsn,immigrants.N, immigrants.trait)
end 

function time_step_DSI!(population, immigrants, pim)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment(R, population )
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    AgeTraitStructuredModels.ageing!(population, R, dsn, R*pim, immigrants.trait)
end 

"""
    time_step_DIS!(populations; immigrants)
updates the popualtion with density dependnece before selection. If immigrants are included
immigration occurs before selection.
"""
function time_step_DIS!(population)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment(R, population )
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 


function time_step_DIS!(population, immigrants)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment(R, population )
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    dsnH, RH = AgeTraitStructuredModels.selection(immigrants.trait, immigrants.N, population)
    AgeTraitStructuredModels.ageing!(population, R, dsn, RH, dsnH)
end 



"""
    time_step_IDS!(populations; immigrants)
updates the popualtion with density dependnece before selection. If immigrants are included
immigration occurs before density dependence.
"""
function time_step_IDS!(population)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment(R, population )
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 


function time_step_IDS!(population, immigrants)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    dsn, R = AgeTraitStructuredModels.immigration(dsn, R, immigrants)
    R = AgeTraitStructuredModels.recruitment(R, population )
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 



"""
    time_step_ISD!(populations; immigrants)
updates the popualtion with selection before density dependnece. If immigrants are included
immigration occurs before selection.
"""
function time_step_ISD!(population)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    R = AgeTraitStructuredModels.recruitment(R, population, true )
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 


function time_step_ISD!(population, immigrants)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    dsn, R = AgeTraitStructuredModels.immigration(dsn, R, immigrants)
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    R = AgeTraitStructuredModels.recruitment(R, population, true )
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 


"""
    time_step_ISD!(populations; immigrants)
updates the popualtion with selection before density dependnece. If immigrants are included
immigration occurs before selection.
"""
function time_step_SDI!(population)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    R = AgeTraitStructuredModels.recruitment(R, population, true )
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 


function time_step_SDI!(population, immigrants)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    R = AgeTraitStructuredModels.recruitment(R, population, true )
    dsn, R = AgeTraitStructuredModels.immigration(dsn, R, immigrants)
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 


"""
    time_step_ISD!(populations; immigrants)
updates the popualtion with selection before density dependnece. If immigrants are included
immigration occurs before selection.
"""
function time_step_SID!(population)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    R = AgeTraitStructuredModels.recruitment(R, population, true )
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 


function time_step_SID!(population, immigrants)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    dsn, R = AgeTraitStructuredModels.selection(dsn, R, population)
    dsn, R = AgeTraitStructuredModels.immigration(dsn, R, immigrants)
    R = AgeTraitStructuredModels.recruitment(R, population, true )
    AgeTraitStructuredModels.ageing!(population, R, dsn)
end 

###################################
### Demographic stochasticity   ###
###################################

"""
    time_step_DSI!(populations; immigrants)
updates the popualtion with density dependnece before selection. If immigrants are included
immigration occurs last
"""
function time_step_DSI_S!(population)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment_S(R, population, false )
    dsn, R = AgeTraitStructuredModels.selection_S(dsn, R, population)
    AgeTraitStructuredModels.ageing_S!(population, R, dsn)
end 


function time_step_DSI_S!(population, immigrants)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment_S(R, population, false )
    dsn, R = AgeTraitStructuredModels.selection_S(dsn, R, population)
    dsn, R = AgeTraitStructuredModels.immigration_S(dsn, R, immigrants)
    AgeTraitStructuredModels.ageing_S!(population, R, dsn)
end 

function time_step_DSI_S!(population, mu, sigma::Float64)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment_S(R, population, false,sigma )
    dsn, R = AgeTraitStructuredModels.selection_S(dsn, R, population)
    AgeTraitStructuredModels.ageing_S!(population, R, dsn)
end 


function time_step_DSI_S!(population, immigrants, mu, sigma::Float64)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment_S(R, population, false,sigma )
    dsn, R = AgeTraitStructuredModels.selection_S(dsn, R, population)
    dsn, R = AgeTraitStructuredModels.immigration_S(dsn, R, immigrants)
    AgeTraitStructuredModels.ageing_S!(population, R, dsn)
end 
"""
    time_step_DIS!(populations; immigrants)
updates the popualtion with density dependnece before selection. If immigrants are included
immigration occurs before selection.
"""
function time_step_DIS_S!(population)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment_S(R, population, false )
    dsn, R = AgeTraitStructuredModels.selection_S(dsn, R, population)
    AgeTraitStructuredModels.ageing_S!(population, R, dsn)
end 


function time_step_DIS_S!(population, immigrants)
    dsn, R = AgeTraitStructuredModels.reproduction(population)
    R = AgeTraitStructuredModels.recruitment_S(R, population, false )
    dsn, R = AgeTraitStructuredModels.immigration_S(dsn, R, immigrants)
    dsn, R = AgeTraitStructuredModels.selection_S(dsn, R, population)
    AgeTraitStructuredModels.ageing_S!(population, R, dsn)
end



## analysis
"""
This funcction computes fittness in terms of the average 
probability of survival of juviniles produved by a popualtion
assuming selection acts on the survival of juviniles to recruitment. 
"""
function fittness(population)
    dsn, f = reproduction(population)
    W = sum(dsn .* population.gradient)
    return W
end 


# 
## analysis
"""
computes the number of individuals of age 1 in the population
"""
function recruitment(population)
    return population.abundanceN[1]
end 


"""
computes the number of individuals of age 1 in the population
"""
function spawning_stock(population)
    max_F = argmax(population.ageStructure.Fecundity)
    return sum(population.abundanceN .* population.ageStructure.Fecundity) ./ population.ageStructure.Fecundity[max_F]
end 
"""
computes the number of individuals of age 1 in the population
"""
function spawning_stockH(population)
    max_F = argmax(population.ageStructure.Fecundity)
    return sum(population.abundanceH .* population.ageStructure.Fecundity) ./ population.ageStructure.Fecundity[max_F]
end 

"""
    trait_moments_spawning(population)

returns the first two moments of the spawning stocks trait distirbution
"""
function trait_moments_spawning(population)
    f_total = sum(population.abundanceN .* population.ageStructure.Fecundity)
    dsn = population.traitN * (population.abundanceN .* 
                population.ageStructure.Fecundity) ./ f_total
    mu = sum(dsn .* population.grid)
    sigma = sum(dsn .* (population.grid .- mu).^2)
    return mu, sqrt(sigma)
end 


"""
    trait_moments_spawning(population)

returns the first two moments of the spawning stocks trait distirbution
"""
function trait_moments_born(population)
    dsn, R = reproduction(population)
    mu = sum(dsn .* population.grid)
    sigma = sum(dsn .* (population.grid .- mu).^2)
    return mu, sqrt(sigma)
end 

"""
    trait_moments_recruits(population)

returns the first two moments of the age one trait distirbution
"""
function trait_moments_recruits(population)
    dsn = population.traitN[:,1] ./ sum(population.traitN[:,1] )
    mu = sum(dsn .* population.grid)
    sigma = sum(dsn .* (population.grid .- mu).^2)
    return mu, sqrt(sigma)
end 






##### functions for analysis 

function equilibrium(population, update!, immigrants,F)
    W0 = 10
    W1 = AgeTraitStructuredModels.spawning_stock(population)
    iter = 0
    while (iter < 200) | (abs(W0 - W1) > 10^-7.5)
        iter += 1
        W0 = W1
        population.abundanceH .= exp(-1*F).*population.abundanceH
        update!(population, immigrants)
        W1 = fittness(population)
    end 
    #println(iter)
    μrec, σrec = trait_moments_recruits(population)
    μSSB, σSSB = trait_moments_spawning(population)
    W = fittness(population)
    SSB = spawning_stock(population)
    rec = recruitment(population)
    return W #, SSB, rec#, μrec, σrec, μSSB, σSSB
end  


# before if immigration before density dependence 
function min_outcomes(population,  update!, immigrants, T, F)

    F0 = 10
    F1 = fittness(population)
    iter = 0

    while (iter < T-5) |(F0 < F1) #| flip
        iter += 1

        F0 = F1
        if iter < T
            population.abundanceH .= exp(-1*F).*population.abundanceH
            update!(population, immigrants)
        else
            population.abundanceH .= exp(-1*F).*population.abundanceH
            update!(population)
        end
             
        F1 = fittness(population)
  
    end 
    #println(iter)
    return F1 
end 





end # module 




#function init_population(A_max, survival, fecundity, r , K, theta, s, min, max, dx, Vle)
    
#     # set age distribution - equilibrium 
#     LEP = age_structure_model.LEP(1.0, survival, fecundity, A_max)
    
#     a = r/LEP
#     b = a/K
#     f(x) = a*x/(1+b*x)
#     g(x )= f(x) - x/LEP
#     print(LEP)
#     x_eq = Roots.find_zero(g, (10^-6, 10^6*K*LEP))
#     y_eq = f(x_eq)
    
#     abundance = zeros(A_max)
#     N = y_eq
#     for i in 1:A_max
#         abundance[i] = N
#         N *= survival[i]
#     end 

#     # set trait at optimum value and variance = Vle
#     grid = collect(min:dx:max)
#     d = Distributions.Normal(theta, sqrt(V_star(Vle, s)))
#     trait = pdf.(d, grid)
#     trait = trait ./sum(trait)
#     trait_A = transpose(repeat(transpose(trait),A_max))
    
    
#     # Vle
#     de = Distributions.Normal(theta, sqrt(Vle/2))
#     grid_Vle = collect((theta-3*Vle):dx:(theta+3*Vle))
#     Vle = pdf.(de,grid_Vle)
#     m = length(Vle)
    
#     # gradient
#     gradient = exp.(-s/2*(grid .- theta).^2)
#     correction = 1/sum(trait .* gradient)
    
#     pop = population(abundance, trait_A, grid, fecundity, survival,A_max, a, b, correction, gradient,m,Vle)
    
#     return pop
# end 



# function init_population_2(A_max, survival, fecundity, R_star, alpha , beta, theta, s, min, max, dx, Vle, correction)
    
#     # set age distribution - equilibrium 
#     LEP = age_structure_model.LEP(1.0, survival, fecundity, A_max)
    
    
#     abundance = zeros(A_max)
#     N = R_star
#     for i in 1:A_max
#         abundance[i] = N 
#         N *= survival[i]
#     end 
#     # set trait at optimum value and variance = Vle 
#     grid = collect(min:dx:max)
#     d = Distributions.Normal(theta, sqrt(V_star(Vle, s)))
#     trait = pdf.(d, grid)
#     trait = trait ./sum(trait)
#     trait_A = transpose(repeat(transpose(trait),A_max))
    
    
#     # Vle
#     de = Distributions.Normal(theta, sqrt(Vle/2))
#     grid_Vle = collect((theta-3*Vle):dx:(theta+3*Vle))
#     Vle = pdf.(de,grid_Vle)
#     m = length(Vle)
    
#     # gradient
#     if correction
#         gradient = exp.(-s/2*(grid .- theta).^2)
#         correction = 1/sum(trait .* gradient)
#     else
#         correction = 1
#     end 
    
    
    
#     pop = population(abundance, trait_A, grid, fecundity, survival,A_max, alpha, beta, correction, gradient,m,Vle)

#     return pop
# end 
