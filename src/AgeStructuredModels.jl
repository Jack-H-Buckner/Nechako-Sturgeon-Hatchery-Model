"""
    AgeStructure

Simple discrete time matrix models of fish populaitons with density dependent recruitment 
"""
module AgeStructuredModels

using Roots
#using NLsolve

"""
stores parameters for matrix model 
"""
mutable struct model 
    Amax::Int64
    SRCurve
    LeslieMatrix::AbstractMatrix{Float64}
    Survival::AbstractVector{Float64}
    Fecundity#::AbstractVector{Float64}
end 

"""
construct Lesli matrix from fecundity and survival curves 
"""
function build_leslie_matrix(Amax,Survival,Fecundity)
    A = zeros(Amax,Amax)
    A[1,:] = Fecundity
    for i in 1:(Amax-1)
        A[i+1,i] = Survival[i]
    end 
    return A
end 

function init(Amax::Int64,SRCurve,Survival::Function,Fecundity::Function)
    L = buildLeslieMatrix(Amax,Survival.(1:Amax),Fecundity.(1:Amax))
    return model(Amax,SRCurve,L,Survival.(1:Amax),Fecundity.(1:Amax))
end 

function init(Amax::Int64,SRCurve,Survival::AbstractVector{Float64},Fecundity)
    L = build_leslie_matrix(Amax,Survival,Fecundity)
    return model(Amax,SRCurve,L,Survival,Fecundity)
end 
    




function fecundity(N,model)
    @assert length(N) == model.Amax
    F = sum(N.*model.Fecundity)
    return F
end 


function percapita_fecundity(N,model)
    return fecundity(N,model)/sum(N)
end 

        
"""
    stable_age_distribution(mode1)
    
returns equilibrium age structure scaled to one
"""
function stable_age_distribution(model)

    N = zeros(model.Amax)
    Na = 1 
    for i in 1:model.Amax
        N[i] = Na
        Na *= model.Survival[i]
    end 
    return N ./ sum(N)
end 

"""
    LEP(model)

computes life time egg production 
"""
function LEP(model)
    N = stable_age_distribution(model)
    N .*= 1/N[1]
    return fecundity(N,model)
end 
    
"""
    compute_Rstar(model)

computes equilibrium recruitment form age structured model object 
"""
function compute_Rstar(model)
    lep = LEP(model)
    f = x -> model.SRCurve(x) - x/lep
    E_star = Roots.find_zero(f, (1/(lep)^2, 1000*lep))
    return model.SRCurve(E_star)
end

    
"""
    stable_age_structure(model)
    
returns equilibrium age disrtibtuion scaled by abundcae 
"""
function stable_age_structure(model)
    dsn = stable_age_distribution(model)
    R_star = compute_Rstar(model)
    dsn .*= 1/dsn[1]
    return R_star * dsn
end 
    


"""
    quilibrium_size(survival::AbstractVector{Float64}, R_star::Float64, Phi::Function )

survival - vertor probility of surviving from age a to a+1
R_star - Float recruitment rate
Phi - Function if supplied weights the population by age
"""
function equilibrium_size(survival::AbstractVector{Float64}, R_star::Float64, Phi::Function )
    N = R_star
    acc = 0
    for i in 1:length(survival)
        acc += Phi(i)*N
        N *= survival[i]*N
    end 
    return acc
end 

function equilibrium_size(survival::AbstractVector{Float64}, R_star::Float64)
    N = R_star
    acc = 0
    for i in 1:length(survival)
        acc += N
        N *= survival[i]*N
    end 
    return acc
end 

"""
    solve_Rstar(N::Float64, survival::AbstractVector{Float64})

N - float equilibrium population size 
N - survival rate
Amat - int if provided is used as age of maturation, and will assume N is tartegt size of matru population
Phi - funtion if provided is used as a age specific weight for calcualting pop size 
"""
function solve_Rstar(N::Float64, survival::AbstractVector{Float64})
    f = x -> equilibrium_size(survival, x) - N
    R_star = Roots.find_zero(f, 0, 100.0*N)
    return R_star
end 

function solve_Rstar(N::Float64, survival::AbstractVector{Float64}, Amat::Float64)
    Phi = x -> 1. * (x<Amat)
    f = x -> equilibrium_size(survival, x, Phi) - N
    R_star = Roots.find_zero(f, 0, 100.0*N)
    return R_star
end 

function solve_Rstar(N::Float64, survival::AbstractVector{Float64}, Phi::Function)
    f = x -> equilibrium_size(survival, x, Phi) - N
    R_star = Roots.find_zero(f, 0, 100.0*N)
    return R_star
end 



end # module 