module DomesticationModel

include("DomesticationParameters.jl")
include("MvGaussHermite.jl")
using Distributions, DSP, Roots, Distributions, ForwardDiff

function quadratic_selection(g,s,theta)
    exp( - s/2 * (g-theta)^2 )
end 

Linfty(g,p) = p.L_infty + p.b_Linfty*g
length_(a,g,p) = Linfty(g,p)*(1-exp(-1*p.k*p.a_r))+p.L_0*exp(-1*p.k*p.a_r)
survival(l,p) = 1/(1+exp(-1*p.s_l*(l - p.l50)))

function condition_selection(a,g,p;m=50) 
    mu = [length_(a,g,p)]
    Cov = zeros(1,1)
    Cov[1,1] = 1/p.h2_lr-1
    quad = MvGaussHermite.init_mutable(m,mu,Cov)
    f = l -> survival(l[1],p)
    MvGaussHermite.expected_value(f,quad)
end 

function selection_gradients(p,grid)
    grad_viability = broadcast(g -> quadratic_selection(g,p.s_E,p.theta), grid)
    
    # juvinile selection 
    if p.decreasing_selection
        s_J = p.s_J*log(p.a_r*p.b_sa+1)/p.b_sa
    else
        s_J = p.a_r*p.s_J  
    end
    grad_juvinile = broadcast(g -> quadratic_selection(g,s_J,p.theta), grid)
    grad_condition = broadcast(g -> condition_selection(p.a_r,g,p,m=20), grid)
    grad_fecundity = broadcast(g -> quadratic_selection(g,p.s_F,p.theta), grid)
    return grad_viability, grad_juvinile, grad_condition, grad_fecundity
end 



mutable struct population 
    Amax::Int64
    ### State variables ###
    trait::AbstractMatrix{Float64} # columns are ages 
    grid::AbstractVector{Float64}
    
    ### Age dependent demographic rates ###
    Fa::AbstractVector{Float64}
    Sa::AbstractVector{Float64}
    abundance::AbstractVector{Float64}
    
    ### Genetic parameters ###
    Vr::Float64 
    s::Float64
    theta::Float64
    s_F::Float64
    s_E::Float64
    s_J::Float64
    s_l::Float64
    decreasing_selection::Float64
    b_sa::Float64
    a_r::Float64 
    L_0::Float64
    L_infty::Float64
    k::Float64
    b_Linfty::Float64
    h2_lr::Float64
    l_r50::Float64
    

    ### paramters of implementation of genetic model ###
    grad_viability::AbstractVector{Float64}
    grad_juvinile::AbstractVector{Float64}
    grad_condition::AbstractVector{Float64}
    grad_fecundity::AbstractVector{Float64}
    m::Int64 
    Vr_::AbstractVector{Float64} 
end 


function La(Sa)
    La = broadcast(i ->prod(Sa[1:i]), 1:length(Sa))  # expected survival to age a
    La = vcat([1.0],La[1:(end-1)])
    return La
end 

function Vstar(Vr::Float64,s::Float64)
    a = s
    b = (0.5-Vr*s)
    c = -Vr
    return (-b+sqrt(b^2-4*a*c))/(2*a)
end 


function init(Sa,Fa,parameters;zmin=-5.0,zmax=15.0,dz=0.05,kernel=10.0)
    
    # equilibirum abundance
    Amax = length(Sa)
    abundance = La(Sa)
    abundance = abundance/abundance[1]
    # equilibirum genotype variance 
    V = Vstar(parameters.Vr,parameters.s)
    grid = zmin:dz:zmax
    trait = pdf.(Distributions.Normal(0,V),grid)
    trait = trait./sum(trait)
    trait = transpose(repeat(transpose(trait),Amax))
    
    # selection grids
    grid = zmin:dz:zmax
    grad_viability, grad_juvinile, grad_condition, grad_fecundity = selection_gradients(parameters,grid)
    
    # convolution kernel within famil variance 
    d = Distributions.Normal(0, sqrt(parameters.Vr))
    grid_Vr = collect((-1*kernel*parameters.Vr):dz:(kernel*parameters.Vr))
    Vr_ = pdf.(d,grid_Vr)
    m = length(Vr_)
    
    population(
            # demographic rates
            Amax,trait,grid, Fa, Sa,abundance,

            ### Genetic parameters ###
            parameters.Vr, parameters.s,parameters.theta,parameters.s_F,parameters.s_E,
            parameters.s_J,parameters.s_l,parameters.decreasing_selection,parameters.b_sa,
            parameters.a_r,parameters.L_0,parameters.L_infty,parameters.k,
            parameters.b_Linfty,parameters.h2_lr,parameters.l50,

            ### paramters of implementation of genetic model ###
            grad_viability,grad_juvinile,grad_condition,grad_fecundity,m, Vr_ )
    
end



function init(Sa,Fa,targets, fixed_pars;zmin=-5.0,zmax=15.0,dz=0.05,kernel=10.0)
    
    parameters = DomesticationParameters.solve_parameters(targets, fixed_pars)
    
    # equilibirum abundance
    Amax = length(Sa)
    abundance = La(Sa) ./ sum(La(Sa))
    
    # equilibirum genotype variance 
    V = Vstar(Vr,parameters.s)
    grid = zmin:dz:zmax
    trait = pdf.(Distributions.Normal(0,V),grid)
    trait = trait./sum(trait)
    trait = transpose(repeat(transpose(trait),Amax))
    
    # selection grids
    grid = zmin:dz:zmax
    grad_viability, grad_juvinile, grad_condition, grad_fecundity = selection_gradients(parameters,grid)
    
    # convolution kernel within famil variance 
    d = Distributions.Normal(0, sqrt(parameters.Vr))
    grid_Vr = collect((-1*kernel*parameters.Vr):dz:(kernel*parameters.Vr))
    Vr_ = pdf.(d,grid_Vr)
    m = length(Vr_)
    
    population(
            # demographic rates
            Amax,trait,grid, Fa, Sa,abundance,

            ### Genetic parameters ###
            parameters.Vr, parameters.s,parameters.theta,parameters.s_F,parameters.s_E,
            parameters.s_J,parameters.s_l,parameters.decreasing_selection,parameters.b_sa,
            parameters.a_r,parameters.L_0,parameters.L_infty,parameters.k,
            parametersb_Linfty,parameters.h2_lr,parameters.l_r50,

            ### paramters of implementation of genetic model ###
            grad_viability,grad_juvinile,grad_condition,grad_fecundity,m, Vr_ )
    
end


function rest!(population;zmin=-5.0,zmax=15.0,dz=0.05,kernel=10.0)

    V = Vstar(population.Vr,population.s)
    grid = zmin:dz:zmax
    trait = pdf.(Distributions.Normal(0,V),grid)
    trait = trait./sum(trait)
    trait = transpose(repeat(transpose(trait),population.Amax))
    population.trait = trait
    population
    
end 

function reproduction(population)
    
    # calcualte trait distribution 
    dsn = population.trait * (population.abundance .*  population.Fa)

    # apply fecundity selection to distribution and total 
    dsn = dsn.*population.grad_fecundity
    dsn = dsn ./ sum(dsn)
    
    # convolution - random mating 
    N = length(population.grid)-1
    dsn = DSP.conv(dsn, dsn)[1:2:(2*N+1)]
    dsn = dsn./sum(dsn)

    # convolution inperfect inheritance 
    m = convert(Int64,floor(population.m/2+1))
    dsn = DSP.conv(dsn, population.Vr_)
     
    # normalize distribution 
    dsn= dsn[m:(N+m)]
    dsn = dsn ./ sum(dsn)

    return dsn
end 


function selection!(dsn,population)
    
    dsn .*= population.grad_viability
    dsn .*= population.grad_juvinile
    dsn .*= population.grad_condition
    dsn ./= sum(dsn)
    
    return dsn
end 


function ageing!(population, dsn)
    
    N = length(dsn)
    new_dsn = zeros(N,population.Amax)
    new_dsn[:,2:end] = population.trait[:,1:end-1]
    new_dsn[:,1] = dsn
    
    population.trait .= new_dsn
    
    return population
end


function time_step!(population)
    dsn = reproduction(population)
    selection!(dsn,population)
    ageing!(population, dsn)
end 

end # module 