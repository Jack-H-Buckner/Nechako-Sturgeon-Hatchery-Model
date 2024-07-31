module GeneticDemographicModel

#######################
###  load packages  ###
#######################

using DSP
using Roots
using Distributions 
using ForwardDiff

###############################
###  Mathematical analysis  ###
###############################

# Genetic model equilibrium 

"""
    Vstar(Vr,s)

Calcualtes the equilibrum variance of the trait distribution 
in balance between stabalizing selection and variance unlocked
by recombination. population sensused after 

Vr - within family variance 
s - summed strength of selection at each life stage
"""
function Vstar(Vr::Float64,s::Float64)
    a = s
    b = (0.5-Vr*s)
    c = -Vr
    return (-b+sqrt(b^2-4*a*c))/(2*a)
end 

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

# RRS 
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
   p_survival(V,z,s)

Calcualtes the proportion of individuals that survive viability selection
or the amount of reproduction compated to a popuatlion of individuals at 
the fitness optimum for fecundity selection. 
"""
function p_survival(V,z,s)
    return ((s*V+1)^(-1/2))*exp(-0.5*V^-1*(1-(s*V+1)^(-1))*z^2)
end 

function dsn_update(p,zbar,V,s)
    Vprime = (V^(-1)+s)^(-1)
    zprime = zbar*Vprime/V
    pprime = p_survival(V,zbar,s)
    return p*pprime, zprime, Vprime
end
# Demographic equilibrium

"""
    Ricker(X,b)

density depndent survival with Ricker functional form
"""
Ricker(X,b) = exp(-b*X)

"""
    BevertonHolt(X,b)

density depndent survival with Beverton Holt functional form
"""
BevertonHolt(X,b) = 1/(1+b*X)


# weight functions for intra cohort competition
"""
    Calc_Weights(Wa,r)

calculate weights 
Wa - weight at age 
r - nice over lap 
"""
function CalcWeights(Wa,r)
    ba = Wa .*r.^(1:length(Wa))
    return ba
end 

# equilibrium analysis 
"""
    La(Sa)

probaiblity survival to age a givne survival rate between age classes

Sa - survival from age a to age a+1 at fitness optimum
"""
function La(Sa)
    La = broadcast(i ->prod(Sa[1:i]), 1:length(Sa))  # expected survival to age a
    La = vcat([1.0],La[1:(end-1)])
    return La
end 

"""
    LEP(Sa,Fa)

calcualtes life time egg production 

Sa - survival from age a to age a+1 at fitness optimum
Fa - fecundity at age a at fitness optimum
"""
function LEP(Sa,Fa)
    @assert length(Sa) == length(Fa)
    return sum(La(Sa).*Fa)
end 

"""
    solve_S0(S0,R0,Sa,Fa,s,Vr)

objective functio used to solve for density independent juvinile survival that yields a specified
R0, given the strength of selection and other demographic rates. Returns zero when correct value of 
S0 is supplied 

S0 - candidate value of survival to age 0 for individual at fitness optimum
R0 - target value of R0
Sa - survival from a to age a+1 starting at age 1 for individual at fitness optimum
Fa - fecundity (egg production) at age starting at age 1 for individual at fitness optimum
s - strength of selection at each life phase including fecundity selection
Vr - within family variance 
"""
function solve_S0(S0,R0,Sa,Fa,s,Vr)
    lep = LEP(Sa,Fa)
    V = Vstar(Vr,s)
    f = p_survival(V,0,s)
    return R0 - lep*S0*f
end 

"""
    S0(R0,Sa,Fa,s,Vr)

The value of S0 (survival from egg to age one at fitness otpimum and no density dependence)
that produces a given value of R0 (reproductive number). 

R0 - reproductive number
Sa - survival from a to age a+1 starting at age 1 for individual at fitness optimum
Fa - fecundity (egg production) at age starting at age 1 for individual at fitness optimum
s - strength of selection at each life phase including fecundity selection
Vr - within family variance 
"""
function S0_(R0,Sa,Fa,s,Vr)
    sol = Roots.find_zero(x -> solve_S0(x,R0,Sa,Fa,s,Vr),[0.0,1.0])
    return sol
end 

"""
    solve_b(b,Rstar,Sa,Fa,S0,s1,s2,s3,Vr,Sd,X_weights)

solves for density dependence paramter b the produces a given value of Rstar
given the other demographic paramters of the model. The popoulation is assumed 
to be at the fitness opimum. returns zero when b is correct value

b - strength of density dependence
Rstar - target value of recruitment at equilibrium
Sa - survival from a to age a+1 starting at age 1 for individual at fitness optimum
Fa - fecundity (egg production) at age starting at age 1 for individual at fitness optimum
s1 - viabiliyt selection before density dependence
s2 - viability selection after density dependence
s3 - fecundity selection
Vr - within family trait variance 
Sd - Sd(b*[S0f1*Rstar +X]) -> survival rate 
ba - weights to calcaulte density dependent contributin from each age class
"""
function solve_b(b,Rstar,R0,Sa,Fa,S0,s1,s2,s3,Vr,Sd,ba,b0)      
    lep = sum(La(Sa) .* Fa)
    V0 = Vstar(Vr,s1+s2+s3) # variance before selection
    V1 = (V0^(-1)+s1)^(-1) # variance after viability selection 1
    V2 = (V1^(-1)+s2)^(-1) # variance after viability selection 2
    f1 = p_survival(V0,0,s1)
    f2 = p_survival(V1,0,s2)
    f3 = p_survival(V2,0,s3)
    X = S0*f1*f3*Rstar*lep*b0 + Rstar.*sum(La(Sa).*ba)
    return 1 - Sd(X,b)*R0
end


"""
    b(Rstar,Sa,Fa,S0,s1,s2,s3,Vr,Sd,X_weights)

return value of b to given value of Rstar and other dempgraphic rates and 
selection strengths. 

Rstar - target value of recruitment at equilibrium
Sa - survival from a to age a+1 starting at age 1 for individual at fitness optimum
Fa - fecundity (egg production) at age starting at age 1 for individual at fitness optimum
s1 - viabiliyt selection before density dependence
s2 - viability selection after density dependence
s3 - fecundity selection
Vr - within family trait variance 
Sd - Sd(b*[S0f1*Rstar +X]) -> survival rate 
ba - weights to calcaulte density dependent contributin from each age class
"""
function b_(Rstar,R0,Sa,Fa,S0,s1,s2,s3,Vr,Sd,ba,b0)
    sol = Roots.find_zero(x -> solve_b(x,Rstar,R0,Sa,Fa,S0,s1,s2,s3,Vr,Sd,ba,b0),[0.0,10.0^10])
    return sol
end 



############################################################
###  Structs to store paramters and state of population  ###
############################################################

"""
    population 

data for model of age structured popuatlion under stabalizing seleciton.

abundanceN - AbstractVector{Float64} * Abundance at age natural origin
abundanceH - AbstractVector{Float64} * Abundance at age hatchery origin
traitN - AbstractMatrix{Float64} * Trait distribution natural origin 
traitH - AbstractMatrix{Float64} * Trait distribution hatchery origin
grid - AbstractVector{Float64} * Grid values for trait distribution aproximation 
Sa - AbstractVector{Float64} * Probability of survival from age a to a+1 at fitness optimum 
Fa - AbstractVector{Float64} * Fecundity at age a at fitness optimum 
S0 - Float64 * Density independent survival from egg to age 1 at fitness optimum 
ba - AbstractVector{Float64} * Contribution to density dependence at age a
Sd - Function * probability of survival to age one given weighted abundance X
Vr - Float64 * with in famil variance 
s1 - Float64 * strength of viability selection before density dependence 
s2 - Float64 * strength of viability selection after density dependence 
s3 - Float64 * strength of fecundity selection 
gradient - AbstractVector{Float64} 
m - Int64 
Vle- AbstractVector{Float64} 
"""
mutable struct population 
    Amax::Int64
    ### State variables ###
    abundanceN::AbstractVector{Float64}
    abundanceH::AbstractVector{Float64}
    traitN::AbstractMatrix{Float64} # columns are ages 
    traitH::AbstractMatrix{Float64}
    grid::AbstractVector{Float64}
    
    ### Age dependent demographic rates ###
    Sa::AbstractVector{Float64}
    Fa::AbstractVector{Float64}
    
    ### hatchery plastic effects ###
    rFaplastic::Float64
    
    ### Density dependence ###
    S0::Float64
    ba::AbstractVector{Float64}
    b0::Float64
    Sd::Function
    b::Float64
    Rstar::Float64
    
    ### Genetic parameters ###
    Vr::Float64 
    s1::Float64 
    s2::Float64
    s3::Float64
    
    ### paramters of implementation of genetic model ###
    gradient1::AbstractVector{Float64}
    gradient2::AbstractVector{Float64}
    gradient3::AbstractVector{Float64}
    m::Int64 
    Vr_::AbstractVector{Float64} 
end 


 

"""
    init(Sa,Fa,Wa,R0,Rstar,Sd,r,Vr,s1,s2,s3)

Initializes the population model object 

Sa - Survival from age a to age a+1 at fitness optimum
Fa - Fecundity at age a ot fitness optimum 
Wa - weight at age a
R0 - reproductive number 
Rstar - equilibrium recruitment 
Sd - density dependence function Sd(X,b)
r - rate of niche shift
Vr - within family variance 
s1 - selection strength viability before density dependence 
s2 - selection strength viability after density dependence 
s3 - selection strength fecundity selection 
"""
function init(Sa,Fa,Wa,R0,Rstar,Sd,r,Vr,s1,s2,s3,rFaplastic;zmin=-5.0,zmax=15.0,dz=0.05,kernel=10.0)
    ### solve for juvinile survival rates ###
    ba = CalcWeights(Wa[2:end],r) # weights for intra cohort competition 
    b0 = Wa[1]
    S0 = S0_(R0,Sa,Fa,s1+s2+s3,Vr) # density independent survival 
    b = b_(Rstar,R0,Sa,Fa,S0,s1,s2,s3,Vr,Sd,ba,b0) # strength of density dependence 
    
    ### initialize states variables ### 
    Amax = length(Sa)
    abundanceN = Rstar*La(Sa)
    abundanceH = zeros(Amax)
    
    ### solve for equilibrium genetic state ###
    V = Vstar(Vr,s1+s2+s3)
    grid = zmin:dz:zmax
    trait = pdf.(Distributions.Normal(0,V),grid)
    trait = trait./sum(trait)
    traitN = transpose(repeat(transpose(trait),Amax))
    traitH = transpose(repeat(transpose(trait),Amax))
    
    ### auxiliaries to compute genetic state ###
    # selection gradient 
    gr1 = exp.(-0.5*s1*grid.^2)
    gr2 = exp.(-0.5*s2*grid.^2)
    gr3 = exp.(-0.5*s3*grid.^2)
    # convolution kernel within famil variance 
    d = Distributions.Normal(0, sqrt(Vr))
    grid_Vr = collect((-1*kernel*Vr):dz:(kernel*Vr))
    Vr_ = pdf.(d,grid_Vr)
    m = length(Vr_)
    
    return population(Amax,
                        abundanceN,
                        abundanceH,
                        traitN,
                        traitH,
                        grid,
                        Sa,Fa,rFaplastic,S0,ba,b0,Sd,b,Rstar,
                        Vr,s1,s2,s3,
                        gr1,gr2,gr3,m,Vr_)
end 


function rest!(population;zmin=-5.0,zmax=15.0,dz=0.05,kernel=10.0)
    
    # unpack parameters
    Sa = population.Sa
    s1 = population.s1
    s2 = population.s2
    s3 = population.s3
    Vr = population.Vr
    Sd = (X,b) -> population.Sd(X,b)
    Rstar = population.Rstar

    ### initialize states variables ### 
    Amax = length(Sa)
    abundanceN = Rstar*La(Sa)
    abundanceH = zeros(Amax)
    
    ### solve for equilibrium genetic state ###
    V = Vstar(Vr,s1+s2+s3)
    grid = zmin:dz:zmax
    trait = pdf.(Distributions.Normal(0,V),grid)
    trait = trait./sum(trait)
    traitN = transpose(repeat(transpose(trait),Amax))
    traitH = transpose(repeat(transpose(trait),Amax))
            
    # update popualtion 
    population.abundanceN=abundanceN
    population.abundanceH=abundanceH
    population.traitN=traitN
    population.traitH=traitH
    population
    
end 


#########################################
###  Functions for state transitions  ###
#########################################

function reproduction(population)
    
    # fecundity of age classes 
    f_totalN = sum(population.abundanceN .* population.Fa)
    f_totalH = sum(population.abundanceH .* population.Fa .*population.rFaplastic)
    
    # calcualte trait distribution 
    dsnN = population.traitN * (population.abundanceN .*  population.Fa)# ./ f_totalN
    dsnH = population.traitH * (population.abundanceH .*  population.Fa .*population.rFaplastic) 
    dsn = (dsnN .+ dsnH)./ (f_totalN+f_totalH)
    
    # apply fecundity selection to distribution and total 
    dsn1 = dsn.*population.gradient3
    ftotal = sum(dsn1)/sum(dsn)*(f_totalN + f_totalH)
    dsn1 = dsn1 ./ sum(dsn1)
    
    # convolution - random mating 
    N = length(population.grid)-1
    dsn1 = DSP.conv(dsn1, dsn1)[1:2:(2*N+1)]
    dsn = dsn1./sum(dsn1)
    

    #Plots.plot!(population.grid,dsn )

    # convolution inperfect inheritance 
    m = convert(Int64,floor(population.m/2+1))
    dsn = DSP.conv(dsn, population.Vr_)
     
    # normalize distribution 
    dsn= dsn[m:(N+m)]
    dsn = dsn ./ sum(dsn)

    return dsn, ftotal
end 




function recruits(population)
    # reproduction
    dsn, E = reproduction(population)
    dsn = dsn./sum(dsn)
    # density independent survival 
    R = population.S0*E
    # viability selection 1
    dsn1 = dsn.*population.gradient1
    R = sum(dsn1) * R
    dsn1 = dsn1./sum(dsn1)
    # density dependent survival
    X = population.b0 .* R + sum(population.ba .* (population.abundanceN .+ population.abundanceH))
    R = R * population.Sd(X,population.b)
    # viability selection 1
    dsn2 = dsn1.*population.gradient2
    R = sum(dsn2) * R
    dsn = dsn2./sum(dsn2)
    return dsn, R
end 



function ageing!(population, dsn, R, dsnH, RH)
    
    population.abundanceN = population.abundanceN .* population.Sa
    population.abundanceH = population.abundanceH .* population.Sa

    new_N = zeros(population.Amax)
    new_N[1] = R 
    new_N[2:end] = population.abundanceN[1:end-1]
    
    
    new_H = zeros(population.Amax)
    new_H[1] = RH
    new_H[2:end] = population.abundanceH[1:end-1]
    
    N = length(dsn)
    new_dsnN = zeros(N,population.Amax)
    new_dsnN[:,2:end] = population.traitN[:,1:end-1]
    new_dsnN[:,1] = dsn
    
    new_dsnH = zeros(N,population.Amax)
    new_dsnH[:,2:end] = population.traitH[:,1:end-1]
    new_dsnH[:,1] = dsnH
    
    population.abundanceN = new_N
    population.traitN = new_dsnN
    
    population.abundanceH = new_H
    population.traitH = new_dsnH
end

"""
    time_step(population, dsnH, RH)

Updates popualtion model one time step

population - populaiton model object
dsnH - trait distribution hatchery
RH - hatchery recruitment 
epsilon - random effect on recriotment 
"""
function time_step!(population, dsnH, RH)
    dsnH = dsnH./sum(dsnH)
    dsn, R = recruits(population)
    ageing!(population, dsn, R, dsnH, RH)
end 

function time_step!(population, dsnH, RH,epsilon)
    dsnH = dsnH./sum(dsnH)
    dsn, R = recruits(population)
    R*=exp(epsilon)
    ageing!(population, dsn, R, dsnH, RH)
end 

"""
with constant natual recruitment
"""
function time_step_fixed_R!(population, Rt, dsnH, RH)
    dsnH = dsnH./sum(dsnH)
    dsn, R = recruits(population)
    ageing!(population, dsn, Rt, dsnH, RH)
end 



################################################################
################################################################
#######                                                  #######
#######                                                  #######
#######      Normal Aproximation and linearization       ####### 
#######                                                  #######
#######                                                  #######
################################################################
################################################################

"""
    reproduction(x,pop)

Takes a vector describing the abundaance, mean and trait variance of each cohort in the population
and returns the trait distribution of the next generation, before viability selection. 

x - vector describing the state of the populaiton
x[1:Amax] - abundace of natrual popualtion as a function of increasing age
x[(Amax+1)+(2*Amax)] - abundace of hatchery origin fish
x[(2*Amax+1)+(3*Amax)] - mean trait natraul origin 
x[(3*Amax+1)+(4*Amax)] - mean trait hatchery origin 
x[(4*Amax+1)+(5*Amax)] - trait variance natural origin
x[(5*Amax+1)+(6*Amax)] - trait variance hatchery origin

pop - population model object with genetic and demographic parameters
"""
function reproduction(x,pop, rho)
    # Unpack parameters
    Amax=pop.Amax;Sa=pop.Sa;Fa=pop.Fa# demographic rates 
    S0=pop.S0;ba=pop.ba;Sd=pop.Sd;b=pop.b # denisty dependence
    Vr=pop.Vr;s1=pop.s1;s2=pop.s2;s3=pop.s3 #
    
    # combine Fa into longer vector for hatchery and natrual popualtion 
    Fa = vcat(Fa,Fa)
    
    # Unpack states 
    abundanceN = x[1:Amax]
    abundanceH = x[(Amax+1):(2*Amax)]
    N = vcat(abundanceN,abundanceH)
    zbarN = x[(2*Amax+1):(3*Amax)]
    zbarH = x[(3*Amax+1):(4*Amax)]
    zbar = vcat(zbarN,zbarH)
    VN = x[(4*Amax+1):(5*Amax)]
    VH = x[(5*Amax+1):end]
    V=vcat(VN,VH)
    
    #update distributions
    dat=broadcast(i -> dsn_update(1.0,zbar[i],V[i],s3),1:(2*Amax))
    Sf=broadcast(i->dat[i][1],1:(2*Amax))
    zbarspawning=broadcast(i->dat[i][2],1:(2*Amax))
    Vspawning=broadcast(i->dat[i][3],1:(2*Amax))
    
    # compute weights
    p = Fa.*N.*Sf
    E = sum(p)
    p = p./E

    # mean trait distribution
    zbar0=sum(p.*zbarspawning) 
    # variance of triat distirubtion
    Vbar= sum(p.*Vspawning)
    Ez2 = sum(p .* zbar.^2)
    Vbreading = Vbar+Ez2-zbar0^2
    V0 = 0.5*(1.0+rho)*Vbreading  + Vr
    
    return E, zbar0, V0
end 



function reproduction_test(x,pop)
    # Unpack parameters
    Amax=pop.Amax;Sa=pop.Sa;Fa=pop.Fa# demographic rates 
    S0=pop.S0;ba=pop.ba;Sd=pop.Sd;b=pop.b # denisty dependence
    Vr=pop.Vr;s1=pop.s1;s2=pop.s2;s3=pop.s3 #
    
    # combine Fa into longer vector for hatchery and natrual popualtion 
    Fa = vcat(Fa,Fa)
    
    # Unpack states 
    abundanceN = x[1:Amax]
    abundanceH = x[(Amax+1):(2*Amax)]
    N = vcat(abundanceN,abundanceH)
    zbarN = x[(2*Amax+1):(3*Amax)]
    zbarH = x[(3*Amax+1):(4*Amax)]
    zbar = vcat(zbarN,zbarH)
    VN = x[(4*Amax+1):(5*Amax)]
    VH = x[(5*Amax+1):end]
    V=vcat(VN,VH)
    
    #update distributions
    dat=broadcast(i -> dsn_update(1.0,zbar[i],V[i],s3),1:(2*Amax))
    Sf=broadcast(i->dat[i][1],1:(2*Amax))
    zbarspawning=broadcast(i->dat[i][2],1:(2*Amax))
    Vspawning=broadcast(i->dat[i][3],1:(2*Amax))
    
    #return Sf,zbarspawning,Vspawning
    # compute weights
    p = Fa.*N.*Sf
    E = sum(p)
    p = p./E

    # mean trait distribution
    zbarN=sum(p[1:Amax].*zbarspawning[1:Amax]) ./sum(p[1:Amax])
    zbarH=sum(p[(Amax+1):(2*Amax)].*zbarspawning[(Amax+1):(2*Amax)]) ./ sum(p[(Amax+1):(2*Amax)])

    # variance of triat distirubtion
    VbarN= sum(p[1:Amax].*Vspawning[1:Amax])./sum(p[1:Amax])
    Ez2N = sum(p[1:Amax] .* zbar[1:Amax].^2)./sum(p[1:Amax])
    VbreadingN = VbarN+Ez2N-zbarN^2
    
    VbarH= sum(p[(Amax+1):(2*Amax)].*Vspawning[(Amax+1):(2*Amax)])./ sum(p[(Amax+1):(2*Amax)])
    Ez2H = sum(p[(Amax+1):(2*Amax)] .* zbar[(Amax+1):(2*Amax)].^2)./ sum(p[(Amax+1):(2*Amax)])
    VbreadingH = VbarH+Ez2H-zbarH^2
    
    pN = sum(p[1:Amax])
    pH = sum(p[(Amax+1):(2*Amax)])
    
    return pN,pH,zbarN,zbarH,VbreadingN,VbreadingH,p
end 


"""
    reproduction(x,pop)

Takes a vector describing the abundaance, mean and trait variance of each cohort in the population
and returns the abundance and mean and variance of the trait distribution of the next generation 
after viability selecton and density dependent mortality. 

x[1:Amax] - abundace of natrual popualtion as a function of increasing age
x[(Amax+1)+(2*Amax)] - abundace of hatchery origin fish
x[(2*Amax+1)+(3*Amax)] - mean trait natraul origin 
x[(3*Amax+1)+(4*Amax)] - mean trait hatchery origin 
x[(4*Amax+1)+(5*Amax)] - trait variance natural origin
x[(5*Amax+1)+(6*Amax)] - trait variance hatchery origin

pop - population model object with genetic and demographic parameters
"""
function recrtuitment(x,population,rho)
    Amax = population.Amax
    E, zbar0, V0 = reproduction(x,population,rho)
    #println(V0)
    # Density independent survival 
    R = population.S0*E
    # Viability selection
    R,zbar0,V0=dsn_update(R,zbar0,V0,population.s1)
    #println(V0)
    # density dependent selection 
    abundanceN = x[1:Amax]
    abundanceH = x[(Amax+1):(2*Amax)]
    X = R + sum(population.ba .* (abundanceN .+ abundanceH))
    R = R * population.Sd(X,population.b)
    # viability selection 2
    R,zbar0,V0=dsn_update(R,zbar0,V0,population.s2)
    #println(V0)
    return R,zbar0,V0
end 

"""
    reproduction(x,pop)

Returns an updated state vector x aproximating the triat distribution with 
mean zbar and variance V. 

x[1:Amax] - abundace of natrual popualtion as a function of increasing age
x[(Amax+1)+(2*Amax)] - abundace of hatchery origin fish
x[(2*Amax+1)+(3*Amax)] - mean trait natraul origin 
x[(3*Amax+1)+(4*Amax)] - mean trait hatchery origin 
x[(4*Amax+1)+(5*Amax)] - trait variance natural origin
x[(5*Amax+1)+(6*Amax)] - trait variance hatchery origin

pop - population model object with genetic and demographic parameters
"""
function time_step(x,population,RH,zbarH0,VH0,rho)
    Amax = population.Amax
    R,zbar0,V0=recrtuitment(x,population,rho)
    
    abundanceN = x[1:Amax]
    abundanceH = x[(Amax+1):(2*Amax)]
    zbarN = x[(2*Amax+1):(3*Amax)]
    zbarH = x[(3*Amax+1):(4*Amax)]
    VN = x[(4*Amax+1):(5*Amax)]
    VH = x[(5*Amax+1):end]
    
    # update abundance 
    abundanceN = abundanceN .* population.Sa
    abundanceN[2:end] = abundanceN[1:(end-1)]
    abundanceN[1] = R
    
    abundanceH = abundanceH .* population.Sa
    abundanceH[2:end] = abundanceH[1:(end-1)]
    abundanceH[1] = RH
    
    # update traits
    zbarN[2:end] = zbarN[1:(end-1)]
    zbarN[1] = zbar0
    
    zbarH[2:end] = zbarH[1:(end-1)]
    zbarH[1] = zbarH0
    
    VN[2:end] = VN[1:(end-1)]
    VN[1] = V0
    
    VH[2:end] = VH[1:(end-1)]
    VH[1] = VH0
    
    return vcat(abundanceN,abundanceH,zbarN,zbarH,VN,VH)
end



### functions for assorataive mating 

"""
exp(2/c (x-y)^2)
"""
function singular_kernel(omega_i,omega_j,g_i,g_j,V_i,V_j,c)
    rho = 1/(c*sqrt((1/V_i + 1/c)*(1/V_j + 1/c)))
    sigma_i = ((1-rho^2)*(1/V_i + 1/c))^(-1/2)
    sigma_j = ((1-rho^2)*(1/V_j + 1/c))^(-1/2)
    mu_i = g_i*(sigma_i^2)/V_i+g_j*(sigma_i*sigma_j*rho)/V_j
    mu_j = g_j*(sigma_j^2)/V_j+g_i*(sigma_i*sigma_j*rho)/V_i
    C = (g_i^2)/V_i+(g_j^2)/V_j-
            1/(1-rho^2)*((mu_i^2)/(sigma_i^2)+(mu_j^2)/(sigma_j^2)-(2*rho*mu_i*mu_j)/(sigma_i*sigma_j)) 
    omega_ij = omega_i*omega_j*(sigma_i*sigma_j*sqrt(1-rho^2)*exp(-0.5*C))/sqrt(V_i*V_j)
    return mu_i,mu_j,sigma_i,sigma_j,rho,omega_ij
end


function sum_singular_kernel(omega_i,omega_j,g_i,g_j,V_i,V_j,c)
    mu_i,mu_j,sigma_i,sigma_j,rho,omega_ij = singular_kernel(omega_i,omega_j,g_i,g_j,V_i,V_j,c)
    mu_ij = 0.5*mu_i+0.5*mu_j
    V_ij = 0.25*sigma_i^2+0.25*sigma_j^2+0.5*sigma_i*sigma_j*rho
    return mu_ij, V_ij, omega_ij
end

function assortative_mating(x,pop,c)
    # Unpack parameters
    Amax=pop.Amax;Sa=pop.Sa;Fa=pop.Fa# demographic rates 
    S0=pop.S0;ba=pop.ba;Sd=pop.Sd;b=pop.b # denisty dependence
    Vr=pop.Vr;s1=pop.s1;s2=pop.s2;s3=pop.s3 #
    
    # combine Fa into longer vector for hatchery and natrual popualtion 
    Fa = vcat(Fa,Fa)
    
    # Unpack states 
    abundanceN = x[1:Amax]
    abundanceH = x[(Amax+1):(2*Amax)]
    N = vcat(abundanceN,abundanceH)
    zbarN = x[(2*Amax+1):(3*Amax)]
    zbarH = x[(3*Amax+1):(4*Amax)]
    zbar = vcat(zbarN,zbarH)
    VN = x[(4*Amax+1):(5*Amax)]
    VH = x[(5*Amax+1):end]
    V=vcat(VN,VH)
    
    # update distributions
    dat=broadcast(i -> dsn_update(1.0,zbar[i],V[i],s3),1:(2*Amax))
    Sf=broadcast(i->dat[i][1],1:(2*Amax))
    zbarspawning=broadcast(i->dat[i][2],1:(2*Amax))
    Vspawning=broadcast(i->dat[i][3],1:(2*Amax))
    
    # compute weights
    p = Fa.*N.*Sf
    E = sum(p)
    p = p./E
    
    # compute assortative mating stuff
    K = length(N)
    means = zeros(K^2)
    variance = zeros(K^2)
    weights = zeros(K^2)
    k = 0
    for i in 1:K
        for j in 1:K
            k+=1
            means[k],variance[k],weights[k] = sum_singular_kernel(p[i],p[j],
                                                zbarspawning[i],zbarspawning[j],
                                                Vspawning[i],Vspawning[j],c)
        end
    end 
    
    #insure weights sum to one 
    weights = weights ./sum(weights)
    
    # compute mean and variance of trait distribution
    # mean trait distribution
    zbar0=sum(weights.*means) 
    # variance of triat distirubtion
    Vbar= sum(weights.*variance)
    Ez2 = sum(weights .* means.^2)
    Vbreading = Vbar+Ez2-zbar0^2
    V0 = Vbreading  + Vr
    
    return E, zbar0, V0
end 

function recrtuitment_assort(x,population,c)
    Amax = population.Amax
    E, zbar0, V0 = assortative_mating(x,population,c)
    #println(V0)
    # Density independent survival 
    R = population.S0*E
    # Viability selection
    R,zbar0,V0=dsn_update(R,zbar0,V0,population.s1)
    #println(V0)
    # density dependent selection 
    abundanceN = x[1:Amax]
    abundanceH = x[(Amax+1):(2*Amax)]
    X = R + sum(population.ba .* (abundanceN .+ abundanceH))
    R = R * population.Sd(X,population.b)
    # viability selection 2
    R,zbar0,V0=dsn_update(R,zbar0,V0,population.s2)
    #println(V0)
    return R,zbar0,V0
end 

function time_step_assort(x,population,RH,zbarH0,VH0,c)
    Amax = population.Amax
    R,zbar0,V0=recrtuitment_assort(x,population,c)
    
    abundanceN = x[1:Amax]
    abundanceH = x[(Amax+1):(2*Amax)]
    zbarN = x[(2*Amax+1):(3*Amax)]
    zbarH = x[(3*Amax+1):(4*Amax)]
    VN = x[(4*Amax+1):(5*Amax)]
    VH = x[(5*Amax+1):end]
    
    # update abundance 
    abundanceN = abundanceN .* population.Sa
    abundanceN[2:end] = abundanceN[1:(end-1)]
    abundanceN[1] = R
    
    abundanceH = abundanceH .* population.Sa
    abundanceH[2:end] = abundanceH[1:(end-1)]
    abundanceH[1] = RH
    
    # update traits
    zbarN[2:end] = zbarN[1:(end-1)]
    zbarN[1] = zbar0
    
    zbarH[2:end] = zbarH[1:(end-1)]
    zbarH[1] = zbarH0
    
    VN[2:end] = VN[1:(end-1)]
    VN[1] = V0
    
    VH[2:end] = VH[1:(end-1)]
    VH[1] = VH0
    
    return vcat(abundanceN,abundanceH,zbarN,zbarH,VN,VH)
end


"""
    equilibrium(population,RH,zbarH0,VH0;tol = 10^-8)

Solve for the equilibrium state of the population given geneflow.

population - populaiton model object (stores parameters)
RH - ratch of recruitment rom the hatcheyr population
zbarH0 - mean trait value from hatchery populaiton
VH0 - trait variance hatchery populaiton
"""
function equilibrium(population,RH,zbarH0,VH0,rho;tol = 10^-8)
    # initialize guess
    La_ = La(population.Sa)
    abundanceN = population.Rstar*La_
    abundanceH = RH*La_
    zbarN = zeros(population.Amax)
    zbarH = repeat([zbarH0],population.Amax)
    VN = repeat([2.0*population.Vr],population.Amax)
    VH = repeat([VH0],population.Amax)
    x0 = vcat(abundanceN,abundanceH,zbarN,zbarH,VN,VH)
    
    # run simulation forward to a steady state
    x = time_step(x0,population,RH,zbarH0,VH0,rho)
    diff = sum(abs.(x0 .- x))
    x0 = x
    while diff > tol
        x = time_step(x0,population,RH,zbarH0,VH0,rho)
        diff = sum(abs.(x0 .- x))
        x0 = x
    end 
    return x
end 

"""
    Jabocian(population,RH,zbarH,VH;tol = 10^-8)

Computes the jacobian of the genetic demographic model around the equilibrium point.

population - populaiton model object (stores parameters)
RH - ratch of recruitment rom the hatcheyr population
zbarH0 - mean trait value from hatchery populaiton
VH0 - trait variance hatchery populaiton
"""
function Jabocian(population,RH,zbarH,VH,rho;tol = 10^-8)
    X=equilibrium(population,RH,zbarH,VH,rho;tol=tol)
    return ForwardDiff.jacobian(x->time_step(x,population,RH,zbarH,VH,rho),X)
end 
    
"""
    Jabocian(X,population,RH,zbarH,VH;tol = 10^-8)

Computes the jacobian of the genetic demographic model around the point X.

population - populaiton model object (stores parameters)
RH - ratch of recruitment rom the hatcheyr population
zbarH0 - mean trait value from hatchery populaiton
VH0 - trait variance hatchery populaiton
"""
function Jabocian(X,population,rho;tol = 10^-8)
    return ForwardDiff.jacobian(x->time_step(x,population,0.0,0.0,1.0,rho),X)
end 

"""
    equilibrium(population;tol = 10^-8)

Solve for the equilibrium condition of the model with out gene flow. 

population - populaiton model object (stores parameters)
"""
function equilibriumN(population,rho;tol = 10^-8)
    Amax = population.Amax
    x = equilibrium(population,0,0,1.0,rho;tol = tol)
    abundanceN = x[1:Amax]
    abundanceH = x[(Amax+1):(2*Amax)]
    zbarN = x[(2*Amax+1):(3*Amax)]
    zbarH = x[(3*Amax+1):(4*Amax)]
    VN = x[(4*Amax+1):(5*Amax)]
    VH = x[(5*Amax+1):end]
    return vcat(abundanceN,zbarN,VN)
end 


"""
    equilibrium(population;tol = 10^-8)

Solve for the Jacobian around the equilibrium of the model with out gene flow. 

population - populaiton model object (stores parameters)
"""
function JabocianN(X,population,rho;tol = 10^-8)
    Amax = population.Amax
    J = Jabocian(X,population,rho;tol = tol)
    inds = vcat(1:Amax,(2*Amax+1):(3*Amax),(4*Amax+1):(5*Amax))
    J = J[inds, inds]
    return J
end


"""
    equilibrium(population;tol = 10^-8)

Solve for the Jacobian around the equilibrium of the model with out gene flow. 

population - populaiton model object (stores parameters)
"""
function JabocianN(population,rho;tol = 10^-8)
    Amax = population.Amax
    J = Jabocian(population,0.0,0.0,1.0,rho;tol = tol)
    inds = vcat(1:Amax,(2*Amax+1):(3*Amax),(4*Amax+1):(5*Amax))
    J = J[inds, inds]
    return J
end




#####################################################
#####################################################
######                                         ######
######    Normal aprox demo stochasticity      ######
######                                         ######
#####################################################
#####################################################


# function sample_R(R, epsilon)
#     if R ==0
#         return 0
#     else
#         dR = Distributions.Poisson(R)
#         return quantile(dR,epsilon)
#     end 
# end

# function sample_N(N,p, epsilon)
#     if N ==0
#         return 0
#     else
#         dN = Distributions.Binomial(N,p)
#         return quantile(dN,epsilon)
#     end 
# end


# function sample_zbar(N,zbar,V, epsilon)
    
#     if N ==0
#         return 0
#     elseif N == 1
#         return zbar
#     else
#         dz = Distributions.Normal(0.0,1.0)
#         zeps = quantile(dz,epsilon)
#         return zbar +  sqrt(V/N)* zeps
#     end 
        
# end

    
# function sample_V(N,zbar,V, epsilon)
#     if N ==0
#         return 1.0
#     elseif N == 1
#         return 1.0
#     else
#         #dV = Distributions.Chisq(N)
#         return V*(N-1)/N #V/(N+1) * quantile(dV,epsilon)
#     end 
# end




# function time_step(x,population,RH,zbarH0,VH0, epsilon)
#     Amax = population.Amax
#     R,zbar0,V0=recrtuitment(x,population)
    

#     abundanceN = broadcast(i -> floor(Int,x[i]),1:Amax)
#     abundanceH = broadcast(i -> floor(Int,x[i]),(Amax+1):(2*Amax))
#     zbarN = x[(2*Amax+1):(3*Amax)]
#     zbarH = x[(3*Amax+1):(4*Amax)]
#     VN = x[(4*Amax+1):(5*Amax)]
#     VH = x[(5*Amax+1):end]
    
#     # update abundance 
    
#     # sample R, zbar0 and V0 
#     R = sample_R(R, epsilon[1,1])
    
#     for i in 1:(population.Amax-1)
#         abundanceN[i] = sample_N(abundanceN[i],population.Sa[i], epsilon[i+1,1])
#     end 
    
#     abundanceN[2:end] = abundanceN[1:(end-1)]
#     abundanceN[1] = R

#     RH = sample_R(RH, epsilon[1,2])
    
#     for i in 1:(population.Amax-1)
#         abundanceH[i] = sample_N(abundanceH[i],population.Sa[i], epsilon[i+1,2])
#     end 
    
#     abundanceH[2:end] = abundanceH[1:(end-1)]
#     abundanceH[1] = RH
    
    
#     # update mean
#     zbar0 = sample_zbar(R,zbar0,V0, epsilon[1,3])

# #     for i in 1:(population.Amax-1)
# #         zbarN[i] = sample_zbar(abundanceN[i],zbarN[i],VN[i], epsilon[i+1,3])
# #     end 
    
#     zbarN[2:end] = zbarN[1:(end-1)]
#     zbarN[1] = zbar0
    
#     zbarH0 = sample_zbar(R,zbarH0,VH0, epsilon[1,4])
    
# #     for i in 1:(population.Amax-1)
# #         zbarH[i] = sample_zbar(abundanceH[i],zbarH[i],VH[i], epsilon[i+1,4])
# #     end 
    
#     zbarH[2:end] = zbarH[1:(end-1)]
#     zbarH[1] = zbarH0
    
#     # update variance 
    
# #     VN[2:end] = VN[1:(end-1)]
# #     VN[1] = V0
    
# #     VH[2:end] = VH[1:(end-1)]
# #     VH[1] = VH0
    
# #     V0 = sample_V(R,zbar0,V0, epsilon[1,5])
    
# #     for i in 1:(population.Amax-1)
# #         VN[i] = sample_V(abundanceN[i],zbarN[i],VN[i], epsilon[i+1,5])
# #     end 
    
#     VN[2:end] = VN[1:(end-1)]
#     VN[1] = V0
    
    
# #     VH0 = sample_V(RH,zbarH0,VH0, epsilon[1,6])
    
# #     for i in 1:(population.Amax-1)
# #         VH[i] = sample_V(abundanceH[i],zbarH[i],VH[i], epsilon[i+1,6])
# #     end 
    
#     VH[2:end] = VH[1:(end-1)]
#     VH[1] = VH0

#     return vcat(abundanceN,abundanceH,zbarN,zbarH,VN,VH)
# end







################################################################
################################################################
#######                                                  #######
#######                                                  #######
#######      Functions for the purposes of analysis      ####### 
#######                                                  #######
#######                                                  #######
################################################################
################################################################










end # module 


# mutable struct population_with_hatchery
#     Amax::Int64
#     ### State variables ###
#     abundanceN::AbstractVector{Float64}
#     abundanceH::AbstractVector{Float64}
#     traitN::AbstractMatrix{Float64} # columns are ages 
#     traitH::AbstractMatrix{Float64}
#     grid::AbstractVector{Float64}
    
#     ### Age dependent demographic rates ###
#     Sa::AbstractVector{Float64}
#     Fa::AbstractVector{Float64}
    
#     ### Density dependence ###
#     S0::Float64
#     ba::AbstractVector{Float64}
#     b0::Float64
#     Sd::Function
#     b::Float64
#     Rstar::Float64
    
    
#     ### Genetic parameters ###
#     Vr::Float64 
#     s1::Float64 
#     s2::Float64
#     s3::Float64
    
#     ### hatchery paramters 
#     R0_hatchery::Float64
#     S0_hatchery::Float64
#     s_hatchery::Float64
#     gopt_hatchery::Float64
    
#     ### paramters of implementation of genetic model ###
#     gradient1::AbstractVector{Float64}
#     gradient2::AbstractVector{Float64}
#     gradient3::AbstractVector{Float64}
#     gradientH::AbstractVector{Float64}
#     m::Int64 
#     Vr_::AbstractVector{Float64} 
# end



# function init(Sa,Fa,Wa,R0,Rstar,Sd,r,Vr,s1,s2,s3,
#                 R0_hatchery,s_hatchery,gopt_hatchery;
#                 zmin=-5.0,zmax=15.0,dz=0.05,kernel=10.0)
    
#     ### solve for juvinile survival rates ###
#     ba = CalcWeights(Wa,r) # weights for intra cohort competition 
#     S0 = S0_(R0,Sa,Fa,s1+s2+s3,Vr) # density independent survival
#     S0_hatchery = S0_(R0_hatchery,Sa,Fa,s_hatchery,Vr) # density independent survival
#     b = b_(Rstar,R0,Sa,Fa,S0,s1,s2,s3,Vr,Sd,ba) # strength of density dependence 
    
#     ### initialize states variables ### 
#     Amax = length(Sa)
#     abundanceN = Rstar*La(Sa)
#     abundanceH = zeros(Amax)
    
#     ### solve for equilibrium genetic state ###
#     V = Vstar(Vr,s1+s2+s3)
#     grid = zmin:dz:zmax
#     trait = pdf.(Distributions.Normal(0,V),grid)
#     trait = trait./sum(trait)
#     traitN = transpose(repeat(transpose(trait),Amax))
#     traitH = transpose(repeat(transpose(trait),Amax))
    
#     ### auxiliaries to compute genetic state ###
#     # selection gradient 
#     gr1 = exp.(-0.5*s1*grid.^2)
#     gr2 = exp.(-0.5*s2*grid.^2)
#     gr3 = exp.(-0.5*s3*grid.^2)
#     grH = exp.(-0.5*(s_hatchery*grid .- gopt_hatchery).^2)
#     # convolution kernel within famil variance 
#     d = Distributions.Normal(0, sqrt(Vr))
#     grid_Vr = collect((-1*kernel*Vr):dz:(kernel*Vr))
#     Vr_ = pdf.(d,grid_Vr)
#     m = length(Vr_)
    
#     return population_with_hatchery(Amax,
#                         abundanceN,
#                         abundanceH,
#                         traitN,
#                         traitH,
#                         grid,
#                         Sa,Fa,S0,ba,b0,Sd,b,Rstar,
#                         Vr,s1,s2,s3,
#                         R0_hatchery,S0_hatchery,s_hatchery,gopt_hatchery,
#                         gr1,gr2,gr3,grH,m,Vr_)
# end 




# """
# population - population_with_hatchery object
# B - target number of brood stock
# pmax - maimum proporiton of natraul origin stock  used for breood stock in a single year
# """
# function reproduction_with_hatchery(population, B, pmax)
    
#     ### Brood stock and natrual origin spawners  
#     if (sum(population.abundanceN) * pmax) > B
#         NOB = B*population.abundanceN./sum(population.abundanceN)
#         NOS = population.abundanceN .- NOB
#         HOB = zeros(population.Amax)
#         HOS = population.abundanceH
#     else
#         NOB = pmax*population.abundanceN./sum(population.abundanceN)
#         NOS = population.abundanceN .- NOB
#         HOB_target = B-sum(NOB)
#         if sum(population.abundanceH) > HOB_target
#             HOB = HOB_target*population.abundanceH./sum(population.abundanceH)
#             HOS = population.abundanceH .- HOB
#         else
#             HOB = population.abundanceH
#             HOS = population.abundanceH .- HOB
#         end 
#     end 
    
#     # captive production
#     # fecundity of age classes 
#     f_totalNOB = sum(NOB .* population.Fa)
#     f_totalHOB = sum(HOB .* population.Fa)
    
#     # calcualte trait distribution 
#     dsnNOB = population.traitN * (NOB.*  population.Fa)# ./ f_totalN
#     dsnHOB = population.traitH * (HOB .*  population.Fa) #./ f_totalH
#     dsn = (dsnNOB.+ dsnHOB)./ (f_totalNOB+f_totalHOB)
    
#     # apply fecundity selection to distribution and total 
#     dsn1 = dsn.*population.gradientH
#     ftotal_B = sum(dsn1)/sum(dsn)*(f_totalNOB+ f_totalHOB)
#     dsn1 = dsn1 ./ sum(dsn1)
    
#     # convolution - random mating 
#     N = length(population.grid)-1
#     dsn1 = DSP.conv(dsn1, dsn1)[1:2:(2*N+1)]
#     dsn_B = dsn1./sum(dsn1)
    
#     # convolution inperfect inheritance 
#     m = convert(Int64,floor(population.m/2+1))
#     dsn_B = DSP.conv(dsn_B, population.Vr_)
     
#     # normalize distribution 
#     dsn_B= dsn_B[m:(N+m)]
#     dsn_B = dsn_B ./ sum(dsn_B)
    
    
    
#     ### natural reproduction 
    
#     # fecundity of age classes 
#     f_totalNOS = sum(NOS .* population.Fa)
#     f_totalHOS = sum(HOS .* population.Fa)
    
#     # calcualte trait distribution 
#     dsnN = population.traitN * (NOS  .*  population.Fa)# ./ f_totalN
#     dsnH = population.traitH * (HOS .*  population.Fa) #./ f_totalH
#     dsn = (dsnN .+ dsnH)./ (f_totalNOS+f_totalHOS)
    
#     # apply fecundity selection to distribution and total 
#     dsn1 = dsn.*population.gradient3
#     ftotal_S = sum(dsn1)/sum(dsn)*(f_totalNOS + f_totalHOS)
#     dsn1 = dsn1 ./ sum(dsn1)
    
#     # convolution - random mating 
#     N = length(population.grid)-1
#     dsn1 = DSP.conv(dsn1, dsn1)[1:2:(2*N+1)]
#     dsn_S = dsn1./sum(dsn1)
    

#     #Plots.plot!(population.grid,dsn )

#     # convolution inperfect inheritance 
#     m = convert(Int64,floor(population.m/2+1))
#     dsn_S = DSP.conv(dsn_S, population.Vr_)
     
#     # normalize distribution 
#     dsn_S = dsn_S[m:(N+m)]
#     dsn_S = dsn_S ./ sum(dsn_S)

#     return dsn_S,dsn_B,ftotal_S,ftotal_B
# end 



# function recruits_with_hatchery(population, B, pmax)
#     # reproduction
#     dsn_S,dsn_B,ftotal_S,ftotal_B = reproduction_with_hatchery(population,B,pmax)
#     dsn1 = dsn_S./sum(dsn_S)
#     # density independent survival 
#     RN = population.S0*ftotal_S
#     RH = population.S0_hatchery*ftotal_B
    
#     # viability selection 1
#     dsn1 = dsn1.*population.gradient1
#     R = sum(dsn1) * RN
#     dsn1 = dsn1./sum(dsn1)
#     # density dependent survival
#     X = population.b0 .*R + sum(population.ba .* (population.abundanceN .+ population.abundanceH))
#     R = R * population.Sd(X,population.b)
#     # viability selection 1
#     dsn2 = dsn1.*population.gradient2
#     R = sum(dsn2) * R
#     dsn = dsn2./sum(dsn2)
#     return dsn,dsn_B,R,RH
# end 


# function time_step!(population::population_with_hatchery, B, pmax)
#     dsn,dsn_B,RN,RH= recruits_with_hatchery(population, B, pmax)
#     ageing!(population, dsn, RN, dsn_B, RH)
# end 