module DemographicParameters


include("AgeStructuredModels.jl")
include("FecundityAgeRelationships.jl")
include("StockRecruitCurves.jl")

##############################################
##############################################
#######                                #######
#######  sturgeon demographic params   #######
#######                                #######
##############################################
##############################################

##########################
#### fecundity at age ####
##########################

A_max = 150
a = collect(1:A_max)

# Wood et al 2007
Wood_07_W100 = 7
Wood_07_Wmat = 165^3 *Wood_07_W100/100
Wood_07_K = 0.023
Wood_07_Linfty = 412
Wood_07_mu_s1 = 80# 100 # 
Wood_07_sigma_s1 = 5 #0 #5
Wood_07_mu_s2 = 200  
Wood_07_sigma_s2 = 0 #0 #5
Wood_07_m = 0.923


# Smyth 2016
Smyth_16_W100 = 9.3
Smyth_16_Wmat = 153^3 *Smyth_16_W100/(100^3)
Smyth_16_K = 0.034
Smyth_16_Linfty = 300
Smyth_16_mu_s = 200  
Smyth_16_sigma_s = 0 #0 #5
Smyth_16_m = 0.96

##########################
####   weight at age  ####
##########################
Smyth_16_WAR = FecundityAgeRelationships.Walters2006_WAR(Smyth_16_Linfty, Smyth_16_W100, Smyth_16_K, 0)

Wood_07_WAR = FecundityAgeRelationships.Walters2006_WAR(Wood_07_Linfty, Wood_07_W100, Wood_07_K, 0)

# fecundity age function
Wood_07_F1 = FecundityAgeRelationships.Walters2006Senescence(Wood_07_Linfty,  Wood_07_Wmat,Wood_07_W100,
                                                            Wood_07_K, 0, Wood_07_mu_s1, Wood_07_sigma_s1)
Wood_07_F1_vec = Wood_07_F1.(a)

Wood_07_F2 = FecundityAgeRelationships.Walters2006(Wood_07_Linfty,Wood_07_Wmat, Wood_07_W100, Wood_07_K, 0)
Wood_07_F2_vec = Wood_07_F2.(a)

Smyth_16_F = FecundityAgeRelationships.Walters2006(Smyth_16_Linfty,  Smyth_16_Wmat, Smyth_16_W100,Smyth_16_K, 0)
Smyth_16_F_vec = Smyth_16_F.(a)

##########################
####  survival rates  ####
##########################

Wood_2007_survival = repeat([Wood_07_m],A_max)
Smyth_2016_survival = repeat([Smyth_16_m],A_max) 
long_lived_survival = vcat(repeat([0.9],80), repeat([0.98],A_max - 80))
    
############################
####  impact functions  ####
############################


# life time egg production 
psurvival = Wood_07_m .^a
psurvival = psurvival./psurvival[1]
Wood_2007_LEP1 = sum(Wood_07_F1_vec .* psurvival)
Wood_2007_impact1 = Wood_07_F1_vec  .* psurvival

psurvival = Wood_07_m .^a
psurvival = psurvival./psurvival[1]
Wood_2007_LEP2 = sum(Wood_07_F2_vec  .* psurvival)
Wood_2007_impact2 = Wood_07_F2_vec  .* psurvival

psurvival = Smyth_16_m .^a
psurvival = psurvival./psurvival[1]
Smyth_2016_LEP = sum(Smyth_16_F_vec  .* psurvival)
Smyth_2016_impact = Smyth_16_F_vec  .* psurvival

psurvival = zeros(A_max)
psurvival[1] = long_lived_survival[1]
for i in 2:A_max
    psurvival[i] = long_lived_survival[i] * psurvival[i-1]
end 
psurvival = psurvival ./ psurvival[1]
Wood_2007_LEP1_ll = sum(Wood_07_F1_vec  .* psurvival)
Wood_2007_impact1_ll = Wood_07_F1_vec  .* psurvival

Wood_2007_LEP2_ll = sum(Wood_07_F2_vec  .* psurvival)
Wood_2007_impact2_ll = Wood_07_F2_vec  .* psurvival

Smyth_2016_LEP_ll = sum(Smyth_16_F_vec  .* psurvival)
Smyth_2016_impact_ll = Smyth_16_F_vec  .* psurvival





##########################
####  stock recruit   ####
##########################



N_star = 1500 # equilibirum popuatlion size 
Smyth_16_R_star = N_star/sum(Smyth_16_m .^a) # equilibrium recruitment rate
Wood_2007_R_star = N_star/sum(Wood_07_m .^a)
LL_R_star = N_star/sum(psurvival)
k = 5.0 #[1.5,3,5,7,9]

Smyth_2016_sr = StockRecruitCurves.init_BevetonHolt(Smyth_16_R_star,k, Smyth_2016_LEP)
Smyth_2016_ll_sr = StockRecruitCurves.init_BevetonHolt(LL_R_star,k, Smyth_2016_LEP_ll)
Wood_2007_1_sr = StockRecruitCurves.init_BevetonHolt(Wood_2007_R_star,k, Wood_2007_LEP1)
Wood_2007_2_sr = StockRecruitCurves.init_BevetonHolt(Wood_2007_R_star,k, Wood_2007_LEP2)
Wood_2007_1_ll_sr = StockRecruitCurves.init_BevetonHolt(LL_R_star,k, Wood_2007_LEP1_ll)
Wood_2007_2_ll_sr = StockRecruitCurves.init_BevetonHolt(LL_R_star,k, Wood_2007_LEP2_ll)



############################
####  generation time   ####
############################

Wood_2007_T1 =  sum(a .* (Wood_07_F1_vec  .* Wood_07_m .^a) ./ Wood_2007_LEP1)
Wood_2007_T2 =  sum(a .* (Wood_07_F2_vec  .* Wood_07_m .^a) ./ Wood_2007_LEP2)
Smyth_2016_T1 =  sum(a .* (Smyth_16_F_vec  .* Smyth_16_m .^a) ./ Smyth_2016_LEP)

Wood_2007_T3 =  sum(a .* (Wood_07_F1_vec   .* psurvival) ./ Wood_2007_LEP1_ll)
Wood_2007_T4 =  sum(a .* (Wood_07_F2_vec   .* psurvival) ./ Wood_2007_LEP2_ll)
Smyth_2016_T2 =  sum(a .* (Smyth_16_F_vec  .* psurvival) ./ Smyth_2016_LEP_ll)


################################
### Age structure param sets ###
################################


mod_Smyth_2016 = AgeStructuredModels.init(150,Smyth_2016_sr,Smyth_2016_survival,Smyth_16_F_vec)

mod_Smyth_2016_ll = AgeStructuredModels.init(150,Smyth_2016_ll_sr,long_lived_survival,Smyth_16_F_vec)

mod_Wood_2007_1 = AgeStructuredModels.init(150,Wood_2007_1_sr,Wood_2007_survival,Wood_07_F1_vec)

mod_Wood_2007_2 = AgeStructuredModels.init(150,Wood_2007_2_sr,Wood_2007_survival,Wood_07_F2_vec)

mod_Wood_2007_ll1 = AgeStructuredModels.init(150,Wood_2007_1_ll_sr,long_lived_survival,Wood_07_F1_vec)

mod_Wood_2007_ll2 = AgeStructuredModels.init(150,Wood_2007_2_ll_sr,long_lived_survival,Wood_07_F2_vec)



################################
### five year time intervals ###
################################


A_max_5years = 30
a_5years = 5*collect(1:A_max_5years)


#### fecundity at age ####

Smyth_16_F= FecundityAgeRelationships.Walters2006(Smyth_16_Linfty, Smyth_16_W100, Smyth_16_Wmat, Smyth_16_K, 0)
Smyth_16_F_vec_5years = Smyth_16_F.(a_5years .- 2.5)


####  survival rates  ####
Smyth_2016_survival_5years = repeat([Smyth_16_m^5],A_max_5years) 

####  impact functions  ####
psurvival_5years = Smyth_16_m .^a_5years
psurvival_5years = psurvival_5years./psurvival_5years[1]
Smyth_2016_LEP_5years = sum(Smyth_16_F_vec_5years  .* psurvival_5years)
Smyth_2016_impact_5years = Smyth_16_F_vec_5years  .* psurvival_5years

#### generation time ####
Smyth_2016_T1_5years = Smyth_2016_T1/5

#### sr curves ####
Smyth_16_R_star_5years = N_star/sum(Smyth_16_m .^(a_5years)) # equilibrium recruitment rate
Smyth_2016_sr_5years = StockRecruitCurves.init_BevetonHolt(Smyth_16_R_star_5years,k, Smyth_2016_LEP_5years)


### Age structure param sets ###
mod_Smyth_2016_5year = AgeStructuredModels.init(30,Smyth_2016_sr_5years, Smyth_2016_survival_5years, Smyth_16_F_vec_5years)





####################################################
####################################################
#######                                      #######
#######  natrual mortaltiy -growth tradoff   #######
#######                                      #######
####################################################
####################################################



function generation(K)
    Amax = 200
    Linfty = 100
    Lm = 0.1(Linfty)

    La = Linfty*(1 .-exp.(-K*(1:Amax)))
    Fa = 10^4*(La.^3 .- Lm^3)./(Linfty.^3 .- Lm^3)  # Assuming max fecundity is 10^4
    Fa[Fa .<0] .= 0.0
    M = exp(0.25 + log(K))
        
    S = exp(-M)
    Sa =  S.^((1:Amax).-1)
    Impact = Sa .* Fa
    Impact=Impact./sum(Impact)
    return sum(Impact .* collect(1:Amax))
        
end 

using Roots
"""
    params(T)

Solves for the fecundity, weight and survival at age for 
a species with aympthotic length Linfty and generation time

I assume constant natrual mortality after recruitment at age one and
von-burtlanfy growth with size at age zero equal to zero  

the relationshio between growth rate and natural mortality is given by

log(M) = 0.25 + log(K)

I solve for a growth rate K to achieve the desiered generation time. 

I also assume an isometric relationship between length and fecundity. 
"""
function params(T)
    Linfty = 100
    K = Roots.find_zero(K -> T-generation(K), [0.001,100.0])
    Amax = 200
    Lm = exp(-1.0 + 0.95*log(Linfty))

    La = Linfty*(1 .-exp.(-K*(1:Amax)))
    Wa = La.^3
    Fa = 10^4*(La.^3 .- Lm^3)./(Linfty.^3 .- Lm^3)  # Assuming max fecundity is 10^4
    Fa[Fa .<0] .= 0.0
    M = exp(0.25 + log(K))
        
    S = exp(-M)
    La = S.^((1:Amax) .-1)
    Amax = argmax((1:Amax)[La.>0.001] )
    Sa = repeat([S],Amax)
    Fa = Fa[1:Amax]
    Wa = Wa[1:Amax]
    return Fa,Sa,Wa
end 


S(K) = exp(-M(K))
        
###############################################
###############################################
#######                                 #######
#######  salmon demographic parameters  #######
#######                                 #######
###############################################
###############################################

# simple model of salmon life cycle with all individuals spawing at age 4
# this is a course aproximation for Chinook salmon and is a bit more accurate
# for coho or chum salmon. Pink salmon have a two year cycle. 
Salmon_Fa = [0.0,0.0,0.0,10000.0]
Salmon_Sa = [1.0,1.0,1.0,0.0]
Salmon_Wa = [0.1,0.1,0.1,0.1] # salmon are unlikely to compete with prerecruits so weights here are st to zero 


end # modul 