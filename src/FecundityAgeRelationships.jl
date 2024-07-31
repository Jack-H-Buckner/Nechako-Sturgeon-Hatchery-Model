"""
calculates fecundity at age relationships based on
Von-bertalanffy growth curve defined as functors
"""
module FecundityAgeRelationships

struct isometricVonBertlanffy
    finfty::Float64
    k::Float64
    a0::Float64
end 

function(FAR::isometricVonBertlanffy)(a)
    FAR.finfty*(1-exp(-FAR.k*(a - FAR.a0)))^3
end


struct allometricVonBertlanffy
    finfty::Float64
    k::Float64
    a0::Float64
    beta::Float64
end 

function (FAR::allometricVonBertlanffy)(a)
    return FAR.finfty*(1-exp(-FAR.k*(a - FAR.a0)))^FAR.beta
end 



struct Walters2006
    Linfty::Float64
    Wmat::Float64
    W100
    k::Float64
    a0::Float64
end 


function (FAR::Walters2006)(a)
    L = FAR.Linfty*(1-exp(-FAR.k*a))
    W = L^3 *FAR.W100/100^3 # this should be 100^3
    F = (W - FAR.Wmat)#*(1 - 1/(1+exp((-(a-mu_s)/sigma_s))))
    if F < 0.0
        F*=0.0
    end 
    return F
end 

struct Walters2006Senescence
    Linfty::Float64
    Wmat::Float64
    W100
    k::Float64
    a0::Float64
    mu::Float64
    sigma::Float64
end 


function (FAR::Walters2006Senescence)(a)
    L = FAR.Linfty*(1-exp(-FAR.k*a))
    W = L^3 * FAR.W100/100^3
    F = (W - FAR.Wmat)*(1 - 1/(1+exp((-(a-FAR.mu)/FAR.sigma))))
    if F < 0
        F=0
    end 
    return F
end 




## weight at age relationship

struct Walters2006_WAR
    Linfty::Float64
    W100
    k::Float64
    a0::Float64
end 


function (WAR::Walters2006_WAR)(a)
    L = WAR.Linfty*(1-exp(-WAR.k*a))
    W = WAR.W100/100*L^3
    return W
end 

end #module 