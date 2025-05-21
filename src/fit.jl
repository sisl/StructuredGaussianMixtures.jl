struct GMMFit
    n_components::Int
    method::Symbol
    kind::Symbol
    nInit::Int
    nIter::Int
    nFinal::Int
end
function fit(gmmfit::GMMFit, x::Matrix)
   gmm = GMM(gmmfit.n_components, permutedims(x); method=gmmfit.method, kind=gmmfit.kind, nInit=gmmfit.nInit, nIter=gmmfit.nIter, nFinal=gmmfit.nFinal)
   return MixtureModel(gmm)
end

struct MPPCA
    n_components::Int
    rank::Int
    gmm_method::Symbol
    gmm_nInit::Int
    gmm_nIter::Int
    gmm_nFinal::Int
end

function fit(mppca::MPPCA, x::Matrix)
    # run PCA on x
    pca = fit(PCA, x)


    # run GMM on the PCA scores
    gmm = fit(GMM, pca.scores; method=mppca.gmm_method, kind=mppca.gmm_kind, nInit=mppca.gmm_nInit, nIter=mppca.gmm_nIter, nFinal=mppca.gmm_nFinal)

    
    return MixtureModel(gmm)
end

struct EM
    n_components::Int
    rank::Int
    gmm_method::Symbol
    gmm_nInit::Int
    gmm_nIter::Int
    gmm_nFinal::Int
end

function fit(em::EM, x::Matrix)
    error("Not implemented")
end