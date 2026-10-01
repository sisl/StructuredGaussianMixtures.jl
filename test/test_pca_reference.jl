using MultivariateStats:
    PCA, fit as pca_fit, projection, mean as pca_mean, predict as pca_predict, reconstruct
@testset "PCA initialization equivalence" begin
    # Exercise the actual PCAEM initialization, checking subspaces rather than
    # factor signs. Cover dense and iterative eigensolvers on both Gram branches.
    for (d, n, r) in
        ((20, 1000, 5), (20, 1000, 6), (150, 400, 5), (400, 150, 5), (200, 30, 5))
        data=randn(MersenneTwister(81), d, n)
        spec=MixtureSpec(LatentCovariance(r), 1; tied=Tied(:F, :D))
        state=initialize(spec, PCAEM(residual_floor=0), data; rng=MersenneTwister(82))
        pca=pca_fit(PCA, data; maxoutdim=r)
        P=projection(pca)
        @test size(state.F)==(d, r)
        @test state.offset ≈ pca_mean(pca) atol=1e-10
        @test state.F*state.F' ≈ P*P' atol=1e-8
        residual=data .- reconstruct(pca, pca_predict(pca, data))
        @test state.D ≈ vec(mean(abs2.(residual); dims=2)) atol=1e-8
    end
end
