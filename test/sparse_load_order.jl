using Test
if get(ENV, "SGM_CONVEX_FIRST", "false")=="true"
    using Convex
    using StructuredGaussianMixtures
else
    using StructuredGaussianMixtures
    @test Base.get_extension(
        StructuredGaussianMixtures, :StructuredGaussianMixturesConvexExt
    )===nothing
    @test_throws ArgumentError fit(SparsePrecision(), GraphicalLasso(), ones(2, 4))
    using Convex
end
@test Base.get_extension(
    StructuredGaussianMixtures, :StructuredGaussianMixturesConvexExt
)!==nothing
