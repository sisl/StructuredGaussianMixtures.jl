using Test
using LinearAlgebra
using Random
using Distributions
using StructuredGaussianMixtures

@testset "Prediction and Marginal Functions" begin
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Create test data
    n = 20
    r = 5
    μ = randn(n)
    
    # Create a simple positive definite matrix
    F = randn(n, r)
    D = abs.(randn(n)) .+ 1e-3  # Ensure diagonal elements are positive
    
    # Create distributions
    dist_lrd = LRDMvNormal(μ, F, D)
    # Skip MvNormal tests for now to focus on LRDMvNormal functionality
    
    # @testset "MvNormal predict" begin
    #     # Skip MvNormal tests for now
    # end
    
    @testset "LRDMvNormal predict" begin
        # Test basic prediction
        n1 = 10
        n2 = n - n1
        x1 = randn(n1)
        
        cond_dist = predict(dist_lrd, x1, 1:n1, n1+1:n)
        @test length(cond_dist) == n2
        
        # Test with custom indices
        input_idx = [1, 3, 5, 7, 9]
        output_idx = [2, 4, 6, 8, 10]
        x_input = randn(length(input_idx))
        
        cond_dist_custom = predict(dist_lrd, x_input, input_idx, output_idx)
        @test length(cond_dist_custom) == length(output_idx)
        
        # Test that conditional mean is correct
        Σ11 = cov(dist_lrd)[input_idx, input_idx]
        Σ12 = cov(dist_lrd)[input_idx, output_idx]
        μ1 = μ[input_idx]
        μ2 = μ[output_idx]
        expected_mean = μ2 + Σ12' * (Σ11 \ (x_input - μ1))
        @test mean(cond_dist_custom) ≈ expected_mean atol=1e-10
        
        # Test with small output dimension (should return MvNormal)
        small_output = [1, 2, 3]
        cond_dist_small = predict(dist_lrd, x_input, input_idx, small_output)
        @test cond_dist_small isa MvNormal
        @test length(cond_dist_small) == length(small_output)
        
        # Test with large output dimension (should return LRDMvNormal)
        large_output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        cond_dist_large = predict(dist_lrd, x_input, input_idx, large_output)
        @test cond_dist_large isa LRDMvNormal
        @test length(cond_dist_large) == length(large_output)
        
        # Test error handling
        @test_throws ArgumentError predict(dist_lrd, x1, [0, 1, 2], [3, 4, 5])  # out of bounds
        @test_throws ArgumentError predict(dist_lrd, x1, [1, 2, 3], [n+1, n+2, n+3])  # out of bounds
    end
    
    # @testset "MvNormal marginal" begin
    #     # Skip MvNormal tests for now
    # end
    
    @testset "LRDMvNormal marginal" begin
        # Test basic marginal
        indices = 1:10
        marg_dist = marginal(dist_lrd, indices)
        @test length(marg_dist) == length(indices)
        
        # Test with custom indices
        custom_indices = [5, 10, 15, 20]
        marg_dist_custom = marginal(dist_lrd, custom_indices)
        @test length(marg_dist_custom) == length(custom_indices)
        
        # Test that marginal mean and covariance are correct
        @test mean(marg_dist_custom) ≈ μ[custom_indices]
        @test cov(marg_dist_custom) ≈ cov(dist_lrd)[custom_indices, custom_indices]
        
        # Test with small dimension (should return MvNormal)
        small_indices = [1, 2, 3]
        marg_dist_small = marginal(dist_lrd, small_indices)
        @test marg_dist_small isa MvNormal
        @test length(marg_dist_small) == length(small_indices)
        
        # Test with large dimension (should return LRDMvNormal)
        large_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        marg_dist_large = marginal(dist_lrd, large_indices)
        @test marg_dist_large isa LRDMvNormal
        @test length(marg_dist_large) == length(large_indices)
        
        # Test with single index
        single_idx = [5]
        marg_dist_single = marginal(dist_lrd, single_idx)
        @test length(marg_dist_single) == 1
        @test marg_dist_single isa MvNormal
        @test mean(marg_dist_single) ≈ [μ[5]]
        @test cov(marg_dist_single) ≈ [cov(dist_lrd)[5, 5]]
    end
    
    # @testset "Mixture Model predict" begin
    #     # Skip mixture model tests for now
    # end
    
    @testset "Default argument handling" begin
        # Test predict with default arguments
        x = randn(5)
        
        # Test LRDMvNormal with defaults
        cond_dist_lrd_default = predict(dist_lrd, x)
        @test length(cond_dist_lrd_default) == n - 5
    end
    
    @testset "Edge Cases" begin
        # Test with very small dimensions
        tiny_dist = LRDMvNormal(randn(3), randn(3, 1), abs.(randn(3)) .+ 1e-6)
        tiny_x = randn(1)
        tiny_cond = predict(tiny_dist, tiny_x, [1], [2, 3])
        @test length(tiny_cond) == 2
        
        # Test with rank 0 (diagonal only)
        diag_dist = LRDMvNormal(μ, zeros(n, 0), D)
        diag_cond = predict(diag_dist, randn(5), 1:5, 6:10)
        @test length(diag_cond) == 5
    end
    
    # @testset "Numerical Stability" begin
    #     # Skip numerical stability tests for now as they test edge cases
    # end
end
