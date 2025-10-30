#!/usr/bin/env python3
"""
Test to verify that optimized algorithm produces identical results to original.
Run this to prove the optimization doesn't change the logic.
"""

import numpy as np
import sys

def sliding_indices(n, win, step):
    """Helper function - unchanged"""
    return range(0, max(n - win + 1, 0), step)

def fisher_z(r):
    """Helper function - unchanged"""
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

# ========== ORIGINAL ALGORITHM ==========
def sliding_congruence_ORIGINAL(tv_arrays, ad_array, window_sec, step_sec, chunk_dur):
    """Original loop-based implementation"""
    W = max(int(round(window_sec / chunk_dur)), 1)
    S = max(int(round(step_sec / chunk_dur)), 1)
    buf = {}
    
    # Precompute ad windows for efficiency (minor optimization, doesn't change logic)
    ad_indices = list(sliding_indices(ad_array.shape[0], W, S))
    ad_windows = []
    for a0 in ad_indices:
        ad_win = ad_array[a0:a0 + W].reshape(1, -1)
        ad_win_norm = np.linalg.norm(ad_win)
        ad_windows.append((a0, ad_win, ad_win_norm))
    
    for tv in tv_arrays:
        n_tv = tv.shape[0]
        for t0 in sliding_indices(n_tv, W, S):
            tv_win = tv[t0:t0 + W].reshape(1, -1)
            tv_win_norm = np.linalg.norm(tv_win)
            
            # Compute correlation for each ad window
            for a0, ad_win, ad_win_norm in ad_windows:
                r = float((tv_win @ ad_win.T) / (tv_win_norm * ad_win_norm + 1e-8))
                z = float(fisher_z(np.array([r]))[0])
                buf.setdefault((t0, a0), []).append(z)
    
    rows = [[t0 * chunk_dur, a0 * chunk_dur, float(np.mean(zs))] 
            for (t0, a0), zs in buf.items()]
    return np.array(rows, float)

# ========== OPTIMIZED ALGORITHM ==========
def sliding_congruence_OPTIMIZED(tv_arrays, ad_array, window_sec, step_sec, chunk_dur):
    """Optimized vectorized implementation"""
    W = max(int(round(window_sec / chunk_dur)), 1)
    S = max(int(round(step_sec / chunk_dur)), 1)
    
    # Precompute ad windows ONCE (flattened and normalized)
    ad_indices = list(sliding_indices(ad_array.shape[0], W, S))
    n_ad_windows = len(ad_indices)
    
    # Create matrix of all ad windows at once (vectorized!)
    ad_windows_matrix = np.zeros((n_ad_windows, W * ad_array.shape[1]))
    for i, a0 in enumerate(ad_indices):
        ad_windows_matrix[i] = ad_array[a0:a0 + W].flatten()
    
    # Precompute norms for all ad windows at once
    ad_norms = np.linalg.norm(ad_windows_matrix, axis=1, keepdims=True)
    ad_windows_normalized = ad_windows_matrix / (ad_norms + 1e-8)
    
    # Store results efficiently
    results = []
    
    for tv in tv_arrays:
        n_tv = tv.shape[0]
        tv_indices = list(sliding_indices(n_tv, W, S))
        n_tv_windows = len(tv_indices)
        
        # Create matrix of all TV windows at once
        tv_windows_matrix = np.zeros((n_tv_windows, W * tv.shape[1]))
        for i, t0 in enumerate(tv_indices):
            tv_windows_matrix[i] = tv[t0:t0 + W].flatten()
        
        # Compute norms for TV windows
        tv_norms = np.linalg.norm(tv_windows_matrix, axis=1, keepdims=True)
        tv_windows_normalized = tv_windows_matrix / (tv_norms + 1e-8)
        
        # VECTORIZED: Compute ALL correlations at once (matrix multiplication)
        # Shape: (n_tv_windows, n_ad_windows)
        correlations = tv_windows_normalized @ ad_windows_normalized.T
        
        # Apply Fisher-z transformation (vectorized)
        correlations_clipped = np.clip(correlations, -0.999999, 0.999999)
        z_scores = 0.5 * np.log((1 + correlations_clipped) / (1 - correlations_clipped))
        
        # Store results for this TV segment
        for i, t0 in enumerate(tv_indices):
            for j, a0 in enumerate(ad_indices):
                results.append([t0 * chunk_dur, a0 * chunk_dur, z_scores[i, j]])
    
    # Average z-scores across multiple TV segments if needed
    if len(tv_arrays) > 1:
        result_dict = {}
        for t_time, a_time, z in results:
            key = (t_time, a_time)
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(z)
        
        final_results = [[t, a, float(np.mean(zs))] for (t, a), zs in result_dict.items()]
        return np.array(final_results, float)
    
    return np.array(results, float)

# ========== TEST SUITE ==========
def test_equivalence():
    """Test that both implementations produce identical results"""
    
    print("="*70)
    print("TESTING OPTIMIZATION EQUIVALENCE")
    print("="*70)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Test Case 1: Single TV segment
    print("\n[Test 1] Single TV segment")
    tv1 = np.random.randn(100, 128)
    ad1 = np.random.randn(50, 128)
    
    result_orig_1 = sliding_congruence_ORIGINAL([tv1], ad1, 10.0, 5.0, 0.96)
    result_opt_1 = sliding_congruence_OPTIMIZED([tv1], ad1, 10.0, 5.0, 0.96)
    
    print(f"  Original shape: {result_orig_1.shape}")
    print(f"  Optimized shape: {result_opt_1.shape}")
    print(f"  Shapes match: {result_orig_1.shape == result_opt_1.shape}")
    
    diff_1 = np.max(np.abs(result_orig_1 - result_opt_1))
    print(f"  Max absolute difference: {diff_1:.2e}")
    print(f"  ✓ Results equal (tol=1e-10): {np.allclose(result_orig_1, result_opt_1, atol=1e-10)}")
    
    # Test Case 2: Multiple TV segments (tests averaging logic)
    print("\n[Test 2] Multiple TV segments")
    tv2_a = np.random.randn(80, 128)
    tv2_b = np.random.randn(75, 128)
    ad2 = np.random.randn(40, 128)
    
    result_orig_2 = sliding_congruence_ORIGINAL([tv2_a, tv2_b], ad2, 10.0, 5.0, 0.96)
    result_opt_2 = sliding_congruence_OPTIMIZED([tv2_a, tv2_b], ad2, 10.0, 5.0, 0.96)
    
    print(f"  Original shape: {result_orig_2.shape}")
    print(f"  Optimized shape: {result_opt_2.shape}")
    print(f"  Shapes match: {result_orig_2.shape == result_opt_2.shape}")
    
    diff_2 = np.max(np.abs(result_orig_2 - result_opt_2))
    print(f"  Max absolute difference: {diff_2:.2e}")
    print(f"  ✓ Results equal (tol=1e-10): {np.allclose(result_orig_2, result_opt_2, atol=1e-10)}")
    
    # Test Case 3: Different parameters
    print("\n[Test 3] Different window parameters")
    tv3 = np.random.randn(120, 128)
    ad3 = np.random.randn(60, 128)
    
    result_orig_3 = sliding_congruence_ORIGINAL([tv3], ad3, 15.0, 10.0, 0.96)
    result_opt_3 = sliding_congruence_OPTIMIZED([tv3], ad3, 15.0, 10.0, 0.96)
    
    print(f"  Window: 15.0s, Step: 10.0s")
    print(f"  Original shape: {result_orig_3.shape}")
    print(f"  Optimized shape: {result_opt_3.shape}")
    
    diff_3 = np.max(np.abs(result_orig_3 - result_opt_3))
    print(f"  Max absolute difference: {diff_3:.2e}")
    print(f"  ✓ Results equal (tol=1e-10): {np.allclose(result_orig_3, result_opt_3, atol=1e-10)}")
    
    # Test Case 4: Real-world size (similar to actual usage)
    print("\n[Test 4] Real-world size (30s video)")
    tv4_a = np.random.randn(31, 128)  # 30s @ 0.96s sampling
    tv4_b = np.random.randn(31, 128)
    ad4 = np.random.randn(21, 128)    # 20s ad
    
    import time
    
    t0 = time.time()
    result_orig_4 = sliding_congruence_ORIGINAL([tv4_a, tv4_b], ad4, 10.0, 5.0, 0.96)
    time_orig = time.time() - t0
    
    t0 = time.time()
    result_opt_4 = sliding_congruence_OPTIMIZED([tv4_a, tv4_b], ad4, 10.0, 5.0, 0.96)
    time_opt = time.time() - t0
    
    diff_4 = np.max(np.abs(result_orig_4 - result_opt_4))
    speedup = time_orig / time_opt
    
    print(f"  Original time: {time_orig*1000:.2f}ms")
    print(f"  Optimized time: {time_opt*1000:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Max absolute difference: {diff_4:.2e}")
    print(f"  ✓ Results equal (tol=1e-10): {np.allclose(result_orig_4, result_opt_4, atol=1e-10)}")
    
    # Final Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_close = all([
        np.allclose(result_orig_1, result_opt_1, atol=1e-10),
        np.allclose(result_orig_2, result_opt_2, atol=1e-10),
        np.allclose(result_orig_3, result_opt_3, atol=1e-10),
        np.allclose(result_orig_4, result_opt_4, atol=1e-10)
    ])
    
    if all_close:
        print("✅ ALL TESTS PASSED")
        print("✅ Optimized version produces IDENTICAL results to original")
        print(f"✅ Speedup achieved: ~{speedup:.1f}x")
        print("\nCONCLUSION: The optimization preserves the original algorithm logic.")
        print("Same input → Same output. Only the computation method changed.")
        return 0
    else:
        print("❌ TESTS FAILED")
        print("❌ Results differ beyond floating-point tolerance")
        return 1

if __name__ == "__main__":
    sys.exit(test_equivalence())

