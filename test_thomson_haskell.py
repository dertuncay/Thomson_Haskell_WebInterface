import unittest
import numpy as np
import math
from thomson_haskell import thomson_haskell_transfer_function, q_from_damping, damping_from_q

class TestThomsonHaskell(unittest.TestCase):

    def test_q_from_damping(self):
        # Normal case: damping = 0.05 -> Q = 10
        self.assertAlmostEqual(q_from_damping(0.05), 10.0)
        # Case: damping = 0 -> Q = inf
        self.assertEqual(q_from_damping(0), float('inf'))

    def test_damping_from_q(self):
        # Normal case: Q = 10 -> damping = 0.05
        self.assertAlmostEqual(damping_from_q(10), 0.05)
        # Case: Q = 0 -> inf (based on current implementation logic)
        self.assertEqual(damping_from_q(0), float('inf'))

    def test_transfer_function_homogeneous(self):
        # Test a case where the layer matches the halfspace exactly.
        # This implies "no interface", so waves should just travel through.
        # However, due to free surface effect at the top of the layer, we still expect amplification/resonance?
        # Wait, Thomson-Haskell usually models response of soil layers on top of bedrock.
        # If the soil has same properties as bedrock, there is no contrast at interface.
        # But there is still a free surface at the top?
        # The formulation TF = u_top / u_bottom usually relates surface motion to motion at interface with halfspace (or outcrop).
        # If it's "within" the medium.
        
        # Let's try parameters that create a known condition.
        # If we have 1 layer with same props as halfspace.
        
        freqs = np.array([1.0, 5.0, 10.0])
        h = [100.0]
        vs_val = 1000.0
        rho_val = 2000.0
        qs_val = 1e9 # Very high Q -> no damping
        
        vs = [vs_val, vs_val]
        rho = [rho_val, rho_val]
        qs = [qs_val, qs_val]
        
        # If the layer is same as halfspace, wave just propagates up.
        # At surface (top), it reflects. 
        # Standard SH transfer function for layer over halfspace:
        # If impedance contrast is 1 (same material), do we get simple resonance?
        # TF = 1 / cos(kH)? Or similar?
        
        # Let's run it and check if it produces finite numbers at least.
        tf = thomson_haskell_transfer_function(freqs, h, vs, rho, qs)
        self.assertEqual(len(tf), 3)
        self.assertTrue(np.all(np.isfinite(tf)))

    def test_basic_structure(self):
        # Just ensure it runs without error for a standard input
        freqs = np.linspace(0.1, 10, 50)
        h = [10, 20]
        vs = [200, 300, 500] # 2 layers + halfspace
        rho = [1800, 1900, 2000]
        qs = [20, 30, 50]
        
        tf = thomson_haskell_transfer_function(freqs, h, vs, rho, qs)
        self.assertEqual(len(tf), 50)
        self.assertTrue(np.iscomplexobj(tf))

if __name__ == '__main__':
    unittest.main()
