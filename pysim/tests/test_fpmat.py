import numpy as np
from lib import fixed_point as fp

# Configuration
BIT_LEN = 32
FRAC_LEN = 24  # Q7.24 is good for Kalman

print(f"--- Starting Test Suite (Q{BIT_LEN - FRAC_LEN}.{FRAC_LEN}) ---")

# 1. Prepare Data
# Use small values (normalized) to avoid overflow
A_float = np.array([[0.5, 0.2], [0.1, 0.4]])
B_float = np.array([[0.3, 0.1], [0.2, 0.6]])

# Create FixedPointMat instances
A = fp.FixedPointMat.from_float(A_float, bit_length=BIT_LEN, frac_length=FRAC_LEN)
B = fp.FixedPointMat.from_float(B_float, bit_length=BIT_LEN, frac_length=FRAC_LEN)

# --- Test Add & Sub ---
print("\n1. Testing Add/Sub...")
res_add = (A + B).to_float()
res_sub = (A - B).to_float()

assert np.allclose(res_add, A_float + B_float, atol=1e-6), "Add failed"
assert np.allclose(res_sub, A_float - B_float, atol=1e-6), "Sub failed"
print("Add/Sub: Pass")

# --- Test Matmul ---
print("\n2. Testing Matmul...")
res_mul = (A @ B).to_float()
expected_mul = A_float @ B_float

print(f"Float Mul: \n{expected_mul}")
print(f"Fxp Mul: \n{res_mul}")
assert np.allclose(res_mul, expected_mul, atol=1e-6), "Matmul failed"
print("Matmul: Pass")

# --- Test Indexing (Get/Set Item) ---
print("\n3. Testing GetItem/SetItem...")
# Test Get
val = A[0, 1].to_float()
assert abs(val - 0.2) < 1e-6, "GetItem failed"

# Test Set
A[0, 0] = 0.99
assert abs(A[0, 0].to_float() - 0.99) < 1e-6, "SetItem failed"
print("GetItem/SetItem: Pass")

# --- Test Inverse 2x2 ---
print("\n4. Testing inv_2x2...")
# Reset A to ensure inverse is valid
A = fp.FixedPointMat.from_float(A_float, bit_length=BIT_LEN, frac_length=FRAC_LEN)

inv_fxp = A.inv().to_float()
inv_float = np.linalg.inv(A_float)

print(f"Float Inv: \n{inv_float}")
print(f"Fxp Inv: \n{inv_fxp}")

# Inverse is sensitive to precision, use higher tolerance
assert np.allclose(inv_fxp, inv_float, atol=1e-4), "Inv failed"
print("Inverse: Pass")

print("\n--- All Tests Passed Successfully ---")
