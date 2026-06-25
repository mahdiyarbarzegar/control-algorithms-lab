from lib.fixed_point import FixedPointVar as fpv

val = fpv.from_float(8, 4, "saturate", True, 3.14)

print(f"Value: {val.to_float():.2f}")
