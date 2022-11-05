from fakify import linfake

lf = linfake(
            turning_point=(0.1,0.99),
            positive_ratio=0.1)

print(lf.get(100))