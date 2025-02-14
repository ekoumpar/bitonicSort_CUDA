import matplotlib.pyplot as plt

# Data
q_values = [20, 21, 22, 23, 24, 25, 26, 27]
times_v0 = [0.013147, 0.019323, 0.035693, 0.066812, 0.133194, 0.267485, 0.551933, 1.261810]
times_v1 = [0.018012, 0.017819, 0.030247, 0.061345, 0.119780, 0.236168, 0.506228, 1.046886]
times_v2 = [0.012206, 0.018887, 0.033829, 0.066934, 0.144393, 0.262563, 0.571653, 1.170857]
times_qsort = [0.203257, 0.433413, 0.931642, 1.831812, 2.941607, 6.077412, 9.561333, 19.102546]

# Compute speedup
speedup_v0 = [tq / tv0 for tq, tv0 in zip(times_qsort, times_v0)]
speedup_v1 = [tq / tv1 for tq, tv1 in zip(times_qsort, times_v1)]
speedup_v2 = [tq / tv2 for tq, tv2 in zip(times_qsort, times_v2)]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(q_values, speedup_v0, marker='o', label='V0 Speedup')
plt.plot(q_values, speedup_v1, marker='o', label='V1 Speedup')
plt.plot(q_values, speedup_v2, marker='o', label='V2 Speedup')

# Labels and Title
plt.xlabel('q Values')
plt.ylabel('Speedup')
plt.title('Speedup of Different Versions Compared to qSort')
plt.legend(title='Versions')
plt.grid()

# Show plot
plt.show()