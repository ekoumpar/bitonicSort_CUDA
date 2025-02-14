import matplotlib.pyplot as plt

# Data
versions = ['V0', 'V1', 'V2']
q_values = [20, 21, 22, 23, 24, 25, 26, 27]
times_v0 = [0.013147, 0.019323, 0.035693, 0.066812, 0.133194, 0.267485, 0.551933, 1.261810]
times_v1 = [0.018012, 0.017819, 0.030247, 0.061345, 0.119780, 0.236168, 0.506228, 1.046886]
times_v2 = [0.012206, 0.018887, 0.033829, 0.066934, 0.144393, 0.262563, 0.571653, 1.170857]
times_QSORT = [0.203257, 0.433413, 0.931642, 1.831812, 2.941607, 6.077412, 9.561333, 19.102546]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(q_values, times_v0, marker='o', label='V0')
plt.plot(q_values, times_v1, marker='o', label='V1')
plt.plot(q_values, times_v2, marker='o', label='V2')
plt.plot(q_values, times_QSORT, marker='o', label='QSORT')

# Labels and Title
plt.xlabel('q Values')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time Comparison for Different Versions')
plt.legend(title='Versions')
plt.grid()

# Show plot
plt.show()
