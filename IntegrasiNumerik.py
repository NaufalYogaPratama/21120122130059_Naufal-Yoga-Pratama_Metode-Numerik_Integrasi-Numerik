#Nama: Naufal Yoga Pratama
#NIM: 21120122130059
#Kelas: C

import numpy as np
import time
import matplotlib.pyplot as plt

def trapezoid_integral(f, a, b, N):
    x = np.linspace(a, b, N+1)
    y = f(x)
    h = (b - a) / N
    integral = (h / 2) * (y[0] + 2 * sum(y[1:N]) + y[N])
    return integral

def f(x):
    return 4 / (1 + x**2)

def calculate_rms_error(estimated_pi, reference_pi):
    return np.sqrt(np.mean((estimated_pi - reference_pi)**2))

# Nilai referensi pi
reference_pi = 3.14159265358979323846

# Variasi nilai N
N_values = [10, 100, 1000, 10000]
results = []

for N in N_values:
    start_time = time.time()
    estimated_pi = trapezoid_integral(f, 0, 1, N)
    end_time = time.time()
    rms_error = calculate_rms_error(estimated_pi, reference_pi)
    execution_time = end_time - start_time
    results.append((N, estimated_pi, rms_error, execution_time))

# Menampilkan hasil
for result in results:
    N, estimated_pi, rms_error, execution_time = result
    print(f"N={N}: Estimated Pi = {estimated_pi}, RMS Error = {rms_error}, Execution Time = {execution_time} seconds")

# Plotting the results
Ns = [result[0] for result in results]
errors = [result[2] for result in results]
times = [result[3] for result in results]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(Ns, errors, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('RMS Error')
plt.title('RMS Error vs N')

plt.subplot(1, 2, 2)
plt.plot(Ns, times, marker='o')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs N')

plt.tight_layout()
plt.show()
