import numpy as np
import matplotlib.pyplot as plt

# Original production data
production_data = [1863, 1614, 2570, 1685, 2101, 1811, 2457, 2171, 2134, 2502, 2358, 2399, 2048, 2523, 2086, 2391, 2150, 2340, 3129, 2277, 2964, 2997, 2747, 2862, 3405, 2677, 2749, 2755, 2963, 3161, 3623, 2768, 3141, 3439, 3601, 3531, 3477, 3376, 4027, 3175, 3274, 3334, 3964, 3649, 3502, 3688, 3657, 4422, 4197, 4441, 4736, 4521, 4485, 4644, 5036, 4876, 4789, 4544, 4975, 5211, 4880, 4933, 5079, 5339, 5232, 5520, 5714, 5260, 6110, 5334, 5988, 6235, 6365, 6266, 6345, 6118, 6497, 6278, 6638, 6590, 6271, 7246, 6584, 6594, 7092, 7326, 7409, 7976, 7959, 8012, 8195, 8008, 8313, 7791, 8368, 8933, 8756, 8613, 8705, 9098, 8769, 9544, 9050, 9186, 10012, 9685, 9966, 10048, 10244, 10740, 10318, 10393, 10986, 10635, 10731, 11749, 11849, 12123, 12274, 11666, 11960, 12629, 12915, 13051, 13387, 13309, 13732, 13162, 13644, 13808, 14101, 13992, 15191, 15018, 14917, 15046, 15556, 15893, 16388, 16782, 16716, 17033, 16896, 17689]

# Create an array of periods
periods = np.arange(1, len(production_data) + 1)

# Function to calculate the Taylor expansion
def taylor_series_expansion(x, coefficient, base, num_terms):
    series_sum = np.zeros_like(x, dtype=float)
    for n in range(num_terms):
        series_sum += (base**n) * (x**n) / np.math.factorial(n)
    return coefficient * series_sum

# Convert periods and production data to numpy arrays
periods_array = np.array(periods)
production_array = np.array(production_data)

# Perform logarithmic transformation on production data
log_production = np.log(production_array)

# Fit a linear model to the log-transformed data
matrix = np.vstack([periods_array, np.ones_like(periods_array)]).T
slope, intercept = np.linalg.lstsq(matrix, log_production, rcond=None)[0]

# Convert the intercept back to the original scale
a_coefficient = np.exp(intercept)
b_base = slope

# Generate points for smooth curves
smooth_periods = np.linspace(1, len(production_data), 10000)
extrapolated_periods = np.linspace(1, 200, 100)

# Calculate the Taylor series approximations
smooth_approximation = taylor_series_expansion(smooth_periods, a_coefficient, b_base, 10)
extrapolated_approximation = taylor_series_expansion(extrapolated_periods, a_coefficient, b_base, 10)

# Determine when the production will reach a specific target
target_production_level = 25000
target_period = (np.log(target_production_level / a_coefficient)) / b_base
print(f"The production is expected to reach {target_production_level} after approximately {target_period:.2f} periods.")

# Plot the original data and the Taylor series approximation
plt.figure(figsize=(10, 6))
plt.plot(periods_array, production_array, 'o', label='Observed Data')
plt.plot(smooth_periods, smooth_approximation, 'r-', label='Taylor Series Approximation')
plt.xlabel('Period')
plt.ylabel('Production')
plt.title('Production Data and Taylor Series Approximation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Display the exponential fit equation
print(f"The exponential fit equation is: y = {a_coefficient} * e^({b_base}x)")

# Optionally print the approximated values
for i, val in enumerate(smooth_approximation):
    print(f"{i+1}: {val}")
