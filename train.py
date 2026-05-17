import csv
import json

# Step 1: read the CSV
def load_data(filename="data.csv"):
	"""
    Read mileage and price columns from CSV.
    
    Returns two lists:
      mileages = [240000, 139800, 150500, ...]
      prices   = [3650, 3800, 4400, ...]
    """
	mileages = []
	prices = []

	with open(filename, "r") as f:
		reader = csv.DictReader(f) # reads header automatically
		for row in reader:
			mileages.append(float(row["km"]))
			prices.append(float(row["price"]))

	print(f"✅ Loaded {len(mileages)} data points from {filename}")
	return mileages, prices

# Step 2: normalize
def normalize(values):
	"""
    Min-max normalization: scale all values to [0, 1]
    
    Formula: x_norm = (x - min) / (max - min)
    
    Example with mileages:
      values = [240000, 139800, 63060]
      min = 63060, max = 240000
      
      240000 → (240000 - 63060) / (240000 - 63060) = 1.0
      139800 → (139800 - 63060) / (240000 - 63060) = 0.434
      63060  → (63060  - 63060) / (240000 - 63060) = 0.0
    """
	min_val = min(values)
	max_val = max(values)
	normalized = [(x - min_val) / (max_val - min_val) for x in values]
	return normalized, min_val, max_val

# Step 3: gradient descent
def estimate_price(mileage, theta0, theta1):
	return theta0 + (theta1 * mileage)

def gradient_descent(mileages_norm, prices_norm, learning_rate=0.1, iterations=1000):
	"""
    The core of the project.
    
    Start with θ0=0, θ1=0 and iteratively improve them.
    
    Concrete example with m=24 data points, iteration 1:
    
      For each point i, compute error:
        error_i = estimatePrice(mileage_norm[i]) - price_norm[i]
      
      tmp_θ0 = lr × (1/m) × sum(all errors)
      tmp_θ1 = lr × (1/m) × sum(error_i × mileage_norm[i])
      
      Update simultaneously:
        θ0 = θ0 - tmp_θ0
        θ1 = θ1 - tmp_θ1
    """
	theta0 = 0.0
	theta1 = 0.0
	m = len(mileages_norm)

	for i in range(iterations):

		#calculate errors for all points
		errors = [
			estimate_price(mileages_norm[j], theta0, theta1) - prices_norm[j]
			for j in range(m)
		]

		# gradients (exactly the formula from the subject)
		tmp_theta0 = learning_rate * (1/m) * sum(errors)
		tmp_theta1 = learning_rate * (1/m) * sum(errors[j] * mileages_norm[j] for j in range(m))

		# simultaneous update
		theta0 = theta0 - tmp_theta0
		theta1 = theta1 - tmp_theta1

		# print progress every 100 iterations
		if i % 100 == 0:
			mse = sum(e**2 for e in errors) / m
			print(f"  Iteration {i:4d} | MSE: {mse:.6f} | "
                  f"θ0: {theta0:.4f} | θ1: {theta1:.4f}")

	return theta0, theta1

# Step 4: denormalize θ back to real scale
def denormalize_thetas(theta0_norm, theta1_norm, km_min, km_max, price_min, price_max):
	"""
    We trained on normalized values, but predict.py uses real km.
    
    The normalized model is:
      price_norm = θ0_norm + θ1_norm × km_norm
    
    Expanding with the normalization formulas:
      (price - price_min)/(price_max - price_min) = 
        θ0_norm + θ1_norm × (km - km_min)/(km_max - km_min)
    
    Solving for price gives us the real θ0 and θ1:
    
      real_θ1 = θ1_norm × (price_max - price_min) / (km_max - km_min)
      real_θ0 = θ0_norm × (price_max - price_min) + price_min
                - real_θ1 × km_min
    """
	price_range = price_max - price_min
	km_range = km_max - km_min

	real_theta1 = theta1_norm * price_range / km_range
	real_theta0 = theta0_norm *price_range + price_min - real_theta1 * km_min

	return real_theta0, real_theta1

# Step 5: save thetas
def save_thetas(theta0, theta1, filename="thetas.json"):
	data = {"theta0": theta0, "theta1": theta1}
	with open(filename, "w") as f:
		json.dump(data, f)
	print(f"\n✅ Saved θ0={theta0:.4f}, θ1={theta1:.6f} → {filename}")

def main():
	print("=" * 40)
	print("   ft_linear_regression - Train")
	print("=" * 40)

	# Step 1: load data
	mileages, prices = load_data("data.csv")

	# Step 2: normalize
	mileages_norm, km_min, km_max = normalize(mileages)
	prices_norm, price_min, price_max = normalize(prices)

	print(f"\n📊 Mileage range: {km_min:.0f} ~ {km_max:.0f} km")
	print(f"📊 Price range:   {price_min:.0f} ~ {price_max:.0f} €")

	# Step 3: gradient descent on normalized data
	print("\n🔄 Training...\n")
	theta0_norm, theta1_norm = gradient_descent(mileages_norm, prices_norm, learning_rate=0.1, iterations=1000)

	# Step 4: denormalize back to real world
	theta0, theta1 = denormalize_thetas(theta0_norm, theta1_norm, km_min, km_max, price_min, price_max)
	print(f"\n🎯 Real-world thetas:")
	print(f"   θ0 = {theta0:.4f}")
	print(f"   θ1 = {theta1:.6f}")

	# Step 5: save
	save_thetas(theta0, theta1)

	print("\n✅ Training complete! Now run predict.py\n")
	


if __name__ == "__main__":
	main()
