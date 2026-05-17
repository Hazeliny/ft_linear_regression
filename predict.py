import json
import os

def load_thetas():
	"""
	Try to load θ0 and θ1 from thetas.json.
	If the file doesn't exist yet (training not done),
    return 0, 0 as default values.
    
    Example of thetas.json content:
    {
        "theta0": 8500.0,
        "theta1": -0.04
    }
    """
	if not os.path.exists("thetas.json"):
		print("⚠️  No thetas.json found. Using θ0=0, θ1=0 (not trained yet).")
		return 0.0, 0.0

	with open("thetas.json", "r") as f:
		data = json.load(f)

	theta0 = data["theta0"]
	theta1 = data["theta1"]

	print(f"✅ Loaded: θ0 = {theta0:.4f}, θ1 = {theta1:.6f}")
	return theta0, theta1

def estimate_price(mileage, theta0, theta1):
	"""
    The hypothesis function from the subject:
    estimatePrice(mileage) = θ0 + (θ1 × mileage)
    
    Example:
      θ0 = 8500, θ1 = -0.04, mileage = 80000
      price = 8500 + (-0.04 × 80000) = 8500 - 3200 = 5300 €
    """
	return theta0 + (theta1 * mileage)

def get_mileage_from_user():
	"""
    Ask user to input a mileage.
    Handle invalid inputs gracefully (letters, negative numbers, etc.)
    
    Example interaction:
      Enter mileage: abc   → error, ask again
      Enter mileage: -500  → error, ask again
      Enter mileage: 80000 → ✅ return 80000.0
    """
	while True:
		try:
			mileage = float(input("\n🚗 Enter mileage (km): "))
			if mileage < 0:
				print("❌ Mileage cannot be negative. Please try again.")
				continue
			return mileage
		except ValueError:
			print("❌ Invalid input. Please enter a number.")
	
def main():
	print("=" * 40)
	print("   ft_linear_regression - Predict")
	print("=" * 40)

	# Step 1: load thetas
	theta0, theta1 = load_thetas()

	# Step 2: get mileage from user
	mileage = get_mileage_from_user()

	# Step 3: predict price
	price = estimate_price(mileage, theta0, theta1)

	# Step 4: display result
	# Price should not be negative in real life
	if price < 0:
		print(f"\n⚠️  Predicted price: 0 € (model returned {price:.2f}, clamped to 0)")
	else:
		print(f"\n💰 Estimated price for {mileage:.0f} km: {price:.2f} €\n")


if __name__ == "__main__":
	main()