import json
import os
import matplotlib.pyplot as plt
import csv
import math


# ─────────────────────────────────────
# 1. loading data and thetas
# ─────────────────────────────────────

def load_data(filename="data.csv"):
    mileages = []
    prices = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mileages.append(float(row["km"]))
            prices.append(float(row["price"]))
    return mileages, prices

def load_thetas(filename="thetas.json"):
    if not os.path.exists(filename):
        print("⚠️  No thetas.json found. Please run train first.")
        return None, None
    with open(filename, "r") as f:
        data = json.load(f)
    return data["theta0"], data["theta1"]

def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


# ─────────────────────────────────────
# 2. precision calculation: R² score
# ─────────────────────────────────────

def calculate_r2(mileages, prices, theta0, theta1):
    """
    R² (coefficient of determination) determination coefficient
    
    measures how much of the variation in the data is explained by the model:
      R² = 1  → perfect prediction, the model explains 100% of the variation
      R² = 0  → the model is no better than simply predicting the mean
      R² < 0  → the model performs worse than predicting the mean
    
    formula:
      R² = 1 - SS_res / SS_tot
      
      SS_res = Σ(real values - predicted values)²   ← model residuals
      SS_tot = Σ(real values - mean)²   ← total variation in the data
    
    specific example (simplified to 3 points):
      real prices:  [3650, 5000, 8290]
      mean price:  (3650+5000+8290)/3 = 5646
      predicted prices:  [3800, 4900, 8100]
      
      SS_res = (3650-3800)² + (5000-4900)² + (8290-8100)²
             = 22500 + 10000 + 36100 = 68600
             
      SS_tot = (3650-5646)² + (5000-5646)² + (8290-5646)²
             = 3984016 + 417316 + 6989796 = 11391128
             
      R² = 1 - 68600/11391128 = 1 - 0.006 = 0.994  ← very good!
    """
    mean_price = sum(prices) / len(prices)

    ss_res = sum(
        (prices[i] - estimate_price(mileages[i], theta0, theta1)) ** 2
        for i in range(len(prices))
    )
    ss_tot = sum(
        (prices[i] - mean_price) ** 2
        for i in range(len(prices))
    )
    r2 = 1 - (ss_res / ss_tot)
    return r2

def calculate_mse(mileages, prices, theta0, theta1):
    m = len(mileages)
    mse = sum(
        (estimate_price(mileages[i], theta0, theta1) - prices[i]) ** 2
        for i in range(m)
    ) / m
    return mse


# ─────────────────────────────────────
# 3. data visualization
# ─────────────────────────────────────

def plot(mileages, prices, theta0, theta1, r2):
    """
    scatter plot + regression line on the same graph
    
    scatter plot: each real data point
    regression line: connect two endpoints calculated from min and max mileage
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # scatter plot: each real data point
    ax.scatter(
        mileages, prices, 
        color="steelblue", 
        alpha=0.7,
        s=60,
        label="Real data"
    )

    # regression line: connect two endpoints calculated from min and max mileage
    x_min = min(mileages)
    x_max = max(mileages)
    y_min = estimate_price(x_min, theta0, theta1)
    y_max = estimate_price(x_max, theta0, theta1)

    ax.plot(
        [x_min, x_max],
        [y_min, y_max],
        color="tomato",
        linewidth=2,
        label=f"Regression line\nθ0={theta0:.1f}, θ1={theta1:.5f}"
    )

    # labels and formatting
    ax.set_xlabel("Mileage (km)", fontsize=12)
    ax.set_ylabel("Price (€)", fontsize=12)
    ax.set_title("Car Price vs Mileage - Linear Regression", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # show R² score on the plot
    ax.text(
        0.05, 0.95,
        f"R² = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig("result.png", dpi=150)
    print("\n📊 Plot saved as result.png")
    #plt.show()


# ─────────────────────────────────────
# 4. main
# ─────────────────────────────────────

def main():
    print("=" * 40)
    print("   ft_linear_regression - Bonus")
    print("=" * 40)

    mileages, prices = load_data()
    theta0, theta1 = load_thetas()

    if theta0 is None:
        return
    
    # calculate precision metrics
    r2 = calculate_r2(mileages, prices, theta0, theta1)
    mse = calculate_mse(mileages, prices, theta0, theta1)

    print(f"\n📐 Model precision:")
    print(f"   R²  = {r2:.4f}  (1.0 = perfect)")
    print(f"   MSE = {mse:.2f} €²")
    print(f"   RMSE= {math.sqrt(mse):.2f} €  (average error per car)")

    # visualize the data and the regression line
    plot(mileages, prices, theta0, theta1, r2)

if __name__ == "__main__":
    main()