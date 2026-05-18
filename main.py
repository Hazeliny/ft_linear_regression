from train import (
    load_data,
    normalize,
    gradient_descent,
    denormalize_thetas,
    save_thetas
)

from predict import (
    load_thetas,
    estimate_price,
    get_mileage_from_user
)

def main():
    print("=" * 40)
    print("   ft_linear_regression - Train and Predict")
    print("=" * 40)

    # ─────────────────────────────────────
    # PART 1 : TRAINING
    # ─────────────────────────────────────
    print("\n📚 STEP 1 : Training the model...\n")

    mileages, prices = load_data("data.csv")

    mileages_norm, km_min, km_max = normalize(mileages)
    prices_norm, price_min, price_max = normalize(prices)

    print(f"📊 Mileage range : {km_min:.0f} ~ {km_max:.0f} km")
    print(f"📊 Price range   : {price_min:.0f} ~ {price_max:.0f} €\n")

    theta0_norm, theta1_norm = gradient_descent(
        mileages_norm, prices_norm, 
        learning_rate=0.1, 
        iterations=1000
    )

    theta0, theta1 = denormalize_thetas(
        theta0_norm, theta1_norm,
        km_min, km_max,
        price_min, price_max
    )

    save_thetas(theta0, theta1)

    print("\n✅ Training complete! Now let's predict a price.\n")

    # ─────────────────────────────────────
    # PART 2 : PREDICTION
    # ─────────────────────────────────────
    print("\n" + "=" * 40)
    print("\n🔮 STEP 2 : Predict a car price...\n")

    # load the thetas we just saved
    theta0, theta1 = load_thetas()

    # ask user for mileage and predict
    mileage = get_mileage_from_user()
    price = estimate_price(mileage, theta0, theta1)

    if price < 0:
        print(f"\n⚠️  Predicted price : 0 € "
              f"(model returned {price:.2f}, clamped to 0)")
    else:
        print(f"\n💰 Estimated price for {mileage:.0f} km : {price:.2f} €")

    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()