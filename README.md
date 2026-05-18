# ft_linear_regression


An introduction to machine learning — predicting car prices with linear regression and gradient descent.


## Overview

A simple linear regression model trained with gradient descent to predict the price of a car based on its mileage.

```
price = θ0 + θ1 × mileage
```

## Project Structure

```
ft_linear_regression/
├── data.csv          # Dataset (mileage, price)
├── thetas.json       # Saved model parameters (auto-generated after training)
├── train.py          # Training program — runs gradient descent
├── predict.py        # Prediction program — estimates price from mileage
├── bonus.py          # Bonus — precision metrics (R², MSE, RMSE) + plot
├── main.py           # Runs train → predict → bonus in one command
├── Dockerfile
└── Makefile
```

## Usage

### With Docker (recommended)

```
# Build image and run full pipeline (train + predict + bonus)
make

# Or step by step
make build      # Build Docker image
make run        # Run full pipeline
make train      # Only train
make predict    # Only predict
make bonus      # Only bonus (requires thetas.json)

# Cleanup
make clean      # Remove container + image
make fclean     # Remove container + image + thetas.json
make re         # Full rebuild and run
```

### Without Docker

```
pip install matplotlib
python3 main.py

# Or separately
python3 train.py
python3 predict.py
python3 bonus.py
```

## How It Works

### Training (train.py)

1. Load data.csv (24 data points)
2. Normalize mileage and price to [0, 1]
3. Run gradient descent for 1000 iterations:
```
tmp_θ0 = learningRate × (1/m) × Σ (estimatePrice(mileage[i]) − price[i])
tmp_θ1 = learningRate × (1/m) × Σ (estimatePrice(mileage[i]) − price[i]) × mileage[i]
```
4. Denormalize θ back to real-world scale
5. Save θ0 and θ1 to thetas.json

### Prediction (predict.py)

Prompts the user for a mileage and returns the estimated price using saved θ values. Defaults to θ0 = θ1 = 0 if not yet trained.

### Bonus (bonus.py)

- R² — how well mileage explains price variation (closer to 1.0 = better)
- MSE — mean squared error in €²
- RMSE — average prediction error in €
- Plot — scatter plot of real data + regression line, saved as result.png

### Example Output

```
🚗 Enter mileage (km): 80000
💰 Estimated price for 80000 km: 7177.33 €

📐 Model precision:
   R²   = 0.7329  (1.0 = perfect)
   MSE  = 445727.42 €²
   RMSE = 667.63 €
```

## Rules

- No libraries that do the regression work (e.g. numpy.polyfit, sklearn are forbidden)
- Gradient descent implemented from scratch
- matplotlib used only for visualization

## Author

42 School project — ft_linear_regression