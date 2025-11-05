import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tkinter import Tk, messagebox, simpledialog


# ===== ENHANCED HELPER FUNCTIONS =====
def validate_experience(years, position_level):
    """Enhanced experience validation with position-specific rules"""
    max_exp = df['Years of Experience'].max()

    min_experience = {
        1: 0.5,  # Business Analyst: 6-month minimum
        2: 1.0,  # Junior Consultant: 1-year minimum
        3: 2.0,  # Senior Consultant: 2-year minimum
        4: 4.0,  # Manager: 4-years minimum
        5: 5.0,  # Country Manager: 5-years minimum
        6: 6.0,  # Region Manager: 6-year minimum
        7: 7.0,  # Partner: 7-year minimum
        8: 8.0,  # Senior Partner: 8--year minimum
        9: 9.0,  # C-level: 9-years minimum
        10: 10.0  # CEO: 10 years minimum
    }

    required_exp = min_experience.get(position_level, 0.5)

    if years < required_exp:
        return False, f"Position level {position_level} requires minimum {required_exp} years of experience"
    elif years > max_exp:
        return False, f"Experience cannot exceed {max_exp} years based on our dataset"
    return True, ""


def format_salary(salary):
    """Format salary as South African Rand"""
    return f"R{salary:,.2f}"


def predict_with_confidence(model, input_data):
    """Calculate mean prediction and 95% confidence interval"""
    predictions = np.array([tree.predict(input_data) for tree in model.estimators_])
    mean = predictions.mean()
    std = predictions.std()
    return mean, mean - 2 * std, mean + 2 * std


def prepare_data():
    """Prepare the dataset"""
    data = {
        'Position Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Position Name': [
            'Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager',
            'Country Manager', 'Region Manager', 'Partner', 'Senior Partner',
            'C-level', 'CEO'
        ],
        'Years of Experience': [2, 5, 1, 10, 3, 7, 4, 12, 6, 8],
        'Salary': [25000, 35000, 20000, 60000, 28000, 45000, 32000, 70000, 38000, 50000]
    }
    return pd.DataFrame(data)


def train_model(df):
    """Train and evaluate the model"""
    X = df[['Position Level', 'Years of Experience']].values
    y = df['Salary'].values

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    # Cross-validation
    scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    # Fit the model on all data
    rf.fit(X, y)
    return rf


def get_position_labels(df):
    """Generate position level labels"""
    labels = [f"{row['Position Level']}: {row['Position Name']}" for _, row in df.iterrows()]
    return "\n".join(labels)


def predict_salary(df, model):
    """Handle salary prediction interface"""
    root = Tk()
    root.withdraw()
    try:
        position_labels = get_position_labels(df)

        position_level = simpledialog.askinteger(
            "Position Level",
            f"Enter Position Level (1-10):\n\n{position_labels}",
            parent=root,
            minvalue=1,
            maxvalue=10
        )
        if position_level is None:
            messagebox.showinfo("Cancelled", "Prediction cancelled.")
            return

        while True:
            years_of_experience = simpledialog.askfloat(
                "Years of Experience",
                "Enter Years of Experience (in years):",
                parent=root,
                minvalue=0.0
            )
            if years_of_experience is None:
                messagebox.showinfo("Cancelled", "Prediction cancelled.")
                return

            is_valid, message = validate_experience(years_of_experience, position_level)
            if is_valid:
                break
            messagebox.showerror("Invalid Input", message)

        # Create input array without feature names
        input_data = np.array([[position_level, years_of_experience]])
        mean_salary, lower_bound, upper_bound = predict_with_confidence(model, input_data)

        confidence_factor = min(1.0, years_of_experience / df['Years of Experience'].mean())
        confidence_message = ""
        if confidence_factor < 0.5:
            confidence_message = "\n\nNote: Prediction confidence is low due to limited experience."
        elif confidence_factor < 0.8:
            confidence_message = "\n\nNote: Prediction confidence is moderate."

        message = (
            f"Predicted Salary: {format_salary(mean_salary)}\n"
            f"95% Confidence Range:\n"
            f"{format_salary(lower_bound)} - {format_salary(upper_bound)}"
            f"{confidence_message}"
        )
        messagebox.showinfo("Prediction Results", message)

    finally:
        root.destroy()


def visualize_predictions(df, model):
    """Create visualization of predictions"""
    years_grid = np.linspace(df['Years of Experience'].min(), df['Years of Experience'].max(), 100)
    plt.figure(figsize=(10, 6))

    for level, name in zip(df['Position Level'], df['Position Name']):
        grid = np.array([[level, x] for x in years_grid])
        pred = model.predict(grid)
        plt.plot(years_grid, pred, label=f'Level {level} - {name}')

    plt.scatter(df['Years of Experience'], df['Salary'], color='black', zorder=5, label='Actual Data')
    for _, row in df.iterrows():
        plt.annotate(
            f"L{int(row['Position Level'])}",
            (row['Years of Experience'], row['Salary']),
            textcoords="offset points", xytext=(5, 5), ha='left'
        )

    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Random Forest Regression: Salary vs Experience by Position Level')
    plt.legend(title='Position Levels', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Prepare data and train model
    df = prepare_data()
    rf = train_model(df)

    # Run prediction interface
    predict_salary(df, rf)

    # Show visualization
    visualize_predictions(df, rf)
