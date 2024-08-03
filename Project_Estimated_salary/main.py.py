import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from preprocessing import preprocess_data
from model import train_and_evaluate_model

# Load the dataset
file_path = "data/University_Graduates_Data__Realistic__10000_.csv"
df = pd.read_csv(file_path)

# Simulate income data (replace with actual income data if available)
df['income_1_year'] = df['city'].map({
    'Gdansk': 6920, 'Katowice': 6310, 'Krakow': 8320, 'Lublin': 6510,
    'Poznan': 7120, 'Szczecin': 6710, 'Warsaw': 8550, 'Wroclaw': 8090
}) + df['average_grade'] * 100  # Simulated adjustment based on average grade

df['income_2_years'] = df['income_1_year'] * 1.05  # Assuming a 5% increase each year
df['income_3_years'] = df['income_2_years'] * 1.05

# Preprocess the data
df_preprocessed, features, target = preprocess_data(df, 'income_1_year')

# Train and evaluate the model
mae, r2 = train_and_evaluate_model(df_preprocessed, features, target)

print("Mean Absolute Error:", mae)
print("R-squared:", r2)


# Function to create subplots for different attributes
def create_subplots(df, attributes, filename):
    unique_value_counts = [len(df[attribute].unique()) for attribute in attributes]
    total_subplots = sum(unique_value_counts)
    num_cols = 2
    num_rows = (total_subplots + num_cols - 1) // num_cols

    # Create subplots
    fig = make_subplots(rows=num_rows, cols=num_cols,
                        subplot_titles=[f"{attr}: {val}" for attr in attributes for val in df[attr].unique()])

    subplot_index = 1
    for attribute in attributes:
        unique_values = df[attribute].unique()
        for value in unique_values:
            filtered_df = df[df[attribute] == value]
            row = (subplot_index - 1) // num_cols + 1
            col = (subplot_index - 1) % num_cols + 1
            fig.add_trace(
                go.Scatter(x=filtered_df.index, y=filtered_df['income_1_year'], mode='lines+markers',
                           name=f'Income in 1 Year ({value})'),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(x=filtered_df.index, y=filtered_df['income_2_years'], mode='lines+markers',
                           name=f'Income in 2 Years ({value})'),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(x=filtered_df.index, y=filtered_df['income_3_years'], mode='lines+markers',
                           name=f'Income in 3 Years ({value})'),
                row=row, col=col
            )
            subplot_index += 1

    fig.update_layout(height=300 * num_rows, width=1200, title_text='Income Estimations')
    fig.write_html(filename)
    print(f"{filename} created successfully.")


# Function to create subplots for best and worst cases
def create_best_worst_subplots(df, filename):
    df_sorted = df.sort_values(by='income_1_year')
    worst_cases = df_sorted.head(20)
    best_cases = df_sorted.tail(20)

    fig = make_subplots(rows=2, cols=1, subplot_titles=['20 Worst Cases', '20 Best Cases'])

    # Plot worst cases
    fig.add_trace(
        go.Scatter(x=worst_cases.index, y=worst_cases['income_1_year'], mode='lines+markers',
                   name='Income in 1 Year (Worst Cases)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=worst_cases.index, y=worst_cases['income_2_years'], mode='lines+markers',
                   name='Income in 2 Years (Worst Cases)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=worst_cases.index, y=worst_cases['income_3_years'], mode='lines+markers',
                   name='Income in 3 Years (Worst Cases)'),
        row=1, col=1
    )

    # Plot best cases
    fig.add_trace(
        go.Scatter(x=best_cases.index, y=best_cases['income_1_year'], mode='lines+markers',
                   name='Income in 1 Year (Best Cases)'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=best_cases.index, y=best_cases['income_2_years'], mode='lines+markers',
                   name='Income in 2 Years (Best Cases)'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=best_cases.index, y=best_cases['income_3_years'], mode='lines+markers',
                   name='Income in 3 Years (Best Cases)'),
        row=2, col=1
    )

    fig.update_layout(height=800, width=1200, title_text='Best and Worst Income Estimations')
    fig.write_html(filename)
    print(f"{filename} created successfully.")


# Create subplots for the first page
create_subplots(df, ['city', 'origin_country', 'language'], 'page1.html')

# Create subplots for the second page
create_subplots(df, ['age', 'degree', 'sex'], 'page2.html')

# Create subplots for the third page
create_subplots(df, ['learning_pace', 'career_focus', 'personality_traits'], 'page3.html')

# Create subplots for best and worst cases
create_best_worst_subplots(df, 'best_worst.html')

