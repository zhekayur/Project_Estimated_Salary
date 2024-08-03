# Project Estimated Salary

This project aims to estimate the income of recent university graduates in Poland over 1 year, 2 years, and 3 years based on various attributes such as city, degree, hobbies, personality traits, etc. The project includes data preprocessing, model training, evaluation, and visualization.

## Project Structure

Project2/
|-- .venv/
|-- data/
| |-- University_Graduates_Data__Realistic__10000_.csv
|-- main.py
|-- preprocessing.py
|-- model.py
|-- requirements.txt
|-- README.md
|-- .gitignore

perl
Copy code

## Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
Usage
Preprocess Data:
Preprocess the dataset and simulate income data.

Train and Evaluate Model:
Train a RandomForestRegressor model to predict income and evaluate its performance.

Generate Visualizations:
Create interactive HTML visualizations for income estimations based on various attributes.

Running the Project
To run the project, execute the main.py script:

bash
Copy code
python main.py
HTML Outputs
The project generates four HTML files containing interactive plots:

page1.html: Income estimations filtered by city, origin country, and language.
page2.html: Income estimations filtered by age, degree, and sex.
page3.html: Income estimations filtered by learning pace, career focus, and personality traits.
best_worst.html: 20 worst and 20 best income estimations.
Files Description
main.py: Main script to run the project.
preprocessing.py: Data preprocessing functions.
model.py: Model training and evaluation functions.
requirements.txt: List of dependencies.
README.md: Project documentation.
.gitignore: Git ignore file.