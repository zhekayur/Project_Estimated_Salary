from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

def preprocess_data(df, target_column):
    # Preprocess the 'hobbies' and 'personality_traits' columns
    mlb_hobbies = MultiLabelBinarizer()
    mlb_personality_traits = MultiLabelBinarizer()

    # Apply MultiLabelBinarizer and add suffix to avoid column name conflicts
    hobbies_df = pd.DataFrame(mlb_hobbies.fit_transform(df['hobbies']),
                              columns=[f"hobby_{i}" for i in mlb_hobbies.classes_],
                              index=df.index)
    personality_traits_df = pd.DataFrame(mlb_personality_traits.fit_transform(df['personality_traits']),
                                         columns=[f"trait_{i}" for i in mlb_personality_traits.classes_],
                                         index=df.index)

    df = df.join(hobbies_df).join(personality_traits_df)

    # Drop the original 'hobbies' and 'personality_traits' columns
    df = df.drop(columns=['hobbies', 'personality_traits'])

    # Define the target variable and features
    target = df[target_column]
    features = df.drop(columns=['id', 'name', 'number', 'income_1_year', 'income_2_years', 'income_3_years'])

    return df, features, target
