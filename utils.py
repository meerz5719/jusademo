import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_input(df):
    df = df.copy()

    # Extract date components
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df.drop(['date', 'id'], axis=1, errors='ignore', inplace=True)

    # Categorize family
    family_groups = {
        'FOODS': ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI','PRODUCE', 'DAIRY','POULTRY','EGGS','SEAFOOD'],
        'HOME': ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES'],
        'CLOTHING': ['LINGERIE', 'LADYSWARE'],
        'GROCERY': ['GROCERY I', 'GROCERY II'],
        'STATIONERY': ['BOOKS', 'MAGAZINES','SCHOOL AND OFFICE SUPPLIES'],
        'CLEANING': ['HOME CARE', 'BABY CARE','PERSONAL CARE'],
        'HARDWARE': ['PLAYERS AND ELECTRONICS','HARDWARE']
    }

    for key, values in family_groups.items():
        df['family'] = np.where(df['family'].isin(values), key, df['family'])

    # One-hot encoding
    cat_cols = ['family']
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded = encoder.fit_transform(df[cat_cols])
    enc_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(cat_cols))

    df = df.drop(cat_cols, axis=1)
    df = pd.concat([df.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)

    # Scaling numeric columns
    scaler = StandardScaler()
    num_cols = ['transactions', 'dcoilwtico']  # Add more if needed
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
