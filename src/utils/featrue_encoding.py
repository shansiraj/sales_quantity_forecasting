from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def encode_outlet_with_given_encoder(df,encoder):

    # Transform the data frame
    encoded_features = encoder.transform(df[['store']])

    # Convert the encoded arrays back to DataFrames with appropriate column names
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['store']))

    # Reset the indices of train_df and test_df to merge them with the encoded dataframes
    df.reset_index(drop=True, inplace=True)

    # Concatenate the original data (excluding the encoded columns) with the encoded data
    df_final = pd.concat([df, encoded_df], axis=1)

    return df_final

def encode_department_with_given_encoder(df,encoder):

    # Transform the data frame
    encoded_features = encoder.transform(df[['item_dept']])

    # Convert the encoded arrays back to DataFrames with appropriate column names
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['item_dept']))

    # Reset the indices of train_df and test_df to merge them with the encoded dataframes
    df.reset_index(drop=True, inplace=True)

    # Concatenate the original data (excluding the encoded columns) with the encoded data
    df_final = pd.concat([df, encoded_df], axis=1)

    return df_final

def encode_features(train_df,test_df):
    """Feature encoding related commands."""

    print ("Encoding outlet(store) features started")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[['store']])

    train_df = encode_outlet_with_given_encoder(train_df,encoder)
    test_df = encode_outlet_with_given_encoder(test_df,encoder)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[['item_dept']])

    train_df = encode_department_with_given_encoder(train_df,encoder)
    test_df = encode_department_with_given_encoder(test_df,encoder)

    print ("Creating item (deparment level) features ended")

    return train_df,test_df
