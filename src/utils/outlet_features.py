from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def create_outlet_features_with_given_encoder(df,encoder):

    # Transform the data frame
    encoded_features = encoder.transform(df[['store']])

    # Convert the encoded arrays back to DataFrames with appropriate column names
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['store']))

    # Reset the indices of train_df and test_df to merge them with the encoded dataframes
    df.reset_index(drop=True, inplace=True)

    # Concatenate the original data (excluding the encoded columns) with the encoded data
    df_final = pd.concat([df.drop(['store'], axis=1), encoded_df], axis=1)

    return df_final

def create_outlet_features(train_df,test_df):
    """Create outlet(store) related features."""

    print ("Creating outlet(store) features started")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit the encoder on the training data
    encoder.fit(train_df[['store']])

    train_df_encoded = create_outlet_features_with_given_encoder(train_df,encoder)
    test_df_encoded = create_outlet_features_with_given_encoder(test_df,encoder)

    print ("Creating outlet(store) features ended")

    return train_df_encoded,test_df_encoded
