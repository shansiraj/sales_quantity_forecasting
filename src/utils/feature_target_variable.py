def define_features_and_target_variables(df):
    """Defines the target variable, which is item_qty."""

    features = df.drop(columns=["item_qty"])
    target = df["item_qty"]

    return features,target