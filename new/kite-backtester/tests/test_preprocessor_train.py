import os
from models.preprocessing.preprocessor_train import PreprocessorTrain

def test_preprocessor_train():
    """
    Test preprocessing on sample data.
    This function loads the STATE_BANK_OF_INDIA.csv file, preprocesses it, and saves the cleaned data to CSV files.
    """ 
    # Load and preprocess
    processor = PreprocessorTrain(filepath="data/NIFTY_50.csv", model_type="with sideways")
    target = "price_dir"
    X, y = processor.run(target=target)

    print("Preprocessing completed.")
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    # Output folder
    # os.makedirs("models", exist_ok=True)

    # Save to CSV
    X.to_csv("data/x_train.csv", index=False)
    y.to_frame(name=target).to_csv("data/y_train.csv", index=False)

    print("Saved cleaned X and y to outputs/")
