import pandas as pd

def check_output_format(output_file, test_file):
    """Sanity check for the format of the output CSV."""
    output_df = pd.read_csv(output_file)
    test_df = pd.read_csv(test_file)

    if len(output_df) != len(test_df):
        print("Error: Number of rows in output file does not match test file.")
        return False

    if not all(col in output_df.columns for col in ['index', 'prediction']):
        print("Error: Missing required columns in the output file.")
        return False

    print("Sanity check passed!")
    return True

if __name__ == "__main__":
    check_output_format('output/submission.csv', 'data/test.csv')
