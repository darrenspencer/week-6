import pandas as pd

def read_data(file_path):
    """
    Read the data from the baseball_salary.csv file and return it as a pandas DataFrame

    Parameters:
    file_path (str): The path to the baseball_salary.csv file

    Returns:
    pandas.DataFrame: The data from the baseball_salary.csv file
    """
    data = pd.read_csv(file_path)
    return data

def calculate_descriptive_statistics(data):
    """
    Calculate the descriptive statistics for the data in the DataFrame

    Parameters:
    data (pandas.DataFrame): The data to calculate the descriptive statistics for

    Returns:
    pandas.DataFrame: The descriptive statistics for the data
    """
    descriptive_stats = data.describe()
    return descriptive_stats

def main():
    file_path = "baseball_salary.csv"
    data = read_data(file_path)
    descriptive_stats = calculate_descriptive_statistics(data)
    print(descriptive_stats)

if __name__ == "__main__":
    main()
