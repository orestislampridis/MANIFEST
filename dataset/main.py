import create_dataset


def main():
    data_separated = create_dataset.get_separated_dataset()
    data_combined = create_dataset.get_combined_dataset()
    print(data_separated)
    print(data_combined)

    data_separated.to_csv('data_csv/data_separated.csv', index=False)
    data_combined.to_csv('data_csv/data_combined.csv', index=False)


if __name__ == "__main__":
    main()
