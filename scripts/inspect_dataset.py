from datasets import load_dataset


DATASET_NAME = "openlifescienceai/medmcqa"


def main() -> None:
    print(f"Loading {DATASET_NAME}...")

    dataset = load_dataset(DATASET_NAME)

    print("\n=== Dataset ===")
    print(dataset)

    print("\n=== Training Features ===")
    print(dataset["train"].features)

    print("\n=== First Training Example ===")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()