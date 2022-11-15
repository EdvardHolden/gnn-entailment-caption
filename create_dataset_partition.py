from glob import glob
from sklearn.model_selection import train_test_split


def save_set(ids, name):
    with open("raw/" + name + ".txt", "w") as f:
        f.write("\n".join(ids))


def main():
    # Get all the problem names
    files = glob("nndata/*")
    # Remove folder name
    files = [f.split("/")[1] for f in files]
    print("Number or of problems: ", len(files))

    # Split into train and test
    train, test = train_test_split(files, shuffle=True, test_size=0.1)

    # Set first 128 as training set
    val = train[:128]
    train = train[128:]
    print("Length of train: ", len(train))
    print("Length of val:   ", len(val))
    print("Length of test:  ", len(test))

    save_set(train, "train")
    save_set(val, "validation")
    save_set(test, "test")


if __name__ == "__main__":
    main()
