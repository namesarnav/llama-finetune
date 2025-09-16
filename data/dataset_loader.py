from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder


def preprocess_data(tokenizer, train_file="train.csv", test_file="test.csv"):
    dataset = load_dataset(
        "csv",
        data_files={"train": train_file, "test": test_file}
    )

    # clean + rename
    dataset['train'] = dataset['train'].rename_column("label", "labels").remove_columns(["idx"])
    dataset['test'] = dataset['test'].rename_column("label", "labels").remove_columns(["idx"])

    # encode labels
    le = LabelEncoder()
    all_labels = list(dataset['train']['labels']) + list(dataset['test']['labels'])
    le.fit(all_labels)

    dataset['train'] = dataset['train'].map(lambda x: {'text': x['text'], 'labels': le.transform([x['labels']])[0]})
    dataset['test'] = dataset['test'].map(lambda x: {'text': x['text'], 'labels': le.transform([x['labels']])[0]})

    # tokenize
    def tokenize(x):
        return tokenizer(x["text"], padding=True, truncation=True, max_length=512)

    tk_data_train = dataset['train'].map(tokenize, batched=False)
    tk_data_test = dataset['test'].map(tokenize, batched=False)

    tk_data_train.set_format('torch')
    tk_data_test.set_format('torch')

    # drop raw text
    tk_data_train = tk_data_train.remove_columns(["text"])
    tk_data_test = tk_data_test.remove_columns(["text"])

    return tk_data_train, tk_data_test, le
