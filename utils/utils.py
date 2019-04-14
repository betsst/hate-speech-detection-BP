import torch


# count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_weights(labels, config):
    dataset_size = len(labels)
    classes = set(labels)
    num_classes = len(classes)

    if config['balanced_weights']:
        return num_classes, [[1.0] * num_classes]

    classes_occurrences = [0] * num_classes
    for label in labels:
        classes_occurrences[label] += 1

    max_count = max(classes_occurrences)
    class_weights = [max_count / n for n in classes_occurrences]

    return num_classes, class_weights


def ids2str(ids, vocab):
    batch_text = []
    for doc in ids:
        doc_text = [vocab.itos[id] for id in doc]
        batch_text.append(doc_text)
    return batch_text


def save_model(model_path, model):
    torch.save(model.state_dict(), model_path)
