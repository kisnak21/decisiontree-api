from flask import Flask, jsonify, request
from flask_cors import CORS
from firebase import firebase
import pandas as pd
import numpy as np
import os

# data dari firebase
firebase = firebase.FirebaseApplication(
    'https://smartlamp-48d48-default-rtdb.firebaseio.com/', None)
result = firebase.get('/Data', '')
data = result

# convert dataframe
data_list = []
for key, value in data.items():
    if 'Tanggal' in value and 'Waktu' in value:
        entry = {
            'Tanggal': value['Tanggal'],
            'Waktu': value['Waktu'],
            'kamar': value['kamar'],
            'kamar2': value['kamar2'],
            'teras': value['teras'],
            'dapur': value['dapur'],
            'toilet': value['toilet'],
            'ruangtamu': value['ruangtamu'],
        }
    data_list.append(value)
data = pd.DataFrame(data_list)

# Mengubah format waktu menjadi timedelta
data['Waktu'] = pd.to_datetime(
    data['Waktu'], format='%H:%M:%S', errors='coerce').dt.time

# Menghitung total menit dari timedelta
data['Waktu'] = data['Waktu'].apply(lambda x: x.hour * 60 + x.minute)

# Mengisi data waktu yang hilang dengan rata-rata waktu
mean_time = data['Waktu'].mean()
data['Waktu'].fillna(mean_time, inplace=True)

# Mengganti nilai hidup dan mati
data['kamar'] = data['kamar'].apply(lambda x: 1 if x == 'hidup' else 2)
data['kamar2'] = data['kamar2'].apply(lambda x: 1 if x == 'hidup' else 2)
data['teras'] = data['teras'].apply(lambda x: 1 if x == 'hidup' else 2)
data['dapur'] = data['dapur'].apply(lambda x: 1 if x == 'hidup' else 2)
data['toilet'] = data['toilet'].apply(lambda x: 1 if x == 'hidup' else 2)
data['ruangtamu'] = data['ruangtamu'].apply(lambda x: 1 if x == 'hidup' else 2)


# Bagi dataset 80:20
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Menghitung entropy


def entropy(data, column):
    labels = data[column]
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    probabilities = label_counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return entropy


# Menghitung information gain


def information_gain(data, feature, column):
    total_entropy = entropy(data, column)
    feature_values = data[feature]
    unique_values = np.unique(feature_values)
    weighted_entropy = 0

    for value in unique_values:
        subset = data[data[feature] == value]
        subset_entropy = entropy(subset, column)
        subset_probability = len(subset) / len(data)
        weighted_entropy += subset_probability * subset_entropy

    information_gain_value = total_entropy - weighted_entropy
    return information_gain_value


# Menghitung gain ratio


def gain_ratio(data, feature, column):
    split_information = 0
    feature_values = data[feature]
    unique_values, value_counts = np.unique(feature_values, return_counts=True)

    for value, count in zip(unique_values, value_counts):
        subset = data[data[feature] == value]
        subset_probability = len(subset) / len(data)
        split_information -= subset_probability * \
            np.log2(subset_probability + 1e-9)

    information_gain_value = information_gain(data, feature, column)
    gain_ratio_value = information_gain_value / \
        (split_information + 1e-9) if split_information != 0 else 0
    return gain_ratio_value


# Mencari atribut terbaik untuk splitting menggunakan Gain Ratio


def find_best_attribute(data, column):
    features = data.columns.drop(['Tanggal', column])
    best_attribute = None
    best_gain_ratio = -1

    for feature in features:
        current_gain_ratio = gain_ratio(data, feature, column)
        if current_gain_ratio > best_gain_ratio:
            best_gain_ratio = current_gain_ratio
            best_attribute = feature

    return best_attribute


# Membangun pohon keputusan C5.0


def build_decision_tree(data, column):
    labels = data[column]

    if len(np.unique(labels)) == 1:
        return {'label': labels.iloc[0]}

    if len(data.columns) == 1:
        majority_label = labels.mode()[0]
        return {'label': majority_label}

    best_attribute = find_best_attribute(data, column)
    if best_attribute is None:
        majority_label = labels.mode()[0]
        return {'label': majority_label}

    tree = {'attribute': best_attribute, 'children': {}}
    unique_values = np.unique(data[best_attribute])

    for value in unique_values:
        subset = data[data[best_attribute] == value].drop(
            columns=[best_attribute])
        tree['children'][value] = build_decision_tree(subset, column)

    return tree


# Prediksi menggunakan pohon keputusan


def predict(tree, sample, column):
    if 'label' in tree:
        return tree['label']
    attribute = tree['attribute']
    value = sample[attribute]

    if value in tree['children']:
        child = tree['children'][value]
        return predict(child, sample, column)
    else:
        return None


# Menghitung confusion matrix


def confusion_matrix(y_true, y_pred, labels):
    num_labels = len(labels)
    cm = np.zeros((num_labels, num_labels), dtype=int)

    for i in range(num_labels):
        true_label = labels[i]
        for j in range(num_labels):
            pred_label = labels[j]
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    return cm


# Menghitung recall


def recall(cm, label_idx):
    true_positives = cm[label_idx, label_idx]
    total_positives = np.sum(cm[label_idx, :])
    recall_value = true_positives / total_positives if total_positives != 0 else 0
    return recall_value

# Menghitung precision


def precision(cm, label_idx):
    true_positives = cm[label_idx, label_idx]
    total_predicted_positives = np.sum(cm[:, label_idx])
    precision_value = true_positives / \
        total_predicted_positives if total_predicted_positives != 0 else 0
    return precision_value


# Memperoleh akurasi prediksi pada data uji


def get_accuracy(tree, data, column):
    predicted_labels = data.apply(lambda x: predict(tree, x, column), axis=1)
    actual_labels = data[column]
    accuracy = (predicted_labels == actual_labels).mean()
    return accuracy


# Menghitung F1-Score


def f1_score(cm, label_idx):
    precision_value = precision(cm, label_idx)
    recall_value = recall(cm, label_idx)
    f1_score_value = 2 * (precision_value * recall_value) / (precision_value +
                                                             recall_value) if (precision_value + recall_value) != 0 else 0
    return f1_score_value


# Membuat rules dari pohon keputusan


def get_rules(tree, column, conditions=None):
    rules = []

    if conditions is None:
        conditions = []

    if 'label' in tree:
        rule = {
            'conditions': conditions,
            'decision': tree['label']
            # 'decision': 1 if tree['label'] == 'hidup' else 2
        }
        rules.append(rule)
        return rules

    attribute = tree['attribute']
    for value, child in tree['children'].items():
        new_conditions = conditions + ["{} == {}".format(attribute, value)]
        rules.extend(get_rules(child, column, new_conditions))

    return rules


# Menghitung akurasi pada data uji
for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
    decision_tree = build_decision_tree(train_data, column)
    accuracy = get_accuracy(decision_tree, test_data, column)
    print("Accuracy ({}) : {:.2f}".format(column, accuracy))


for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
    decision_tree = build_decision_tree(train_data, column)
    y_test = test_data[column]
    y_pred = test_data.apply(lambda x: predict(
        decision_tree, x, column), axis=1)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels)

    print("Confusion Matrix ({}) :".format(column))
    print(cm)
    print()

    for i, label in enumerate(labels):
        r = recall(cm, i)
        p = precision(cm, i)
        print("Label: {} ({})".format(label, column))
        print("Recall: {:.2f}".format(r))
        print("Precision: {:.2f}".format(p))
        print()

for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
    decision_tree = build_decision_tree(train_data, column)
    rules = get_rules(decision_tree, column)

    print("Rules ({}) :".format(column))
    for i, rule in enumerate(rules):
        print("Rule {}: {}".format(i+1, rule))
    print()


# Membuat aturan dari pohon keputusan
print("Rules:")
rl = get_rules(decision_tree, column)
print(rl)

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

# Endpoint untuk menghitung entropy


@app.route('/api/entropy')
def calculate_entropy():
    entropy_values = {}
    for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
        entropy_value = entropy(data, column)
        entropy_values[column] = entropy_value
    return jsonify(entropy=entropy_values)


# Endpoint untuk menghitung information gain


@app.route('/api/information_gain')
def calculate_information_gain():
    # Mendapatkan parameter 'feature' dari URL
    feature = request.args.get('feature')
    information_gain_value = information_gain(data, feature, column)
    return jsonify(information_gain=information_gain_value)

# Endpoint untuk menghitung gain ratio


@app.route('/api/gain_ratio')
def calculate_gain_ratio():
    # Mendapatkan parameter 'feature' dari URL
    feature = request.args.get('feature')
    gain_ratio_value = gain_ratio(data, feature, column)
    return jsonify(gain_ratio=gain_ratio_value)

# Endpoint untuk mencari atribut terbaik


@app.route('/api/best_attribute')
def find_best_attribute_endpoint():
    best_attribute = find_best_attribute(data)
    return jsonify(best_attribute=best_attribute)

# Buat endpoint untuk menghitung akurasi


@app.route('/api/accuracy')
def calculate_accuracy():
    accuracy_values = {}
    for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
        decision_tree = build_decision_tree(train_data, column)
        accuracy = get_accuracy(decision_tree, test_data, column)
        accuracy_percentage = "{:.2f}".format(accuracy, column)
        accuracy_values[column] = accuracy_percentage
    return jsonify(accuracy=accuracy_values)


# rules pohon


@app.route('/api/rules')
def show_rules():
    rules = {}
    for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
        decision_tree = build_decision_tree(train_data, column)
        column_rules = get_rules(decision_tree, column)
        formatted_rules = []
        for i, rule in enumerate(column_rules):
            formatted_rules.append("Rule {}: {}".format(i+1, rule))
        rules[column] = formatted_rules
    return jsonify(rules=rules)


@app.route('/api/result')
def show_result():
    result = {}
    for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
        decision_tree = build_decision_tree(train_data, column)
        y_test = test_data[column]
        y_pred = test_data.apply(lambda x: predict(
            decision_tree, x, column), axis=1)
        labels = np.unique(y_test)
        cm = confusion_matrix(y_test, y_pred, labels)

        recall_values = []
        precision_values = []
        for i, label in enumerate(labels):
            r = recall(cm, i)
            p = precision(cm, i)
            recall_values.append(r)
            precision_values.append(p)

        accuracy = get_accuracy(decision_tree, test_data, column)
        entropy_value = entropy(data, column)
        precision_value_avg = np.mean(precision_values)
        recall_value_avg = np.mean(recall_values)

        result[column] = {
            'entropy': entropy_value,
            'accuracy': "{:.2f}".format(accuracy),
            'precision on': "{:.2f}".format(precision_values[0]),
            'precision off': "{:.2f}".format(precision_values[1]),
            'recall on': "{:.2f}".format(recall_values[0]),
            'recall off': "{:.2f}".format(recall_values[1]),
            'precision_avg': "{:.2f}".format(precision_value_avg),
            'recall_avg': "{:.2f}".format(recall_value_avg)
        }

    return jsonify(result=result)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
