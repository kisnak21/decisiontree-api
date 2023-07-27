from flask import Flask, jsonify, request
from flask_cors import CORS
from firebase import firebase
from datetime import datetime
import pandas as pd
import numpy as np
import os

# data dari firebase langsung
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
data['kamar'] = data['kamar'].map({'mati': 1, 'hidup': 2})
data['kamar2'] = data['kamar2'].map({'mati': 1, 'hidup': 2})
data['teras'] = data['teras'].map({'mati': 1, 'hidup': 2})
data['dapur'] = data['dapur'].map({'mati': 1, 'hidup': 2})
data['toilet'] = data['toilet'].map({'mati': 1, 'hidup': 2})
data['ruangtamu'] = data['ruangtamu'].map({'mati': 1, 'hidup': 2})


# Bagi dataset 80:20
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# question node


class CreateQuestion:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def check(self, data):
        val = data[self.column]
        return val >= self.value

    def __repr__(self):
        return "Is %s >= %s?" % (self.column, str(self.value))

# Decision tree class


class DecisionTree:
    def __init__(self):
        self.tree = None

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy

    def information_gain(self, X, y, feature):
        entropy_before = self.entropy(y)
        unique_values = np.unique(X[feature])
        weighted_entropy = 0

        for value in unique_values:
            subset_indices = X[feature] == value
            subset_y = y[subset_indices]
            subset_weight = len(subset_indices) / len(y)
            subset_entropy = self.entropy(subset_y)
            weighted_entropy += subset_weight * subset_entropy

        information_gain = entropy_before - weighted_entropy
        return information_gain

    def gain_ratio(self, X, y, feature):
        information_gain = self.information_gain(X, y, feature)
        split_info = self.entropy(X[feature])
        if split_info == 0:
            return 0
        gain_ratio = information_gain / split_info
        return gain_ratio

    def find_best_attribute(self, X, y):
        best_gain_ratio = -np.inf
        best_attribute = None

        for attribute in X.columns:
            gain_ratio = self.gain_ratio(X, y, attribute)
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_attribute = attribute

        return best_attribute

    def build_decision_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return {'label': y.iloc[0]}
        if X.empty or len(X.columns) == 0:
            majority_label = np.argmax(np.bincount(y))
            return {'label': majority_label}

        best_attribute = self.find_best_attribute(X, y)
        if best_attribute is None:
            majority_label = np.argmax(np.bincount(y))
            return {'label': majority_label}

        tree = {'attribute': best_attribute, 'children': {}}
        unique_values = np.unique(X[best_attribute])

        for value in unique_values:
            subset_indices = X[best_attribute] == value
            subset_X = X.loc[subset_indices].drop(columns=[best_attribute])
            subset_y = y[subset_indices]
            tree['children'][value] = self.build_decision_tree(
                subset_X, subset_y)

        return tree

    def fit(self, X, y):
        self.tree = self.build_decision_tree(X, y)

    def predict_instance(self, instance, node):
        if 'label' in node:
            return node['label']
        attribute = node['attribute']
        value = instance[attribute]

        if value in node['children']:
            child = node['children'][value]
            return self.predict_instance(instance, child)
        else:
            return None

    def predict(self, X):
        predictions = []
        for _, instance in X.iterrows():
            predictions.append(self.predict_instance(instance, self.tree))
        return predictions

    def print_rules(self, node=None, question='', rules_data=None):
        if node is None:
            node = self.tree

        if rules_data is None:
            rules_data = []

        if 'label' in node:
            rules_data.append({'Question': question, 'Label': node['label']})
            return

        attribute = node['attribute']
        for value, child_node in node['children'].items():
            q = CreateQuestion(attribute, value)
            self.print_rules(child_node, question +
                             str(q), rules_data)

        return rules_data

    def print_tree(self, node=None, indent=0, tree_data=None):
        if node is None:
            node = self.tree

        if tree_data is None:
            tree_data = []

        if 'label' in node:
            tree_data.append(
                {'Node': 'Label', 'Value': node['label'], 'Indent': indent})
            return

        attribute = node['attribute']
        tree_data.append(
            {'Node': 'Attribute', 'Value': attribute, 'Indent': indent})

        for value, child_node in node['children'].items():
            question = CreateQuestion(attribute, value)
            tree_data.append(
                {'Node': 'Question', 'Value': repr(question), 'Indent': indent + 1})
            self.print_tree(child_node, indent + 2, tree_data)

        return tree_data

# akurasi


def calculate_accuracy(predictions, labels):
    correct_count = sum(predictions == labels)
    total_count = len(predictions)
    accuracy = correct_count / total_count
    return accuracy


def confusion_matrix(y_true, y_pred, labels):
    num_labels = len(labels)
    cm = np.zeros((num_labels, num_labels), dtype=int)

    for i in range(num_labels):
        true_label = labels[i]
        for j in range(num_labels):
            pred_label = labels[j]
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    return cm


def recall(cm, label_idx):
    true_positives = cm[label_idx, label_idx]
    total_positives = np.sum(cm[label_idx, :])
    recall_value = true_positives / total_positives if total_positives != 0 else 0
    return recall_value


def precision(cm, label_idx):
    true_positives = cm[label_idx, label_idx]
    total_predicted_positives = np.sum(cm[:, label_idx])
    precision_value = true_positives / \
        total_predicted_positives if total_predicted_positives != 0 else 0
    return precision_value


def generate_new_data():
    current_time = datetime.now().time().strftime('%H:%M:%S')
    new_data = pd.DataFrame({'Waktu': [current_time]})
    new_data['Waktu'] = pd.to_datetime(
        new_data['Waktu'], format='%H:%M:%S', errors='coerce').dt.time
    new_data['Waktu'] = new_data['Waktu'].apply(
        lambda x: x.hour * 60 + x.minute)
    new_data['Waktu'].fillna(mean_time, inplace=True)
    return new_data


# Membangun pohon keputusan C5.0 untuk setiap ruangan
decision_tree_kamar = DecisionTree()
decision_tree_kamar.fit(train_data[['Waktu']], train_data['kamar'])

decision_tree_kamar2 = DecisionTree()
decision_tree_kamar2.fit(train_data[['Waktu']], train_data['kamar2'])

decision_tree_teras = DecisionTree()
decision_tree_teras.fit(train_data[['Waktu']], train_data['teras'])

decision_tree_dapur = DecisionTree()
decision_tree_dapur.fit(train_data[['Waktu']], train_data['dapur'])

decision_tree_toilet = DecisionTree()
decision_tree_toilet.fit(train_data[['Waktu']], train_data['toilet'])

decision_tree_ruangtamu = DecisionTree()
decision_tree_ruangtamu.fit(train_data[['Waktu']], train_data['ruangtamu'])

# Mengabaikan data yang tidak diketahui pada setiap kolom target
test_data_kamar = test_data[test_data['kamar'] != 'unknown']
test_data_kamar2 = test_data[test_data['kamar2'] != 'unknown']
test_data_teras = test_data[test_data['teras'] != 'unknown']
test_data_dapur = test_data[test_data['dapur'] != 'unknown']
test_data_toilet = test_data[test_data['toilet'] != 'unknown']
test_data_ruangtamu = test_data[test_data['ruangtamu'] != 'unknown']

# Melakukan prediksi pada data uji yang telah difilter
predictions_kamar = decision_tree_kamar.predict(test_data_kamar[['Waktu']])
predictions_kamar2 = decision_tree_kamar2.predict(test_data_kamar2[['Waktu']])
predictions_teras = decision_tree_teras.predict(test_data_teras[['Waktu']])
predictions_dapur = decision_tree_dapur.predict(test_data_dapur[['Waktu']])
predictions_toilet = decision_tree_toilet.predict(test_data_toilet[['Waktu']])
predictions_ruangtamu = decision_tree_ruangtamu.predict(
    test_data_ruangtamu[['Waktu']])

# Menghitung akurasi pada data uji
accuracy_kamar = calculate_accuracy(predictions_kamar, test_data['kamar'])
accuracy_kamar2 = calculate_accuracy(predictions_kamar2, test_data['kamar2'])
accuracy_teras = calculate_accuracy(predictions_teras, test_data['teras'])
accuracy_dapur = calculate_accuracy(predictions_dapur, test_data['dapur'])
accuracy_toilet = calculate_accuracy(predictions_toilet, test_data['toilet'])
accuracy_ruangtamu = calculate_accuracy(
    predictions_ruangtamu, test_data['ruangtamu'])

# akurasi dataframe
accuracies = {
    'Ruangan': ['Kamar', 'Kamar2', 'Teras', 'Dapur', 'Toilet', 'RuangTamu'],
    'Accuracy': [
        accuracy_kamar, accuracy_kamar2, accuracy_teras,
        accuracy_dapur, accuracy_toilet, accuracy_ruangtamu
    ]
}

accuracy_df = pd.DataFrame(accuracies)
print(accuracy_df)
print()

# recall, precision dan cm
for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
    decision_tree = DecisionTree()
    decision_tree.fit(train_data[['Waktu']], train_data[column])
    y_test = test_data[column]
    y_pred = test_data.apply(
        lambda x: decision_tree.predict_instance(x, decision_tree.tree), axis=1)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels)

    print("Confusion Matrix ({}) :".format(column))
    print(cm)
    print()

    recall_values = []
    precision_values = []
    for i, label in enumerate(labels):
        r = recall(cm, i)
        p = precision(cm, i)
        recall_values.append(r)
        precision_values.append(p)

        print("Label: {} ({})".format(label, column))
        print("Recall: {:.2f}".format(r))
        print("Precision: {:.2f}".format(p))
        print()

# Mencetak rules untuk setiap ruangan
rules_data_kamar = decision_tree_kamar.print_rules()
print("Rules Kamar:")
print(rules_data_kamar)
print()

rules_data_kamar2 = decision_tree_kamar2.print_rules()

print("Rules Kamar2:")
print(rules_data_kamar2)
print()

rules_data_teras = decision_tree_teras.print_rules()

print("Rules Teras:")
print(rules_data_teras)
print()

rules_data_dapur = decision_tree_dapur.print_rules()
print("Rules Dapur:")
print(rules_data_dapur)
print()

rules_data_toilet = decision_tree_toilet.print_rules()

print("Rules Toilet:")
print(rules_data_toilet)
print()

rules_data_ruangtamu = decision_tree_ruangtamu.print_rules()
print("Rules RuangTamu:")
print(rules_data_ruangtamu)
print()


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

decision_tree_kamar = DecisionTree()
decision_tree_kamar.fit(train_data[['Waktu']], train_data['kamar'])

decision_tree_kamar2 = DecisionTree()
decision_tree_kamar2.fit(train_data[['Waktu']], train_data['kamar2'])

decision_tree_teras = DecisionTree()
decision_tree_teras.fit(train_data[['Waktu']], train_data['teras'])

decision_tree_dapur = DecisionTree()
decision_tree_dapur.fit(train_data[['Waktu']], train_data['dapur'])

decision_tree_toilet = DecisionTree()
decision_tree_toilet.fit(train_data[['Waktu']], train_data['toilet'])

decision_tree_ruangtamu = DecisionTree()
decision_tree_ruangtamu.fit(train_data[['Waktu']], train_data['ruangtamu'])


@app.route('/api/accuracy')
def get_accuracy_all_rooms():
    accuracies = {
        'kamar': calculate_accuracy(decision_tree_kamar.predict(test_data_kamar[['Waktu']]), test_data_kamar['kamar']),
        'kamar2': calculate_accuracy(decision_tree_kamar2.predict(test_data_kamar2[['Waktu']]), test_data_kamar2['kamar2']),
        'teras': calculate_accuracy(decision_tree_teras.predict(test_data_teras[['Waktu']]), test_data_teras['teras']),
        'dapur': calculate_accuracy(decision_tree_dapur.predict(test_data_dapur[['Waktu']]), test_data_dapur['dapur']),
        'toilet': calculate_accuracy(decision_tree_toilet.predict(test_data_toilet[['Waktu']]), test_data_toilet['toilet']),
        'ruangtamu': calculate_accuracy(decision_tree_ruangtamu.predict(test_data_ruangtamu[['Waktu']]), test_data_ruangtamu['ruangtamu']),
    }
    return jsonify(accuracies)

# helper


def convert_to_python_int(data):
    if isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, dict):
        return {key: convert_to_python_int(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_int(item) for item in data]
    else:
        return data


@app.route('/api/rules')
def get_rules_all_rooms():
    rules = {
        'kamar': convert_to_python_int(decision_tree_kamar.print_rules()),
        'kamar2': convert_to_python_int(decision_tree_kamar2.print_rules()),
        'teras': convert_to_python_int(decision_tree_teras.print_rules()),
        'dapur': convert_to_python_int(decision_tree_dapur.print_rules()),
        'toilet': convert_to_python_int(decision_tree_toilet.print_rules()),
        'ruangtamu': convert_to_python_int(decision_tree_ruangtamu.print_rules()),
    }
    return jsonify(rules)


# Endpoint to display accuracy, precision, recall, and decision tree rules for each room
@app.route('/api/result')
def show_result():
    result = {}
    for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
        decision_tree = DecisionTree()
        decision_tree.fit(train_data[['Waktu']], train_data[column])
        y_test = test_data[column]
        y_pred = test_data.apply(
            lambda x: decision_tree.predict_instance(x, decision_tree.tree), axis=1)
        labels = np.unique(y_test)
        cm = confusion_matrix(y_test, y_pred, labels)

        print("Confusion Matrix ({}) :".format(column))
        print(cm)
        print()

        recall_values = []
        precision_values = []
        for i, label in enumerate(labels):
            r = recall(cm, i)
            p = precision(cm, i)
            recall_values.append(r)
            precision_values.append(p)

            print("Label: {} ({})".format(label, column))
            print("Recall: {:.2f}".format(r))
            print("Precision: {:.2f}".format(p))
            print()

        accuracy = calculate_accuracy(y_pred, y_test)
        precision_value_avg = np.mean(precision_values)
        recall_value_avg = np.mean(recall_values)

        result[column] = {
            'accuracy': "{:.2f}".format(accuracy),
            'precision on': "{:.2f}".format(precision_values[0]),
            'precision off': "{:.2f}".format(precision_values[1]),
            'recall on': "{:.2f}".format(recall_values[0]),
            'recall off': "{:.2f}".format(recall_values[1]),
            'precision_avg': "{:.2f}".format(precision_value_avg),
            'recall_avg': "{:.2f}".format(recall_value_avg)
        }

    return jsonify(result=result)


@app.route('/api/classify')
def classify_new_data():
    new_data = generate_new_data()
    result = {}
    for column in ['kamar', 'kamar2', 'teras', 'dapur', 'toilet', 'ruangtamu']:
        decision_tree = None
        if column == 'kamar':
            decision_tree = decision_tree_kamar
        elif column == 'kamar2':
            decision_tree = decision_tree_kamar2
        elif column == 'teras':
            decision_tree = decision_tree_teras
        elif column == 'dapur':
            decision_tree = decision_tree_dapur
        elif column == 'toilet':
            decision_tree = decision_tree_toilet
        elif column == 'ruangtamu':
            decision_tree = decision_tree_ruangtamu

        if decision_tree:
            predictions = decision_tree.predict(new_data[['Waktu']])
            result[column] = [1 if p == 1 else 2 for p in predictions]

    return jsonify(result=result)


@app.route('/api/rules/kamar')
def get_rules_kamar():
    rules = {
        'kamar': convert_to_python_int(decision_tree_kamar.print_rules())
    }
    return jsonify(rules=rules)


@app.route('/api/rules/kamar2')
def get_rules_kamar2():
    rules = {
        'kamar2': convert_to_python_int(decision_tree_kamar2.print_rules())
    }
    return jsonify(rules=rules)


@app.route('/api/rules/teras')
def get_rules_teras():
    rules = {
        'teras': convert_to_python_int(decision_tree_teras.print_rules())
    }
    return jsonify(rules=rules)


@app.route('/api/rules/dapur')
def get_rules_dapur():
    rules = {
        'dapur': convert_to_python_int(decision_tree_dapur.print_rules())
    }
    return jsonify(rules=rules)


@app.route('/api/rules/toilet')
def get_rules_toilet():
    rules = {
        'toilet': convert_to_python_int(decision_tree_toilet.print_rules())
    }
    return jsonify(rules=rules)


@app.route('/api/rules/ruangtamu')
def get_rules_ruangtamu():
    rules = {
        'ruangtamu': convert_to_python_int(decision_tree_ruangtamu.print_rules())
    }
    return jsonify(rules=rules)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
