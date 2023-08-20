from flask import Flask, jsonify
from flask_cors import CORS
import datetime
import pytz
import pandas as pd
import numpy as np


# data from drive
data_url = 'https://drive.google.com/file/d/1utoPLUMN1JCsl6VaRwYE0WscoAh7dmds/view?usp=share_link'
data_url = 'https://drive.google.com/uc?id=' + data_url.split('/')[-2]
data = pd.read_csv(data_url, encoding='unicode_escape')


# train data 100%
train_data = data.copy()

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

    def print_rules(self, node=None, condition='', rules_data=None):
        if node is None:
            node = self.tree

        if rules_data is None:
            rules_data = []

        if 'label' in node:
            rules_data.append({'Condition': condition, 'Label': node['label']})
            return

        attribute = node['attribute']
        for value, child_node in node['children'].items():
            child_condition = f"{attribute} = {value}"
            self.print_rules(child_node, condition +
                             child_condition, rules_data)

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


timeloc = pytz.timezone('Asia/Jakarta')


def generate_new_data():
    current_time = datetime.datetime.now(tz=timeloc).time()
    new_data = pd.DataFrame({'Waktu': [current_time]})
    new_data['Waktu'] = new_data['Waktu'].apply(
        lambda x: x.hour * 60 + x.minute)
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


@app.route('/api/status/kamar')
def classify_kamar():
    new_data = generate_new_data()
    decision_tree = decision_tree_kamar
    if decision_tree:
        predictions = decision_tree.predict(new_data[['Waktu']])
        return str(1 if predictions[0] == 1 else 2)


@app.route('/api/status/kamar2')
def classify_kamar2():
    new_data = generate_new_data()
    decision_tree = decision_tree_kamar2
    if decision_tree:
        predictions = decision_tree.predict(new_data[['Waktu']])
        return str(1 if predictions[0] == 1 else 2)


@app.route('/api/status/teras')
def classify_teras():
    new_data = generate_new_data()
    decision_tree = decision_tree_teras
    if decision_tree:
        predictions = decision_tree.predict(new_data[['Waktu']])
        return str(1 if predictions[0] == 1 else 2)


@app.route('/api/status/dapur')
def classify_dapur():
    new_data = generate_new_data()
    decision_tree = decision_tree_dapur
    if decision_tree:
        predictions = decision_tree.predict(new_data[['Waktu']])
        return str(1 if predictions[0] == 1 else 2)


@app.route('/api/status/toilet')
def classify_toilet():
    new_data = generate_new_data()
    decision_tree = decision_tree_toilet
    if decision_tree:
        predictions = decision_tree.predict(new_data[['Waktu']])
        return str(1 if predictions[0] == 1 else 2)


@app.route('/api/status/ruangtamu')
def classify_ruangtamu():
    new_data = generate_new_data()
    decision_tree = decision_tree_ruangtamu
    if decision_tree:
        predictions = decision_tree.predict(new_data[['Waktu']])
        return str(1 if predictions[0] == 1 else 2)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
