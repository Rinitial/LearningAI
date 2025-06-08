import pandas as pd

# =================== DATA LOADING & PREPROCESSING =====================
def preprocess(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        elif df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.upper()
            if df[col].isin(['YES', 'NO']).all():
                df[col] = df[col].map({'YES': 1, 'NO': 0})
            elif df[col].isin(['M', 'F']).all():
                df[col] = df[col].map({'M': 1, 'F': 0})
    return df

data_latih = pd.read_excel("data_latih.xlsx")
data_uji = pd.read_excel("data_uji.xlsx")

data_latih = preprocess(data_latih)
data_uji = preprocess(data_uji)

X_train = data_latih.drop(columns='LUNG_CANCER').values.tolist()
y_train = data_latih['LUNG_CANCER'].values.tolist()
X_test = data_uji.drop(columns='LUNG_CANCER', errors='ignore').values.tolist()

# ======================= DECISION TREE ============================
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini_index(groups, classes):
    total = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0: continue
        score = 0.0
        labels = [row[-1] for row in group]
        for c in classes:
            p = labels.count(c) / size
            score += p * p
        gini += (1 - score) * (size / total)
    return gini

def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value: left.append(row)
        else: right.append(row)
    return left, right

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = None, None, float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if isinstance(node, dict):
        if row[node['index']] < node['value']:
            return predict(node['left'], row)
        else:
            return predict(node['right'], row)
    else:
        return node

# ======================= TRAINING & EVALUASI ========================
dataset_train = [x + [y] for x, y in zip(X_train, y_train)]
tree = build_tree(dataset_train, max_depth=5, min_size=10)

y_train_pred = [predict(tree, row) for row in X_train]

def evaluate(y_true, y_pred):
    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    acc = (TP + TN) / len(y_true)
    prec = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) != 0 else 0

    print("\n========== EVALUASI DATA LATIH ==========")
    print(f"Akurasi  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Presisi  : {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall   : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score : {f1:.4f} ({f1*100:.2f}%)")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Model mempelajari data latih dengan {'baik' if acc > 0.8 else 'kurang optimal'}.")

evaluate(y_train, y_train_pred)

# ========== PREDIKSI DATA UJI TANPA LABEL ===============
y_test_pred = [predict(tree, row) for row in X_test]
yes_prediksi = y_test_pred.count(1)
no_prediksi = y_test_pred.count(0)

print("\n========== PREDIKSI DATA UJI ==========")
print(f"Total data uji: {len(y_test_pred)}")
print(f"Prediksi 'YES' kanker: {yes_prediksi} ({(yes_prediksi/len(y_test_pred))*100:.2f}%)")
print(f"Prediksi 'NO'  kanker: {no_prediksi} ({(no_prediksi/len(y_test_pred))*100:.2f}%)")

print("\n========== INPUT MANUAL PREDIKSI ==========")
input_data = {}

for col in data_latih.columns:
    if col == 'LUNG_CANCER':
        continue
    user_input = input(f"Masukkan nilai untuk {col} (contoh: M/F, YES/NO, atau angka): ").strip().upper()
    
    # Deteksi dan konversi input numerik
    if user_input.isdigit():
        input_data[col] = int(user_input)
    else:
        input_data[col] = user_input

# Buat DataFrame dari input dan preprocess
df_input = pd.DataFrame([input_data])
df_input = preprocess(df_input)

# Ubah ke list untuk diprediksi
row_input = df_input.values[0].tolist()

# Prediksi hasil
hasil_prediksi = predict(tree, row_input)

print("\nHasil Prediksi: ", "YES (Berisiko Kanker Paru-Paru)" if hasil_prediksi == 1 else "NO (Tidak Berisiko Kanker Paru-Paru)")