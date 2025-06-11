import pandas as pd
import math

# =================== KONVERSI NILAI =====================
def konversi_nilai(data_rows):
    hasil = []
    for row in data_rows:
        data = []
        for key, val in row.items():
            if key == 'GENDER':
                val = 1 if str(val).strip().upper() == 'M' else 0
            elif isinstance(val, str):
                val = val.strip().upper()
                if val == 'YES':
                    val = 1
                elif val == 'NO':
                    val = 0
                else:
                    try:
                        val = int(val)
                    except:
                        pass
            elif isinstance(val, bool):
                val = int(val)
            data.append(val)
        hasil.append(data)
    return hasil

# =================== ENTROPY DAN INFORMATION GAIN =====================
def entropy(group):
    total = len(group)
    if total == 0:
        return 0
    count = {}
    for row in group:
        label = row[-1]
        if label not in count:
            count[label] = 0
        count[label] += 1
    ent = 0.0
    for c in count.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def information_gain(parent, groups):
    total_len = len(parent)
    parent_entropy = entropy(parent)
    weighted_entropy = 0.0
    for group in groups:
        weighted_entropy += (len(group) / total_len) * entropy(group)
    return parent_entropy - weighted_entropy

# ================== TREE BUILDING USING ENTROPY ===================
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_split(dataset):
    best_gain = -1
    best_index = None
    best_value = None
    best_groups = None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gain = information_gain(dataset, groups)
            if gain > best_gain:
                best_index = index
                best_value = row[index]
                best_gain = gain
                best_groups = groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

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

# =================== DATA LOADING =====================
data_latih_df = pd.read_excel("data_latih.xlsx")
data_uji_df = pd.read_excel("data_uji.xlsx")

data_latih_dict = data_latih_df.to_dict(orient='records')
data_uji_dict = data_uji_df.to_dict(orient='records')

data_latih_processed = konversi_nilai(data_latih_dict)
data_uji_processed = konversi_nilai(data_uji_dict)

X_train = [row[:-1] for row in data_latih_processed]
y_train = [row[-1] for row in data_latih_processed]

dataset_train = [x + [y] for x, y in zip(X_train, y_train)]
tree = build_tree(dataset_train, max_depth=5, min_size=10)

# ======================= EVALUASI ========================
def evaluate(y_true, y_pred, label="LATIH"):
    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    acc = (TP + TN) / len(y_true)
    prec = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) != 0 else 0

    print(f"\n========== EVALUASI DATA {label.upper()} ==========")
    print(f"Akurasi  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Presisi  : {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall   : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score : {f1:.4f} ({f1*100:.2f}%)")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Model memprediksi data {label.lower()} dengan {'baik' if acc > 0.8 else 'perlu perbaikan'}.")

# ======================= EVALUASI LATIH & UJI ========================
y_train_pred = [predict(tree, row) for row in X_train]
evaluate(y_train, y_train_pred, label="latih")

if 'LUNG_CANCER' in data_uji_df.columns:
    y_test_true = [row[-1] for row in data_uji_processed]
    X_test = [row[:-1] for row in data_uji_processed]
    y_test_pred = [predict(tree, row) for row in X_test]
    evaluate(y_test_true, y_test_pred, label="uji")
else:
    X_test = data_uji_processed
    y_test_pred = [predict(tree, row) for row in X_test]
    yes_prediksi = y_test_pred.count(1)
    no_prediksi = y_test_pred.count(0)
    print("\n========== PREDIKSI DATA UJI ==========")
    print(f"Total data uji: {len(y_test_pred)}")
    print(f"Prediksi 'YES' kanker: {yes_prediksi} ({(yes_prediksi/len(y_test_pred))*100:.2f}%)")
    print(f"Prediksi 'NO'  kanker: {no_prediksi} ({(no_prediksi/len(y_test_pred))*100:.2f}%)")

# ========== INPUT MANUAL PREDIKSI ==========
print("\n========== INPUT MANUAL PREDIKSI ==========")
input_data = {}
fitur = list(data_latih_df.columns)
fitur.remove("LUNG_CANCER")

for col in fitur:
    while True:
        val = input(f"Masukkan nilai untuk {col} (M/F, YES/NO, atau angka): ").strip().upper()
        if col == 'GENDER' and val in ['M', 'F']:
            input_data[col] = val
            break
        elif val in ['YES', 'NO']:
            input_data[col] = val
            break
        else:
            try:
                input_data[col] = int(val)
                break
            except:
                print("Input tidak valid. Ulangi.")

row_input = konversi_nilai([input_data])[0]
hasil_prediksi = predict(tree, row_input)

print("\nHasil Prediksi: ", "YES (Berisiko Kanker Paru-Paru)" if hasil_prediksi == 1 else "NO (Tidak Berisiko Kanker Paru-Paru)")
