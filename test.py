import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from tensorflow.keras import Input, Model
# Hàm đọc dữ liệu từ file JSON
def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    features = np.array([item['features'] for item in data])
    labels = np.array([item['label'] for item in data])
    return features, labels

# Hàm tạo triplets
def create_triplets(features, labels):
    triplets = []
    for i in range(len(features)):
        anchor = features[i]
        label_anchor = labels[i]
        
        # Tìm positive sample (cùng lớp với anchor)
        positive_indices = np.where(labels == label_anchor)[0]
        positive_index = np.random.choice(positive_indices)
        positive = features[positive_index]
        
        # Tìm negative sample (khác lớp với anchor)
        negative_indices = np.where(labels != label_anchor)[0]
        negative_index = np.random.choice(negative_indices)
        negative = features[negative_index]
        
        triplets.append([anchor, positive, negative])
    
    return np.array(triplets)

# Hàm xây dựng mô hình MLP
def build_mlp(input_shape):
    anchor_input = Input(shape=input_shape, name='anchor')
    positive_input = Input(shape=input_shape, name='positive')
    negative_input = Input(shape=input_shape, name='negative')

    # Các lớp MLP
    hidden_layer_1 = layers.Dense(128, activation='relu')(anchor_input)
    hidden_layer_1=layers.Dropout(0.3)(hidden_layer_1) 
    hidden_layer_2 = layers.Dense(64, activation='relu')(hidden_layer_1)
    hidden_layer_2=layers.Dropout(0.3)(hidden_layer_2) 
    embedding_layer = layers.Dense(32, activation='relu')(hidden_layer_2)

    # Tạo mô hình với ba đầu vào
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=embedding_layer)
    return model

# Hàm Triplet Loss
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    
    return loss

# Đọc dữ liệu từ JSON
train_features, train_labels = load_data_from_json('train.json')
valid_features, valid_labels = load_data_from_json('valid.json')
test_features, test_labels = load_data_from_json('test.json')

# Tạo triplets
train_triplets = create_triplets(train_features, train_labels)
valid_triplets = create_triplets(valid_features, valid_labels)
test_triplets = create_triplets(test_features, test_labels)

# Xây dựng mô hình MLP
input_shape = (train_features.shape[1],)
print("input shape : ",input_shape)
mlp_model = build_mlp(input_shape)

# Compile mô hình
mlp_model.compile(optimizer='adam', loss=triplet_loss)

# Dữ liệu đầu vào
train_anchor, train_positive, train_negative = train_triplets[:, 0], train_triplets[:, 1], train_triplets[:, 2]
valid_anchor, valid_positive, valid_negative = valid_triplets[:, 0], valid_triplets[:, 1], valid_triplets[:, 2]
test_anchor, test_positive, test_negative = test_triplets[:, 0], test_triplets[:, 1], test_triplets[:, 2]

# Dummy y_true (Triplet Loss không cần nhãn)
y_train = np.zeros((len(train_triplets),))
y_valid = np.zeros((len(valid_triplets),))
y_test = np.zeros((len(test_triplets),))

print('y_train: ', y_train)
# Lưu embedding trước khi train
extractor_before = models.Model(inputs=mlp_model.inputs, outputs=mlp_model.layers[-1].output)
features_before_test = test_anchor

print("truoc khi train : ", features_before_test)
# Huấn luyện mô hình
mlp_model.fit(
    [train_anchor, train_positive, train_negative], y_train,
    validation_data=([valid_anchor, valid_positive, valid_negative], y_valid),
    epochs=10,
    batch_size=32
)

# Lấy embedding sau khi train cho tập test
extractor_after = models.Model(inputs=mlp_model.inputs, outputs=mlp_model.layers[-1].output)
features_after_test = extractor_after.predict([test_anchor, test_positive, test_negative])

# Đánh giá mô hình trên tập test
test_loss = mlp_model.evaluate([test_anchor, test_positive, test_negative], y_test)
print(f"Test Loss: {test_loss}")

# Giảm chiều dữ liệu xuống 2D bằng PCA cho tập test
pca = PCA(n_components=2)
features_before_2d_test = pca.fit_transform(features_before_test)
features_after_2d_test = pca.fit_transform(features_after_test)

# Vẽ biểu đồ trước và sau khi train trên tập test
plt.figure(figsize=(12, 5))

# Trước khi train (tập test)
plt.subplot(1, 2, 1)
plt.scatter(features_before_2d_test[:, 0], features_before_2d_test[:, 1], c=test_labels, cmap='jet', alpha=0.6)
plt.title("Embedding trước khi train (test)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Sau khi train (tập test)
plt.subplot(1, 2, 2)
plt.scatter(features_after_2d_test[:, 0], features_after_2d_test[:, 1], c=test_labels, cmap='jet', alpha=0.6)
plt.title("Embedding sau khi train (test)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.colorbar()
plt.show()