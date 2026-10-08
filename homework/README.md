# CV2026
### Homework1

homework1-1 -Selfimode

Selfimode(https://github.com/user-attachments/assets/4c77cbf0-4094-4e8c-aec6-b424d5570648)


homework1-2 -Yolo

Yolo(https://github.com/user-attachments/assets/a35fc457-c548-4f87-b5f0-19e65ded6d7f)

homework2 - Classification 
code(Simple neural network)
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential([
    Dense(units=2, activation='relu', input_shape=(2,)),
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
acc = np.mean(y_pred.flatten() == y_test)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=pyplot.cm.coolwarm, edgecolors="k")
plt.title(f"Simple neural network (1 hidden layer, 2 neurons)\nAccuracy: {acc:.2f}")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
<img width="742" height="587" alt="image" src="https://github.com/user-attachments/assets/297e83c3-baca-46e4-846f-a1a6f4b927f7" />

code(Appropriate neural network)
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential([
    Dense(units=2, activation='relu', input_shape=(2,)),
    Dense(units=2, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
acc = np.mean(y_pred.flatten() == y_test)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=pyplot.cm.coolwarm, edgecolors="k")
plt.title(f"Appropriate neural network (2 hidden layers, 2 neurons)\nAccuracy: {acc:.2f}")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
<img width="747" height="617" alt="image" src="https://github.com/user-attachments/assets/0dfe6e54-d378-4f3e-8c36-24ef0c8a9fa7" />

code(Complex neural network)
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential([
    Dense(units=16, activation='tanh', input_shape=(2,)),
    Dense(units=16, activation='tanh'),
    Dense(units=16, activation='tanh'),
    Dense(units=16, activation='tanh'),
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, batch_size=10, verbose=0)

y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
acc = np.mean(y_pred.flatten() == y_test)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=pyplot.cm.coolwarm, edgecolors="k")
plt.title(f"Complex neural network (4 hidden layers, 16 neurons)\nAccuracy: {acc:.2f}")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

<img width="761" height="631" alt="image" src="https://github.com/user-attachments/assets/ec8abf5f-3eb2-404d-961c-6c1be0bda461" />

