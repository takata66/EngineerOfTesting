import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageDraw
import torch.nn.functional as F
import numpy as np

# Загрузка данных из CSV
def load_data(train_csv, val_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Разделяем метки и признаки
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    X_val = val_df.drop('label', axis=1).values
    y_val = val_df['label'].values

    # Кодируем метки
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)

    # Преобразуем в тензоры
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    return X_train, y_train, X_val, y_val, label_encoder

# Функция для создания загрузчиков данных
def get_data_loaders(train_csv, val_csv, batch_size=32):
    X_train, y_train, X_val, y_val, label_encoder = load_data(train_csv, val_csv)

    # Создаем DataLoader для тренировочных и валидационных данных
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, label_encoder

# Определение модели
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        
        # Добавление скрытых слоев
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())  # Можно выбрать другую активацию, кроме пороговой
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, num_classes))  # Выходной слой
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Функция для обучения модели
def train_model(model, train_loader, val_loader, epochs, optimizer, criterion):
    for epoch in range(epochs):
        model.train()  # Включаем режим обучения
        train_loss = 0
        all_preds = []
        all_labels = []
        
        # Тренировка на тренировочной выборке
        for images, labels in train_loader:
            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # Обратное распространение ошибки
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

        train_accuracy = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds, average='weighted')
        train_recall = recall_score(all_labels, all_preds, average='weighted')
        
        print(f'Epoch {epoch+1}, Loss: {train_loss}, Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}')

        # Валидация на валидационной выборке
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.numpy())
                val_labels.extend(val_labels.numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        
        print(f'Validation - Loss: {val_loss}, Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}')

# Создание класса для рисования и распознавания
class PaintApp:
    def __init__(self, root, model, label_encoder):
        self.root = root
        self.model = model
        self.label_encoder = label_encoder

        # Создание холста для рисования
        self.canvas_size = 256
        self.image_size = 64
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        # Изображение для 64x64
        self.image = Image.new("L", (self.image_size, self.image_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Кнопки
        self.train_button = tk.Button(self.root, text="Train Network", command=self.start_training)
        self.train_button.pack(side=tk.LEFT)

        self.recognize_button = tk.Button(self.root, text="Recognize", command=self.recognize_image)
        self.recognize_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(self.root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        # Переменные для рисования
        self.old_x = None
        self.old_y = None

        # Связывание событий рисования
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def start_training(self):
        epochs = int(simpledialog.askstring("Input", "Enter number of epochs:"))
        learning_rate = float(simpledialog.askstring("Input", "Enter learning rate:"))

        # Обновляем скорость обучения в оптимизаторе
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Загружаем данные и запускаем обучение
        train_loader, val_loader, _ = get_data_loaders('train_data.csv', 'val_data.csv')

        self.train_model(train_loader, val_loader, epochs, optimizer)

    def train_model(self, train_loader, val_loader, epochs, optimizer):
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}/{epochs} completed')

        messagebox.showinfo("Training", "Training completed successfully")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.image_size, self.image_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        if self.old_x and self.old_y:
            # Рисуем на увеличенном холсте
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=8, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)

            # Рисуем на 64x64 изображении
            scaled_x1, scaled_y1 = self.old_x * (self.image_size / self.canvas_size), self.old_y * (self.image_size / self.canvas_size)
            scaled_x2, scaled_y2 = event.x * (self.image_size / self.canvas_size), event.y * (self.image_size / self.canvas_size)
            self.draw.line([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill=0, width=3)

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def recognize_image(self):
        # Преобразуем изображение в тензор для подачи в сеть
        img_data = np.array(self.image).reshape(-1) / 255.0  # Преобразуем в вектор
        img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)  # Добавляем размер батча

        # Подаем в модель и получаем результат
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1).numpy()[0]
            predicted_class = np.argmax(probabilities)
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        # Отображаем результат
        result_text = f"Predicted: {predicted_label}\nProbabilities:\n"
        for i, prob in enumerate(probabilities):
            label = self.label_encoder.inverse_transform([i])[0]
            result_text += f"{label}: {prob:.4f}\n"

        messagebox.showinfo("Recognition Result", result_text)


# Загрузка данных
train_loader, val_loader, label_encoder = get_data_loaders('train_data.csv', 'val_data.csv')

# Инициализация модели
input_size = 64 * 64
hidden_layers = [128, 64]
num_classes = len(label_encoder.classes_)

model = MLP(input_size, hidden_layers, num_classes)

# Запуск интерфейса
root = tk.Tk()
app = PaintApp(root, model, label_encoder)
root.mainloop()