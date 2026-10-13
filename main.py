
# Практическая работа: случайный лес
# Windows 11 + VS Code

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==========================================

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

print("=" * 60)
print("1. ИНФОРМАЦИЯ О ДАННЫХ")
print("=" * 60)
print("Количество объектов:", X.shape[0])
print("Количество признаков:", X.shape[1])
print("Пропущенные значения:", X.isnull().sum().sum())


# Делим данные на обучающую и тестовую части.
# Тестовые данные не участвуют в подборе параметров.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 2. МОДЕЛЬ С ПАРАМЕТРАМИ ПО УМОЛЧАНИЮ
# ==========================================

print("\n" + "=" * 60)
print("2. МОДЕЛЬ ПО УМОЛЧАНИЮ")
print("=" * 60)

default_model = RandomForestClassifier(random_state=42)
default_model.fit(X_train, y_train)

default_predictions = default_model.predict(X_test)
default_accuracy = accuracy_score(y_test, default_predictions)

print("Точность модели по умолчанию:",
      round(default_accuracy, 4))


# ==========================================
# 3. ПОДБОР ГИПЕРПАРАМЕТРОВ
# ==========================================

print("\n" + "=" * 60)
print("3. ПОДБОР ГИПЕРПАРАМЕТРОВ")
print("=" * 60)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Лучшие параметры:")
print(grid_search.best_params_)

print("Лучшая средняя точность на кросс-валидации:",
      round(grid_search.best_score_, 4))

best_predictions = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_predictions)

print("Точность лучшей модели на тестовых данных:",
      round(best_accuracy, 4))


# ==========================================
# 4. СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ
# ==========================================

print("\n" + "=" * 60)
print("4. СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

comparison = pd.DataFrame({
    "Модель": [
        "Параметры по умолчанию",
        "Подобранные параметры"
    ],
    "Точность на тесте": [
        default_accuracy,
        best_accuracy
    ]
})

print(comparison.to_string(index=False))


# ==========================================
# 5. ВАЖНОСТЬ ПРИЗНАКОВ
# ==========================================

print("\n" + "=" * 60)
print("5. ВАЖНОСТЬ ПРИЗНАКОВ")
print("=" * 60)

importance = pd.Series(
    best_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\n10 наиболее важных признаков:")
print(importance.head(10).round(4).to_string())

print("\n10 наименее важных признаков:")
print(importance.tail(10).sort_values().round(4).to_string())


# График важности десяти наиболее значимых признаков

importance.head(10).sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)

plt.title("10 наиболее важных признаков случайного леса")
plt.xlabel("Важность признака")
plt.ylabel("Признак")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()


# ==========================================
# 6. ЭКСПЕРИМЕНТ: УДАЛЕНИЕ СЛАБЫХ ПРИЗНАКОВ
# ==========================================

print("\n" + "=" * 60)
print("6. УДАЛЕНИЕ НАИМЕНЕЕ ВАЖНЫХ ПРИЗНАКОВ")
print("=" * 60)

# Важность рассчитана по обучающей выборке.
# Удаляем 5, 10 и 15 наименее важных признаков.

results = []

results.append({
    "Вариант": "Все признаки",
    "Количество признаков": X_train.shape[1],
    "Точность": best_accuracy
})

for remove_count in [5, 10, 15]:

    # Оставляем наиболее важные признаки
    selected_features = importance.index[:-remove_count]

    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]

    # Подбираем параметры заново для сокращённого набора
    reduced_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    reduced_search.fit(X_train_reduced, y_train)

    reduced_model = reduced_search.best_estimator_
    reduced_predictions = reduced_model.predict(X_test_reduced)

    reduced_accuracy = accuracy_score(
        y_test,
        reduced_predictions
    )

    results.append({
        "Вариант": f"Удалено {remove_count}",
        "Количество признаков": len(selected_features),
        "Точность": reduced_accuracy
    })

    print(f"\nУдалено признаков: {remove_count}")
    print("Осталось признаков:", len(selected_features))
    print("Точность на тесте:", round(reduced_accuracy, 4))
    print("Лучшие параметры:", reduced_search.best_params_)

    print("Удалённые признаки:")
    print(", ".join(importance.tail(remove_count).index))


# ==========================================
# 7. ИТОГОВАЯ ТАБЛИЦА
# ==========================================

print("\n" + "=" * 60)
print("7. ИТОГОВОЕ СРАВНЕНИЕ")
print("=" * 60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

results_df.to_csv(
    "feature_experiment_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nРезультаты сохранены в feature_experiment_results.csv")
print("График сохранён в feature_importance.png")
print("\nПрактическая работа завершена.")


# ... здесь у тебя обучение модели Random Forest ...

# Получаем важность признаков
importances = model.feature_importances_
features = X.columns   # или как у тебя называются признаки

# Строим и сохраняем график
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Top 10 features (Random Forest)")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")  # ← вот эта строка сохраняет картинку
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 1. Загружаем датасет
df = pd.read_csv("winequality-red.csv", sep=";")

print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

print("\nПропущенные значения:")
print(df.isnull().sum())


# 2. Создаем три класса качества вина
# 0 — низкое качество, 1 — среднее, 2 — высокое
def classify_quality(quality):
    if quality <= 5:
        return 0
    elif quality == 6:
        return 1
    else:
        return 2


df["quality_class"] = df["quality"].apply(classify_quality)

# quality — исходная оценка, quality_class — наша цель.
# Исходную оценку убираем из признаков, чтобы не было утечки данных.
X = df.drop(columns=["quality", "quality_class"])
y = df["quality_class"]

print("\nКоличество объектов каждого класса:")
print(y.value_counts().sort_index())


# 3. Делим данные на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nОбучающих объектов:", len(X_train))
print("Тестовых объектов:", len(X_test))


# 4. Создаем модель с параметрами по умолчанию
default_model = RandomForestClassifier(random_state=42)
default_model.fit(X_train, y_train)

default_predictions = default_model.predict(X_test)

default_accuracy = accuracy_score(y_test, default_predictions)
default_f1 = f1_score(y_test, default_predictions, average="macro")

print("\n--- Модель по умолчанию ---")
print("Accuracy:", round(default_accuracy, 4))
print("F1-macro:", round(default_f1, 4))


# 5. Настраиваем гиперпараметры с кросс-валидацией
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("\nНачинается подбор параметров...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n--- Лучшие параметры ---")
print(grid_search.best_params_)
print("Лучший F1-macro на кросс-валидации:",
      round(grid_search.best_score_, 4))


# 6. Проверяем лучшую модель на тестовых данных
best_predictions = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, best_predictions)
best_f1 = f1_score(y_test, best_predictions, average="macro")

print("\n--- Сравнение моделей ---")
print("Модель по умолчанию:")
print("Accuracy:", round(default_accuracy, 4))
print("F1-macro:", round(default_f1, 4))

print("\nНастроенная модель:")
print("Accuracy:", round(best_accuracy, 4))
print("F1-macro:", round(best_f1, 4))

print("\nОтчет по настроенной модели:")
print(classification_report(
    y_test,
    best_predictions,
    labels=[0, 1, 2],
    target_names=["Низкое", "Среднее", "Высокое"],
    zero_division=0
))


# 7. Анализируем важность признаков
importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n--- Важность признаков ---")
print(importance.round(4))

plt.figure(figsize=(10, 6))
importance.sort_values().plot(kind="barh")
plt.title("Важность признаков случайного леса")
plt.xlabel("Важность")
plt.ylabel("Признак")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()


# 8. Проверяем удаление наименее важных признаков
# Удаляем сначала один, затем два и три самых слабых признака.
least_important = importance.sort_values().index.tolist()

results = []

results.append({
    "Удалено признаков": 0,
    "Удаленные признаки": "Нет",
    "Accuracy": best_accuracy,
    "F1-macro": best_f1
})

print("\n--- Эксперимент с удалением признаков ---")

for number_to_remove in [1, 2, 3]:
    removed_features = least_important[:number_to_remove]

    kept_features = [
        column for column in X.columns
        if column not in removed_features
    ]

    reduced_model = RandomForestClassifier(
        **grid_search.best_params_,
        random_state=42
    )

    reduced_model.fit(X_train[kept_features], y_train)

    reduced_predictions = reduced_model.predict(
        X_test[kept_features]
    )

    reduced_accuracy = accuracy_score(
        y_test, reduced_predictions
    )

    reduced_f1 = f1_score(
        y_test, reduced_predictions, average="macro"
    )

    results.append({
        "Удалено признаков": number_to_remove,
        "Удаленные признаки": ", ".join(removed_features),
        "Accuracy": reduced_accuracy,
        "F1-macro": reduced_f1
    })

    print(f"\nУдалено признаков: {number_to_remove}")
    print("Какие признаки:", removed_features)
    print("Accuracy:", round(reduced_accuracy, 4))
    print("F1-macro:", round(reduced_f1, 4))


# 9. Сохраняем таблицу экспериментов
results_df = pd.DataFrame(results)
results_df.to_csv("feature_selection_results.csv", index=False)

print("\n--- Итоговая таблица ---")
print(results_df.round(4).to_string(index=False))

print("\nФайлы feature_importance.png и "
      "feature_selection_results.csv сохранены.")


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import numpy as np

results = []

# Сортируем признаки по важности
sorted_idx = importances.argsort()[::-1]
sorted_features = features[sorted_idx]

for n_remove in range(0, 4):  # удаляем 0, 1, 2, 3 самых важных
    if n_remove == 0:
        selected_features = sorted_features
        removed = "Нет"
    else:
        selected_features = sorted_features[n_remove:]
        removed = ", ".join(sorted_features[:n_remove])

    X_sel = X[selected_features]

    acc = cross_val_score(best_model, X_sel, y, cv=5, scoring="accuracy").mean()
    f1 = cross_val_score(best_model, X_sel, y, cv=5, scoring=make_scorer(f1_score, average="macro")).mean()

    results.append({
        "Удалено признаков": n_remove,
        "Удалённые признаки": removed,
        "Accuracy": round(acc, 4),
        "F1-macro": round(f1, 4)
    })

# Сохраняем результаты
results_df = pd.DataFrame(results)
results_df.to_csv("feature_selection_results.csv", index=False, encoding="utf-8-sig")

print("\nРезультаты эксперимента:")
print(results_df)
print("\nФайл сохранён: feature_selection_results.csv")