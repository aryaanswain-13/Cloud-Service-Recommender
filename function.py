import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from scipy.optimize import minimize
from scipy.stats import entropy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# MAIN FUNCTION
# ==================================================

def run_csp_system(book_file, cap_file, rating_file, service_name):

    print("\n" + "="*100)
    print(f"{service_name.upper()} SERVICE - UNIFIED CSP SYSTEM (FINAL STABLE VERSION)")
    print("="*100)

    df = pd.read_excel(book_file).iloc[1:301].fillna(0)

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # ================= DNN =================
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(le.classes_), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nDNN Accuracy: {accuracy:.2%}")

    # ================= RF =================
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, np.argmax(y_train, axis=1))

    # ================= USER INPUT =================
    print(f"\nAvailable {service_name} Services:")
    print(", ".join(X.columns))

    user_services = input("\nEnter required services: ").lower()
    valid_services = set(col.lower() for col in X.columns)

    user_list = [s.strip() for s in user_services.split(',')]
    valid_inputs = [s for s in user_list if s in valid_services]

    if not valid_inputs:
        print("No valid input.")
        return None, None, None

    user_input = {col: 1 if col.lower() in valid_inputs else 0 for col in X.columns}
    user_df = pd.DataFrame([user_input], columns=X.columns)

    # ================= HYBRID + UNCERTAINTY =================
    mc_runs = 50

    dnn_preds = np.array([
        model(user_df, training=True).numpy()[0]
        for _ in range(mc_runs)
    ])

    dnn_mean = dnn_preds.mean(axis=0)
    dnn_std = dnn_preds.std(axis=0)

    tree_preds = np.array([
        tree.predict_proba(user_df)[0]
        for tree in rf_model.estimators_
    ])

    rf_mean = tree_preds.mean(axis=0)
    rf_std = tree_preds.std(axis=0)

    alpha = 0.5
    hybrid_probs = alpha * dnn_mean + (1 - alpha) * rf_mean
    hybrid_uncertainty = np.sqrt(alpha**2 * dnn_std**2 + (1-alpha)**2 * rf_std**2)

    suitability_scores = dict(zip(le.classes_, hybrid_probs * 100))
    uncertainty_scores = dict(zip(le.classes_, hybrid_uncertainty * 100))

    entropy_val = entropy(hybrid_probs)
    entropy_scores = {provider: entropy_val for provider in le.classes_}

    # ================= CAPABILITY =================
    cap_df = pd.read_excel(cap_file)
    cap_df.columns = cap_df.columns.str.strip()

    cap_features = cap_df.columns[1:]
    cap_scores = cap_df.groupby("Service Provider")[cap_features].mean().mean(axis=1) * 20

    cap_stability = cap_df.groupby("Service Provider")[cap_features].mean().std(axis=1)
    cap_stability = 100 * (1 - cap_stability / cap_stability.max())

    # ================= USER RATINGS =================
    user_df2 = pd.read_excel(rating_file)
    user_df2.columns = user_df2.columns.str.strip()
    user_df2.rename(columns={user_df2.columns[0]: "Service Provider"}, inplace=True)

    rating_cols = user_df2.columns[1:]
    user_df2[rating_cols] = user_df2[rating_cols].apply(pd.to_numeric, errors="coerce")

    user_scores = user_df2.groupby("Service Provider")[rating_cols].mean().mean(axis=1) * 20

    user_stability = user_df2.groupby("Service Provider")[rating_cols].mean().std(axis=1)
    user_stability = 100 * (1 - user_stability / user_stability.max())

    # ================= FUSION =================
    fusion_df = pd.DataFrame({
        "Suitability %": pd.Series(suitability_scores),
        "Capability %": cap_scores,
        "User Rating %": user_scores,
        "Uncertainty": pd.Series(uncertainty_scores),
        "Entropy": pd.Series(entropy_scores),
        "Cap Stability %": cap_stability,
        "User Stability %": user_stability
    })

    # better NaN handling
    fusion_df = fusion_df.fillna(fusion_df.mean())

    fusion_df["Overall Stability %"] = (
        0.5 * fusion_df["Cap Stability %"] +
        0.5 * fusion_df["User Stability %"]
    )

    # ================= NORMALIZATION (CRITICAL FIX) =================
    norm_df = fusion_df.copy()
    for col in ["Suitability %", "Capability %", "User Rating %"]:
        norm_df[col] = (norm_df[col] - norm_df[col].min()) / (
            norm_df[col].max() - norm_df[col].min() + 1e-6
        )

    # ================= OPTIMIZATION =================
    def objective(weights):
        w1, w2, w3 = weights

        scores = (
            w1 * norm_df["Suitability %"] +
            w2 * norm_df["Capability %"] +
            w3 * norm_df["User Rating %"]
        )

        sorted_scores = scores.sort_values(ascending=False)

        # FIXED GAP
        gap = (sorted_scores.iloc[0] - sorted_scores.iloc[1]) / (sorted_scores.std() + 1e-6)

        unc = fusion_df["Uncertainty"].mean() / 100
        stab = fusion_df["Overall Stability %"].mean() / 100

        # STRONGER PENALTY
        penalty = ((w1 - 1/3)**2 + (w2 - 1/3)**2 + (w3 - 1/3)**2)

        return -(0.5 * gap - 0.3 * unc + 0.2 * stab - 0.4 * penalty)

    result = minimize(
        objective,
        [0.4,0.3,0.3],
        method='SLSQP',
        bounds=[(0.1,0.7)]*3,
        constraints={'type':'eq','fun':lambda w: sum(w)-1}
    )

    W_SUIT, W_CAP, W_USER = result.x
    print(f"\nOptimized Weights → {W_SUIT:.3f}, {W_CAP:.3f}, {W_USER:.3f}")

    # ================= FINAL SCORE =================
    fusion_df["Unified Score %"] = (
        W_SUIT*fusion_df["Suitability %"] +
        W_CAP*fusion_df["Capability %"] +
        W_USER*fusion_df["User Rating %"]
    )

    fusion_df["Final Score %"] = (
        fusion_df["Unified Score %"] *
        (1 - 0.05*fusion_df["Uncertainty"]/100) *
        (1 + 0.05*fusion_df["Overall Stability %"]/100)
    )

    fusion_df = fusion_df.sort_values("Final Score %", ascending=False)
    fusion_df["Final Rank"] = range(1, len(fusion_df)+1)

    # FIXED CONFIDENCE
    fusion_df["Confidence %"] = 100 * np.exp(-fusion_df["Uncertainty"] / 20)

    print("\nFINAL RANKING:")
    print(fusion_df[[
        "Final Rank","Suitability %","Capability %",
        "User Rating %","Final Score %","Confidence %"
    ]].round(2))

    best_provider = fusion_df.index[0]
    best_score = fusion_df.iloc[0]["Final Score %"]

    print(f"\n🏆 BEST {service_name.upper()} PROVIDER: {best_provider} ({best_score:.2f}%)")

    return fusion_df, (W_SUIT, W_CAP, W_USER), accuracy


# ==================================================
# RUN + STORE RESULTS
# ==================================================

results = {}

for name, files in {
    "Computing": ("Book1_Computing.xlsx","sp-cap-cs.xlsx","User Ratings CS.xlsx"),
    "Storage": ("Book2_Storage.xlsx","sp-cap-storage.xlsx","User Ratings storage.xlsx"),
    "Network": ("Book3_Network.xlsx","sp-cap-network.xlsx","User Ratings network.xlsx"),
    "Management": ("Book4_Management.xlsx","sp-cap-management.xlsx","User Ratings Management.xlsx"),
    "Security": ("Book5_Security.xlsx","sp-cap-security.xlsx","User Ratings security.xlsx")
}.items():

    df, weights, acc = run_csp_system(*files, name)
    results[name] = {"df": df, "weights": weights, "accuracy": acc}


# ==================================================
# GRAPH FUNCTIONS
# ==================================================

def plot_weights(results):
    domains, suit, cap, user = [], [], [], []

    for d, data in results.items():
        domains.append(d)
        w = data["weights"]
        suit.append(w[0])
        cap.append(w[1])
        user.append(w[2])

    x = range(len(domains))

    plt.plot(x, suit, marker='o', label="Suitability")
    plt.plot(x, cap, marker='o', label="Capability")
    plt.plot(x, user, marker='o', label="User Rating")

    plt.xticks(x, domains)
    plt.title("Optimized Weights Across Domains")
    plt.legend()
    plt.show()


def plot_top_scores(results):
    domains, scores = [], []

    for d, data in results.items():
        domains.append(d)
        scores.append(data["df"].iloc[0]["Final Score %"])

    plt.bar(domains, scores)
    plt.title("Top CSP Scores Across Domains")
    plt.ylabel("Score")
    plt.show()


def plot_uncertainty(df, title):
    plt.scatter(df["Uncertainty"], df["Confidence %"])
    plt.xlabel("Uncertainty")
    plt.ylabel("Confidence")
    plt.title(title)
    plt.show()


# ==================================================
# CALL GRAPHS
# ==================================================

plot_weights(results)
plot_top_scores(results)

plot_uncertainty(results["Computing"]["df"], "Computing Uncertainty vs Confidence")
