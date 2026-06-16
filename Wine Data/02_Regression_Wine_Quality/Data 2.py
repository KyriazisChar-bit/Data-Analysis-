#==================================================================
#                             Ergasia 2h 
# =================================================================
#κάνετ copy-paste το σκρθπτ αυτό κάτω απο το σκριπτ Data 1 για να τρεξει 

print("\n--- Δευτερη ΕΡγασια (2Η)--------")


import statsmodels.api as sm
import seaborn as sns
#===============================
#   Ερωτημα 1
#===============================
# Predictors 
predictors = sarxeio.columns.difference(['quality', 'wine_type'])

results_table = []
regression_table = []
significant_predictors = []

for col in predictors:

    X = sm.add_constant(sarxeio[col])
    y = sarxeio['quality']

    model = sm.OLS(y, X).fit()

    # Extract metrics
    coef = model.params[col]
    std_err = model.bse[col]
    t_stat = model.tvalues[col]
    p_value = model.pvalues[col]

    # Residual Standard Error
    rse = (model.mse_resid) ** 0.5

    r_squared = model.rsquared
    f_stat = model.fvalue

    # Αποθήκευση σε πίνακα
    results_table.append({
        "Variable": col,
        "Coefficient": coef,
        "Std Error": std_err,
        "t-statistic": t_stat,
        "p-value": p_value,
        "RSE": rse,
        "R^2": r_squared,
        "F-statistic": f_stat
    })
    # b0 και b1
    b0 = model.params['const']
    b1 = model.params[col]

    regression_table.append({
        "Variable": col,
        "Model": f"quality = {b0:.4f} + {b1:.4f} * {col}"
    })

    print(f"\n--- {col} ---")
    print(f"coef = {coef:.4f}, std_err = {std_err:.4f}, t = {t_stat:.4f}, p = {p_value:.6f}")
    print(f"RSE = {rse:.4f}, R^2 = {r_squared:.4f}, F = {f_stat:.4f}")

    
    print(f"\n--- {col} ---")
    print(f"Μοντέλο: quality = {b0:.4f} + {b1:.4f} * {col}")
    print(f"p-value = {model.pvalues[col]:.6f}")
    # Έλεγχος σημαντικότητας
    if p_value < 0.05:
        significant_predictors.append(col)
        print(f"Στατιστικά σημαντική εξάρτηση με: {col}")

        # Plot (όπως ζήτησες)
        plt.figure(figsize=(6,4))
        sns.regplot(x=sarxeio[col], y=sarxeio['quality'], line_kws={"color": "red"})
        plt.title(f"Quality vs {col}")
        plt.xlabel(col)
        plt.ylabel("quality")
        #plt.show()

# Δημιουργία πίνακα αποτελεσμάτων
results_df = pd.DataFrame(results_table)

print("\n================ FINAL TABLE ================\n")
print(results_df)

df_models = pd.DataFrame(regression_table)
print(df_models)

#===============================
#   Ερωτημα 2
#===============================

import statsmodels.api as sm
import pandas as pd

# Όλες οι υποψήφιες προβλεπτικές μεταβλητές
all_predictors = sarxeio.columns.difference(['quality', 'wine_type'])

X_all = sm.add_constant(sarxeio[all_predictors])
y = sarxeio['quality']

full_model = sm.OLS(y, X_all).fit()

# Βασικές μετρικές προσαρμογής
r2 = full_model.rsquared
adj_r2 = full_model.rsquared_adj
f_stat = full_model.fvalue
f_pvalue = full_model.f_pvalue

print("\n--- MODEL FIT METRICS ---")
print(f"R^2 = {r2:.6f}")
print(f"Adjusted R^2 = {adj_r2:.6f}")
print(f"F-statistic = {f_stat:.6f}")
print(f"Prob(F-statistic) = {f_pvalue:.6e}")

# Εμφάνιση πλήρους πίνακα αποτελεσμάτων
print(full_model.summary())

pvalues = full_model.pvalues

significant_multi = pvalues[pvalues < 0.05].index.tolist()

# Αφαιρούμε το const
significant_multi = [var for var in significant_multi if var != 'const']

print("\nΜεταβλητές για τις οποίες απορρίπτεται η H0: βj = 0")
for var in significant_multi:
    print(f"- {var} (p-value = {full_model.pvalues[var]:.6f})")

multi_table = pd.DataFrame({
    "Variable": full_model.params.index,
    "Coefficient": full_model.params.values,
    "Std Error": full_model.bse.values,
    "t-statistic": full_model.tvalues.values,
    "p-value": full_model.pvalues.values
})


multi_table_no_const = multi_table[multi_table["Variable"] != "const"]
print(multi_table_no_const)

print("\n--- SIGNIFICANT VARIABLES ---")

for var in full_model.pvalues.index:
    if var != "const":
        if full_model.pvalues[var] < 0.05:
            print(f"{var}: SIGNIFICANT (p={full_model.pvalues[var]:.6f})")
        else:
            print(f"{var}: NOT significant (p={full_model.pvalues[var]:.6f})")

# =============================
# Ερωτημα 3
# =============================
print("\n================ forward selection ================\n")
remaining = list(predictors)
selected = []

current_rss = float("inf")
final_model = None

while remaining:

    best_rss = float("inf")
    best_col = None
    best_model = None

    for col in remaining:
        X = sarxeio[selected + [col]]
        X = sm.add_constant(X)
        y = sarxeio["quality"]

        model = sm.OLS(y, X).fit()
        rss = sum(model.resid ** 2)

        if rss < best_rss:
            best_rss = rss
            best_col = col
            best_model = model

    # stopping rule:
    # κρατάμε τη μεταβλητή μόνο αν βελτιώνει το RSS
    # και αν είναι στατιστικά σημαντική
    if best_col is not None and best_rss < current_rss and best_model.pvalues[best_col] < 0.05:
        remaining.remove(best_col)
        selected.append(best_col)
        current_rss = best_rss
        final_model = best_model
        print(f"Added: {best_col} | RSS = {best_rss:.4f} | p-value = {best_model.pvalues[best_col]:.6f}")
    else:
        break

print("\nSelected variables:", selected)

if final_model is not None:
    print(f"Final R^2: {final_model.rsquared:.6f}")
    print(f"Final Adjusted R^2: {final_model.rsquared_adj:.6f}")
    print(f"Final RSS: {current_rss:.6f}")

# =======================================
# ΕΡΩΤΗΜΑ 4
# =======================================

print("\n--- ΕΡΩΤΗΜΑ 4 ---")

poly_results = []

for col in ["residual sugar", "chlorides", "alcohol"]:
    print(f"\n==================== {col.upper()} ====================")

    y = sarxeio["quality"]
    x = sarxeio[col]

    # -------- 1. Linear model --------
    X1 = pd.DataFrame({
        col: x
    })
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(y, X1).fit()

    # -------- 2. Quadratic model --------
    X2 = pd.DataFrame({
        col: x,
        f"{col}_2": x**2
    })
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()

    # -------- 3. Cubic model --------
    X3 = pd.DataFrame({
        col: x,
        f"{col}_2": x**2,
        f"{col}_3": x**3
    })
    X3 = sm.add_constant(X3)
    model3 = sm.OLS(y, X3).fit()

    # -------- Εκτύπωση βασικών μετρικών --------
    print("\n--- Linear model ---")
    print(f"R^2 = {model1.rsquared:.6f}")
    print(f"Adj R^2 = {model1.rsquared_adj:.6f}")
    print(model1.summary())

    print("\n--- Quadratic model ---")
    print(f"R^2 = {model2.rsquared:.6f}")
    print(f"Adj R^2 = {model2.rsquared_adj:.6f}")
    print(model2.summary())

    print("\n--- Cubic model ---")
    print(f"R^2 = {model3.rsquared:.6f}")
    print(f"Adj R^2 = {model3.rsquared_adj:.6f}")
    print(model3.summary())

    # -------- Επιλογή καταλληλότερου μοντέλου --------
    candidates = {
        "Linear": model1,
        "Quadratic": model2,
        "Cubic": model3
    }

    best_name = max(candidates, key=lambda k: candidates[k].rsquared_adj)
    best_model = candidates[best_name]

    print(f"\n>>> Best model for {col}: {best_name}")
    print(f"Adj R^2 = {best_model.rsquared_adj:.6f}")

    
   

    poly_results.append({
        "Variable": col,
        "R^2 (Linear X)": model1.rsquared,
        "R^2 (Quadratic X^2)": model2.rsquared,
        "R^2 (Cubic X^3)": model3.rsquared,
        "Adj R^2 (Linear)": model1.rsquared_adj,
        "Adj R^2 (Quadratic)": model2.rsquared_adj,
        "Adj R^2 (Cubic)": model3.rsquared_adj,
    })




    # -------- Plot observed data + fitted curve --------
    x_sorted = x.sort_values()
    if best_name == "Linear":
        X_plot = pd.DataFrame({col: x_sorted})
    elif best_name == "Quadratic":
        X_plot = pd.DataFrame({
            col: x_sorted,
            f"{col}_2": x_sorted**2
        })
    else:
        X_plot = pd.DataFrame({
            col: x_sorted,
            f"{col}_2": x_sorted**2,
            f"{col}_3": x_sorted**3
        })

    X_plot = sm.add_constant(X_plot, has_constant='add')
    y_pred = best_model.predict(X_plot)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=sarxeio[col], y=sarxeio["quality"], alpha=0.5)
    plt.plot(x_sorted, y_pred, color="red", linewidth=2)
    plt.title(f"Quality vs {col} ({best_name} fit)")
    plt.xlabel(col)
    plt.ylabel("quality")
    plt.show()

    # -------- Residual plot --------
    residuals = best_model.resid
    fitted = best_model.fittedvalues

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"Residual plot for {col} ({best_name} fit)")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.show()

# -------- Συγκεντρωτικός πίνακας --------
poly_results_df = pd.DataFrame(poly_results)

print("\n=========== ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ ===========")
print(poly_results_df.to_string(index=False))

poly_results_df = pd.DataFrame(poly_results)
