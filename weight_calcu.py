
weights = {"bayes": 0.7660, "ada": 0.8946, "rf": 0.9453, "dt": 0.9014, "lgs": 0.8406, "gbc": 0.9407, "tf_idf": 0.9325, "URLGledac": 0.9453}

sum = 0
for key, value in weights.items():
    sum += value
print("sum:", sum)

w_bayes_new = weights["bayes"] / sum
w_ada_new = weights["ada"] / sum
w_rf_new = weights["rf"] / sum
w_dt_new = weights["dt"] / sum
w_lgs_new = weights["lgs"] / sum
w_gbc_new = weights["gbc"] / sum
w_tf_idf_new = weights["tf_idf"] / sum
w_URLGledac_new = weights["URLGledac"] / sum

print("w_bayes_new:", w_bayes_new)
print("w_ada_new:", w_ada_new)
print("w_rf_new:", w_rf_new)
print("w_dt_new:", w_dt_new)
print("w_lgs_new:", w_lgs_new)
print("w_gbc_new:", w_gbc_new)
print("w_tf_idf_new:", w_tf_idf_new)
print("w_URLGledac_new:", w_URLGledac_new) 

# {"bayes": 0.10688769814690781, "ada": 0.12483255190890824, "rf": 0.13190723375753519, "dt": 0.1257814244250949, "lgs": 0.11729738780977898, "gbc": 0.13126534940835008, "tf_idf": 0.13012112078588972, "URLGledac": 0.13190723375753519}