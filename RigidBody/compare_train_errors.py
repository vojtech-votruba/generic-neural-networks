import pandas as pd
import matplotlib.pyplot as plt

df_impl = pd.read_csv("data/errors_implicit.csv")
df_soft = pd.read_csv("data/errors_soft.csv")
df_witho = pd.read_csv("data/errors_without.csv")

plt.plot(df_impl.train_mov, label="Implicit")
plt.plot(df_soft.train_mov, label="Soft")
plt.plot(df_witho.train_mov, label="Without Jacobi loss")

plt.ylim((0,0.01))

plt.legend()

plt.show()

plt.plot(df_impl.val_mov, label="Implicit")
plt.plot(df_soft.val_mov, label="Soft")
plt.plot(df_witho.val_mov, label="Without Jacobi loss")

plt.ylim((0,0.01))

plt.legend()

plt.show()
