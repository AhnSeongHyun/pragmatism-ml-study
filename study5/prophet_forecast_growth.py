import pandas as pd
import numpy as np
from fbprophet import Prophet

df = pd.read_csv('./examples/example_wp_R.csv')
df['y'] = np.log(df['y'])
df['cap'] = 8.5

m = Prophet(growth='logistic')
m.fit(df)

future = m.make_future_dataframe(periods=2)
future['cap'] = 8.5

forecast = m.predict(future)
print(forecast)

m.plot(forecast)
from matplotlib import pyplot as plt
plt.show()