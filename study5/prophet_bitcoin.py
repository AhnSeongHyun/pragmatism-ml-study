import pandas as pd
import numpy as np
from fbprophet import Prophet

df = pd.read_csv('./examples/korbit_btckrw2.csv')

print(df)
df['y'] = np.log(df['last'])
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=10)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
print(forecast)

m.plot(forecast)
from matplotlib import pyplot as plt
plt.show()