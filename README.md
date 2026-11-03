# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 3.11.2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files
from io import BytesIO

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

uploaded = files.upload()

file_name = next(iter(uploaded))
df = pd.read_csv(BytesIO(uploaded[file_name]))

df.columns = df.columns.str.strip().str.lower()

df['time period'] = pd.to_datetime(
    df['time period'], format='mixed', dayfirst=True, errors='coerce'
)

df['subscribers'] = pd.to_numeric(
    df['subscribers'], errors='coerce'
)

df = df.dropna(subset=['time period', 'subscribers'])
df = df.sort_values('time period')
df = df.set_index('time period')

df = df.groupby(level=0)['subscribers'].mean().to_frame()

X = df['subscribers']

plt.figure(figsize=(12, 6))
plt.plot(X, color='b')
plt.title('Netflix Subscribers Time Series')
plt.xlabel('Date')
plt.ylabel('Subscribers')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plot_acf(X, lags=min(20, len(X)//2 - 1), ax=plt.gca())
plt.title('ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=min(20, len(X)//2 - 1),
          ax=plt.gca(), method='ywm')
plt.title('PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 1, 1)).fit()

phi1 = arma11_model.arparams[0]
theta1 = arma11_model.maparams[0]

print("ARMA(1,1) parameters:")
print("phi1 =", phi1)
print("theta1 =", theta1)

ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

arma1 = ArmaProcess(ar1, ma1).generate_sample(nsample=500)

plt.figure(figsize=(10, 4))
plt.plot(arma1, color='g')
plt.title('Simulated ARMA(1,1)')
plt.xlabel('Time')
plt.ylabel('Values')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plot_acf(arma1, lags=30)
plt.title('Autocorrelation - ARMA(1,1)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(arma1, lags=30, method='ywm')
plt.title('Partial Autocorrelation - ARMA(1,1)')
plt.show()

arma22_model = ARIMA(X, order=(2, 1, 2)).fit()

phi1 = arma22_model.arparams[0]
phi2 = arma22_model.arparams[1]

theta1 = arma22_model.maparams[0]
theta2 = arma22_model.maparams[1]

print("\nARMA(2,2) parameters:")
print("phi1 =", phi1, "phi2 =", phi2)
print("theta1 =", theta1, "theta2 =", theta2)

ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

arma2 = ArmaProcess(ar2, ma2).generate_sample(nsample=500)

plt.figure(figsize=(10, 4))
plt.plot(arma2, color='r')
plt.title('Simulated ARMA(2,2)')
plt.xlabel('Time')
plt.ylabel('Values')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plot_acf(arma2, lags=30)
plt.title('Autocorrelation - ARMA(2,2)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(arma2, lags=30, method='ywm')
plt.title('Partial Autocorrelation - ARMA(2,2)')
plt.show()
```

OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img width="844" height="393" alt="image" src="https://github.com/user-attachments/assets/978692b5-1ead-4890-a029-8ff150494e7d" />


Partial Autocorrelation

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/2d72666d-746b-4786-b343-2c5d950acf61" />

Autocorrelation

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/29f3120b-8315-4a87-abe9-89ab443aa364" />



SIMULATED ARMA(2,2) PROCESS:

<img width="844" height="393" alt="image" src="https://github.com/user-attachments/assets/e45f5cb3-9744-4ddb-a996-ae125d5b8838" />

Partial Autocorrelation

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/ea0a1a99-abb0-4e45-9a02-be91416894d0" />


Autocorrelation

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/f4553bdd-d471-4766-ad30-473db99645b1" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
