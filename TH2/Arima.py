import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# 1. Load và chuẩn hóa
df = pd.read_csv(r'D:\doAn\TH2\Code\china_disease_data.csv')
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df = df.set_index('Date').sort_index()
ts = df['Reported_Cases'].resample('MS').sum()

# 2. Kiểm tra tính dừng
adf_stat, p_value, *_ = adfuller(ts)
print(f'ADF Statistic: {adf_stat:.4f}, p-value: {p_value:.4f}')
d = 0 if p_value < 0.05 else 1

# 3. Chia train (2021) và chuẩn bị thử các order
train = ts['2021-01-01':'2021-12-01']
candidate_orders = [
    (1, d, 1),
    (1, d, 0),
    (0, d, 1),
    (0, d, 0)
]

# 4. Thử lần lượt các ARIMA(order) đến khi fit thành công
res = None
for order in candidate_orders:
    try:
        print(f"Thử ARIMA{order}...")
        model = ARIMA(train, order=order)
        res = model.fit()
        print(f"► ARIMA{order} hội tụ với AIC = {res.aic:.2f}")
        break
    except Exception as e:
        print(f"  – ARIMA{order} không hội tụ: {e}")

if res is None:
    raise RuntimeError("Không có ARIMA nào hội tụ. Hãy kiểm tra lại dữ liệu hoặc giảm bớt tham số.")

# 5. In-sample prediction và forecast 2 bước
pred_in = res.predict(start=train.index[0], end=train.index[-1])
fc      = res.get_forecast(steps=2)
pred_fc = fc.predicted_mean
pred_ci = fc.conf_int()

# 6. Đánh giá MAE in-sample
mae = mean_absolute_error(train, pred_in)
print(f"MAE (in-sample): {mae:.4f}")

# 7. Vẽ kết quả
plt.figure(figsize=(9,5))
plt.plot(ts,                   label='Thực tế', marker='o')
plt.plot(pred_in,              label=f'Fit {res.model.order}', linestyle='--')
plt.plot(pred_fc.index, pred_fc, label='Forecast (Jan–Feb 2022)', marker='o')
plt.fill_between(pred_ci.index,
                 pred_ci.iloc[:,0], pred_ci.iloc[:,1],
                 alpha=0.2, label='95% CI')
plt.title('ARIMA Fit & 2-Step Forecast')
plt.xlabel('Month')
plt.ylabel('Reported Cases')
plt.legend()
plt.tight_layout()
plt.show()
