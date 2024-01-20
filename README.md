# FastTag-Fraud-Detection
> The dataset was generated to provide a realistic example for developing and evaluating fraud detection models without relying on sensitive real-world data. It's intended for students, researchers, and practitioners to practice data analysis and machine learning techniques in a safe environment.

**Take Away from this dataset**
The decision to drop one feature in the presence of a strong positive correlation depends on the specific context and goals of your analysis. Here are a few considerations:

1. Redundancy:
>If two features have a strong positive correlation, it may suggest that they are conveying similar information. Redundant features might not provide additional insights and could potentially lead to overfitting in predictive modeling.

2. Multicollinearity:
> High correlation between features can lead to multicollinearity in regression models. Multicollinearity can make it challenging to interpret the individual contributions of each feature to the dependent variable. In such cases, dropping one of the highly correlated features might improve model interpretability.

3. Dimensionality Reduction:
> If you are working on a predictive modeling task, having fewer features can simplify the model and reduce the risk of overfitting. Dimensionality reduction techniques such as Principal Component Analysis (PCA) can also be considered.

4. Domain Knowledge:
> Consider domain knowledge. If both features are theoretically important and contribute valuable information, it might not be appropriate to drop one based solely on correlation. In some cases, features might be correlated, but each may still offer unique insights.

5. Model Performance:
> Evaluate the impact of dropping one feature on the performance of your models. Train models both with and without the feature, and assess their performance using metrics like accuracy, precision, recall, etc.

6. Data Exploration:
> Explore the data further to understand the relationship between the features. Are there specific patterns or trends that provide insights into why the features are correlated? Visualization and additional analysis can help in making informed decisions.

**In conclusion, while strong positive correlation might suggest redundancy, the decision to drop a feature should be based on a combination of statistical analysis, domain knowledge, and the goals of your analysis or modeling task. It's always a good idea to experiment with different scenarios, assess the impact on model performance, and validate decisions based on the specific requirements of your project.**

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Generate a DataFrame with 500 data points
np.random.seed(42)  # Setting seed for reproducibility
feature1 = np.arange(1, 501)
feature2 = 3 * feature1 + np.random.normal(0, 10, 500)  # Linear relationship with some noise
df = pd.DataFrame({'feature1': feature1, 'feature2': feature2})

# Calculate the regression line
coefficients = np.polyfit(df['feature1'], df['feature2'], 1)
regression_line = np.polyval(coefficients, df['feature1'])

# Plot a scatter plot with the regression line
plt.scatter(df['feature1'], df['feature2'], color='blue', label='Data points')
plt.plot(df['feature1'], regression_line, color='red', label='Regression Line')
plt.title('Scatter Plot with Regression Line (500 Data Points)')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.grid(True)
plt.show()
```

<p align="center">
    <img src="https://www.shutterstock.com/image-vector/types-correlation-scatter-plot-positive-260nw-2140738797.jpg" alt="Correlation Scatter Plot" />
</p>

### Another way to plot regression line
```python
# Scatter plot with regression line between 'Transaction_Amount' and 'Amount_paid'
sns.regplot(x='Transaction_Amount', y='Amount_paid', data=data)
plt.show()
```

<p align="center">
    <img src="https://www.kaggleusercontent.com/kf/159413024/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0.._bJ1ea4NjwXtG20043kxgA.dDAEtNVw1RYj8fv0Xdm8a02mNFi_jrLxm9Rd4mF1usGg8BS4w74MNlDwJ0E1Wx2CpO0TdbwKBTUGlK8jyUtsyUqkG4UhhrUwU7i0Om6o6zwwamj_CNzneGUCgHIW4mqdM6zCZdvXasD8vaaRqBCfvXf2L2lg-AX-ZhaT9dh5AE4FZXsOCc5zkvwsxMHUDIAnIe4tOOKDzIRmyAMzFTHaNgye-qbmwOUDUCMuISWhdUI32_4GkSpio7-oo9m_F7V12_2yAku0-c_HqzNwB3j0hG8mzXiMfTXoYm7lN6h8w4o6Z24CN4ZE1HHh_AINw5J-gFkkcNgrakPfHXpNiF-Yz-QcSBT9MIw7ESSs-PUlhbzx7L9Wr5fClaejTopvVCk9HJIX-7pFOEEvSMUVLharMyY22I_Kez8itUdwM9SfX-9Td2Jor7wO3SKlLQ3imJJ5yTH9VyeRe9kDIInAzo9rBEZDFvjEGeC5Zd7Aou5GiRFKdZSR7lG9QMPgt75FH-l2y1mwQB00Kp96el2CyZgDiuA1fTvBykr9H4PzZiXAbu8KXeaSP3rD64iIGrBdC2rWTkG84eygXoABvlNTfON1R_ifyLtSwPKf05l-fz_6tyZZeDSXT1RQVLdCgO-vW11x-EXnkmF8Tkyzcr0_JzqAYVe9vGDfQoB6phhnaOTvOtDTrfhUWo3RCmcxhxCM3IIc.a0jUyrnj2PmNftFyRUqhxA/__results___files/__results___17_0.png" alt="Correlation Scatter Plot" width="400" height="300" />
</p>

## Visulaize the Metrices
```python
import matplotlib.pyplot as plt

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

# Define custom colors for bars and text
bar_colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
plt.bar(metrics, values, color=bar_colors)

# Adding values on top of each bar
for i, v in enumerate(values):
    plt.text(i, v + 0.05, str(round(v, 2)), ha='center', va='bottom')

plt.ylabel('Score')
plt.title('Model Metrics')
plt.ylim(0, 1.4)
plt.show()
```
<p align="center">
    <img src="https://www.kaggleusercontent.com/kf/159711698/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Qb3jzDUGuvwN9AqVXKDdIw.6o1LTHu6itZlCGofOw-7GdUxZ9dDjzOs3EwtSKY8k44IEKKbftaZbCl82QhuDFby5CfidY3AnDDVJnCTeVbofIvMWiIFYIrx8c3om1xB4ViCybIvbIpCZHhPwBEu1OJapyytxJ0FGztihYJJirdWV9tdJdMZKc4rO8TWVoGlPmNF434RB7x9nfSrKP8cYqPjCU5sXHNgc5G4yRC7BdjP_Usrg7jSM7ECEwpew5O-ascjtnnfxJ_9FViSfLx5lxVQWMwREWN-K1gFBqVjD6IEBG-OYaVaZFXZhsMu-PEiVqH7ULbXm-brZ67jz1bUsBCoix9wZ7sL4Y9wuW6yZTZ5KfZFpKyQ6kjvvUihbNgGIcY5o-zRObiAFPjp1h9dXXYFGmrQ7Yo0xtom0BgmKQudGr3SGK4zhmSJPGI_Mo-xjDJ2YUd3kKZgpI9s6nETDwje4sKnvs7XpSNlQHKePrYn3NZyBqjp7_9bnDRH5u3y7mmdOSgwNeLmbFlKBKSbVkITCIPiG2CK3gSXnYvDtiq74-3uFjeDDB8kyVkveozBW9Jd97r0QPP6SbhERvA7BfjGi700IMAww9BDR1M6CPqAiEgkGUu7Khzr8rvUuvqru7OYPrnzvbolW4Dw9im7Ct7v7G_Y8d8U6uwPISPd0HixFZSOEGxkBHjw1Kqp1PyU01c.e-3VIegNUXTA_qyTbJBjkA/__results___files/__results___35_0.png" alt="Correlation Scatter Plot" width="400" height="300" />
</p>

### Counter the total number of Categorical Value present in dataset
```python
sns.countplot(x='Fraud_indicator', data=data)
plt.show()
```

<p align="center">
    <img src="https://www.kaggleusercontent.com/kf/159711698/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ijLEroIcw0NWKkK1HQyHtg.PgwQx2RON_fG3AD6KNxgZx-n3TYA6I4g0IUUiBOmhuWPQcv7XlkDwU38gRLlGdkb2xGdV49oF_vwQeYXgoIiYwi9v2dZdhEh7j59732sagGr9eNVKOVo4x9BmO2IyaR4yRLTvDqyHRtPbVep8p9bwops8pZULuQ0Xfi7ONB_tbQNIA32A8b_lqNNr8XjVurymjgHUHkAH-5dPXaJl3giU9CektNs_NBy1e7FYehAqQRS4PGsBwgLYh6e4sW9lJx7PazH-I-d1Oluqhquck1xnIlaj3zQd7NiYXK6SHep_YvUSsje0_08554z4lKMCl_FwsFwfXCgPQrqBdwiRh6BIoOsL2mC5YOcmPxkXcX1Z1-ZOKb6VhE_42aQexKFoqJ01yhLzdYVSG-ed_ssj7nqPBcCvSkFn7_LzDbP9XAR1x_x7EmAyTERDQ8loQWBECk_ZWOc_hne1PhEVeCbtIdX1jZt0S3hMfvCKfN2FuQK3lk8RL3uRuPi3VWQc0d9OFNm1PZAUq8YbIrA9bbaTOkvGfgOCQn8vnJUDCwcwWtH1_rfqf1QRYh45nOVhJdSU2a0G9qBQW38aYL08NPXuIr2bCIO1nIMdIH_rKJIWxZ7TTUSeQh6ZKyxXTwXn-sO2uPEQvHPt_S5yMf1BSvOwyOH-HgZ-gHBOTRtIsWvkXjLdew.gB88i88htQtgUYsJ7KLlSw/__results___files/__results___11_0.png" alt="Correlation Scatter Plot" width="400" height="300" />
</p>


### boxplot that visualizes the distribution of the variable 'Vehicle_Speed' for different levels of the categorical variable 'Fraud_indicator'

```python
sns.boxplot(x='Fraud_indicator', y='Vehicle_Speed', data=data)
plt.show()
```

<p align="center">
    <img src="https://www.kaggleusercontent.com/kf/159711698/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ijLEroIcw0NWKkK1HQyHtg.PgwQx2RON_fG3AD6KNxgZx-n3TYA6I4g0IUUiBOmhuWPQcv7XlkDwU38gRLlGdkb2xGdV49oF_vwQeYXgoIiYwi9v2dZdhEh7j59732sagGr9eNVKOVo4x9BmO2IyaR4yRLTvDqyHRtPbVep8p9bwops8pZULuQ0Xfi7ONB_tbQNIA32A8b_lqNNr8XjVurymjgHUHkAH-5dPXaJl3giU9CektNs_NBy1e7FYehAqQRS4PGsBwgLYh6e4sW9lJx7PazH-I-d1Oluqhquck1xnIlaj3zQd7NiYXK6SHep_YvUSsje0_08554z4lKMCl_FwsFwfXCgPQrqBdwiRh6BIoOsL2mC5YOcmPxkXcX1Z1-ZOKb6VhE_42aQexKFoqJ01yhLzdYVSG-ed_ssj7nqPBcCvSkFn7_LzDbP9XAR1x_x7EmAyTERDQ8loQWBECk_ZWOc_hne1PhEVeCbtIdX1jZt0S3hMfvCKfN2FuQK3lk8RL3uRuPi3VWQc0d9OFNm1PZAUq8YbIrA9bbaTOkvGfgOCQn8vnJUDCwcwWtH1_rfqf1QRYh45nOVhJdSU2a0G9qBQW38aYL08NPXuIr2bCIO1nIMdIH_rKJIWxZ7TTUSeQh6ZKyxXTwXn-sO2uPEQvHPt_S5yMf1BSvOwyOH-HgZ-gHBOTRtIsWvkXjLdew.gB88i88htQtgUYsJ7KLlSw/__results___files/__results___14_0.png" alt="Correlation Scatter Plot" width="400" height="300" />
</p>

### Scatter plot with regression line between 'Transaction_Amount' and 'Amount_paid'
```python
sns.regplot(x='Transaction_Amount', y='Amount_paid', data=data)
plt.show()
```

<p align="center">
    <img src="https://www.kaggleusercontent.com/kf/159711698/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ijLEroIcw0NWKkK1HQyHtg.PgwQx2RON_fG3AD6KNxgZx-n3TYA6I4g0IUUiBOmhuWPQcv7XlkDwU38gRLlGdkb2xGdV49oF_vwQeYXgoIiYwi9v2dZdhEh7j59732sagGr9eNVKOVo4x9BmO2IyaR4yRLTvDqyHRtPbVep8p9bwops8pZULuQ0Xfi7ONB_tbQNIA32A8b_lqNNr8XjVurymjgHUHkAH-5dPXaJl3giU9CektNs_NBy1e7FYehAqQRS4PGsBwgLYh6e4sW9lJx7PazH-I-d1Oluqhquck1xnIlaj3zQd7NiYXK6SHep_YvUSsje0_08554z4lKMCl_FwsFwfXCgPQrqBdwiRh6BIoOsL2mC5YOcmPxkXcX1Z1-ZOKb6VhE_42aQexKFoqJ01yhLzdYVSG-ed_ssj7nqPBcCvSkFn7_LzDbP9XAR1x_x7EmAyTERDQ8loQWBECk_ZWOc_hne1PhEVeCbtIdX1jZt0S3hMfvCKfN2FuQK3lk8RL3uRuPi3VWQc0d9OFNm1PZAUq8YbIrA9bbaTOkvGfgOCQn8vnJUDCwcwWtH1_rfqf1QRYh45nOVhJdSU2a0G9qBQW38aYL08NPXuIr2bCIO1nIMdIH_rKJIWxZ7TTUSeQh6ZKyxXTwXn-sO2uPEQvHPt_S5yMf1BSvOwyOH-HgZ-gHBOTRtIsWvkXjLdew.gB88i88htQtgUYsJ7KLlSw/__results___files/__results___18_0.png" alt="Correlation Scatter Plot" width="400" height="300" />
</p>







