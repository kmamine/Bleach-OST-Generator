
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

combined_df = pd.read_csv('soundtrack_data.csv')

# Descriptive statistics
print(combined_df.describe(include='all'))

# Check for missing values
print(combined_df.isnull().sum())

# Distribution of numerical features
numerical_features = ['likes', 'comments', 'views', 'length']
for feature in numerical_features:
  plt.figure()
  sns.histplot(combined_df[feature], kde=True)
  plt.title(f'Distribution of {feature}')
  plt.show()

# Explore categorical features
categorical_features = ['release_date']
for feature in categorical_features:
  plt.figure()
  sns.countplot(x=feature, data=combined_df)
  plt.title(f'Count of {feature}')
  plt.xticks(rotation=45)
  plt.show()

# Correlation between numerical features
correlation_matrix = combined_df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
