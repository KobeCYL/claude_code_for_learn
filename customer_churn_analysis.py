#!/usr/bin/env python3
"""
Customer Churn Analysis and Feature Engineering Script

This script performs comprehensive data analysis and feature engineering
on the customer churn dataset to prepare it for machine learning modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerChurnAnalyzer:
    """
    A comprehensive class for analyzing and preprocessing customer churn data
    """

    def __init__(self, file_path):
        """Initialize the analyzer with the dataset"""
        self.file_path = file_path
        self.df = None
        self.numerical_features = []
        self.categorical_features = []
        self.target_column = 'Exited'

    def load_data(self):
        """Load and display basic information about the dataset"""
        print("Loading customer churn dataset...")
        self.df = pd.read_csv(self.file_path)

        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of customers: {len(self.df)}")
        print(f"Number of features: {len(self.df.columns)}")

        # Display basic info
        print("\nDataset Info:")
        print(self.df.info())

        print("\nFirst 5 rows:")
        print(self.df.head())

        return self.df

    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)

        # Basic statistics
        print("\n1. Basic Statistics:")
        print(self.df.describe())

        # Check for missing values
        print("\n2. Missing Values:")
        missing_data = self.df.isnull().sum()
        print(missing_data[missing_data > 0])

        # Target variable distribution
        print("\n3. Target Variable Distribution:")
        target_dist = self.df[self.target_column].value_counts()
        print(target_dist)
        print(f"\nChurn Rate: {target_dist[1] / len(self.df):.2%}")

        # Identify feature types
        self._identify_feature_types()

        # Visualizations
        self._create_visualizations()

    def _identify_feature_types(self):
        """Identify numerical and categorical features"""
        # Columns to exclude
        exclude_cols = ['RowNumber', 'CustomerId', 'Surname', self.target_column]

        all_features = [col for col in self.df.columns if col not in exclude_cols]

        for col in all_features:
            if self.df[col].dtype in ['int64', 'float64']:
                self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)

        print(f"\nNumerical Features: {self.numerical_features}")
        print(f"Categorical Features: {self.categorical_features}")

    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n4. Creating Visualizations...")

        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Churn Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Target distribution
        ax1 = axes[0, 0]
        self.df[self.target_column].value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title('Churn Distribution')
        ax1.set_xlabel('Churn Status')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(['Not Churned', 'Churned'], rotation=0)

        # 2. Age distribution by churn
        ax2 = axes[0, 1]
        for exited in [0, 1]:
            sns.histplot(data=self.df[self.df[self.target_column] == exited],
                        x='Age', ax=ax2, label=f'Churned: {exited}', alpha=0.7)
        ax2.set_title('Age Distribution by Churn Status')
        ax2.legend()

        # 3. Geography distribution
        ax3 = axes[0, 2]
        geography_churn = pd.crosstab(self.df['Geography'], self.df[self.target_column])
        geography_churn.plot(kind='bar', ax=ax3, color=['skyblue', 'salmon'])
        ax3.set_title('Churn by Geography')
        ax3.set_xlabel('Geography')
        ax3.set_ylabel('Count')
        ax3.legend(['Not Churned', 'Churned'])

        # 4. Gender distribution
        ax4 = axes[1, 0]
        gender_churn = pd.crosstab(self.df['Gender'], self.df[self.target_column])
        gender_churn.plot(kind='bar', ax=ax4, color=['skyblue', 'salmon'])
        ax4.set_title('Churn by Gender')
        ax4.set_xlabel('Gender')
        ax4.set_ylabel('Count')
        ax4.legend(['Not Churned', 'Churned'])

        # 5. Balance distribution
        ax5 = axes[1, 1]
        for exited in [0, 1]:
            sns.histplot(data=self.df[self.df[self.target_column] == exited],
                        x='Balance', ax=ax5, label=f'Churned: {exited}', alpha=0.7)
        ax5.set_title('Balance Distribution by Churn Status')
        ax5.legend()

        # 6. Credit Score distribution
        ax6 = axes[1, 2]
        for exited in [0, 1]:
            sns.histplot(data=self.df[self.df[self.target_column] == exited],
                        x='CreditScore', ax=ax6, label=f'Churned: {exited}', alpha=0.7)
        ax6.set_title('Credit Score Distribution by Churn Status')
        ax6.legend()

        plt.tight_layout()
        plt.savefig('churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.df[self.numerical_features + [self.target_column]].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def feature_engineering(self):
        """Perform feature engineering to create new features"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)

        # Create a copy for feature engineering
        df_engineered = self.df.copy()

        # 1. Age groups
        print("\n1. Creating Age Groups...")
        df_engineered['AgeGroup'] = pd.cut(df_engineered['Age'],
                                          bins=[0, 30, 40, 50, 60, 100],
                                          labels=['18-30', '31-40', '41-50', '51-60', '60+'])

        # 2. Balance categories
        print("2. Creating Balance Categories...")
        df_engineered['BalanceCategory'] = pd.cut(df_engineered['Balance'],
                                                 bins=[-1, 0, 50000, 100000, 150000, 200000, float('inf')],
                                                 labels=['No Balance', 'Low', 'Medium', 'High', 'Very High', 'Extreme'])

        # 3. Credit Score categories
        print("3. Creating Credit Score Categories...")
        df_engineered['CreditScoreCategory'] = pd.cut(df_engineered['CreditScore'],
                                                     bins=[0, 580, 670, 740, 800, 850],
                                                     labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])

        # 4. Salary to Balance ratio
        print("4. Creating Salary to Balance Ratio...")
        df_engineered['SalaryBalanceRatio'] = np.where(df_engineered['Balance'] > 0,
                                                      df_engineered['EstimatedSalary'] / df_engineered['Balance'],
                                                      0)

        # 5. Customer value score (composite feature)
        print("5. Creating Customer Value Score...")
        df_engineered['CustomerValueScore'] = (
            df_engineered['CreditScore'] / 850 * 0.3 +
            df_engineered['Balance'] / df_engineered['Balance'].max() * 0.3 +
            df_engineered['EstimatedSalary'] / df_engineered['EstimatedSalary'].max() * 0.2 +
            df_engineered['Tenure'] / 10 * 0.2
        )

        # 6. Product usage intensity
        print("6. Creating Product Usage Intensity...")
        df_engineered['ProductUsageIntensity'] = df_engineered['NumOfProducts'] * df_engineered['IsActiveMember']

        # 7. Interaction features
        print("7. Creating Interaction Features...")
        df_engineered['AgeBalanceInteraction'] = df_engineered['Age'] * df_engineered['Balance']
        df_engineered['CreditScoreTenureInteraction'] = df_engineered['CreditScore'] * df_engineered['Tenure']

        print("\nNew Features Created:")
        new_features = ['AgeGroup', 'BalanceCategory', 'CreditScoreCategory',
                       'SalaryBalanceRatio', 'CustomerValueScore', 'ProductUsageIntensity',
                       'AgeBalanceInteraction', 'CreditScoreTenureInteraction']

        for feature in new_features:
            print(f"  - {feature}")

        print(f"\nTotal features after engineering: {len(df_engineered.columns)}")

        return df_engineered

    def data_preprocessing(self, df_engineered):
        """Preprocess the data for machine learning"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)

        # Create a copy for preprocessing
        df_processed = df_engineered.copy()

        # Remove unnecessary columns
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        df_processed = df_processed.drop(columns=columns_to_drop)

        # Handle categorical variables
        print("\n1. Encoding Categorical Variables...")
        categorical_cols = ['Geography', 'Gender'] + [col for col in df_processed.columns
                                                     if df_processed[col].dtype == 'object']

        label_encoders = {}
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
                print(f"  - Encoded {col}")

        # Scale numerical features
        print("\n2. Scaling Numerical Features...")
        numerical_cols = [col for col in df_processed.columns
                         if df_processed[col].dtype in ['int64', 'float64']
                         and col != self.target_column]

        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        print(f"  - Scaled {len(numerical_cols)} numerical features")

        # Split the data
        print("\n3. Splitting Data into Train/Test Sets...")
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"  - Training set shape: {X_train.shape}")
        print(f"  - Test set shape: {X_test.shape}")
        print(f"  - Training target distribution: {y_train.value_counts().to_dict()}")
        print(f"  - Test target distribution: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test, df_processed, label_encoders, scaler

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("CUSTOMER CHURN COMPREHENSIVE ANALYSIS")
        print("="*60)

        # Load data
        self.load_data()

        # Exploratory Data Analysis
        self.exploratory_data_analysis()

        # Feature Engineering
        df_engineered = self.feature_engineering()

        # Data Preprocessing
        X_train, X_test, y_train, y_test, df_processed, label_encoders, scaler = self.data_preprocessing(df_engineered)

        # Final summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"[OK] Original dataset shape: {self.df.shape}")
        print(f"[OK] Engineered dataset shape: {df_engineered.shape}")
        print(f"[OK] Processed dataset shape: {df_processed.shape}")
        print(f"[OK] Training set ready: {X_train.shape}")
        print(f"[OK] Test set ready: {X_test.shape}")
        print(f"[OK] Total features created: {len(df_engineered.columns) - len(self.df.columns)}")
        print(f"[OK] Data preprocessing completed successfully!")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'df_original': self.df,
            'df_engineered': df_engineered,
            'df_processed': df_processed,
            'label_encoders': label_encoders,
            'scaler': scaler
        }

def main():
    """Main function to run the complete analysis"""
    # Initialize analyzer
    analyzer = CustomerChurnAnalyzer("Customer Churn Dataset.csv")

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    # Save processed data
    results['df_processed'].to_csv('processed_churn_data.csv', index=False)
    results['df_engineered'].to_csv('engineered_churn_data.csv', index=False)

    print("\n[OK] Processed data saved to 'processed_churn_data.csv'")
    print("[OK] Engineered data saved to 'engineered_churn_data.csv'")
    print("[OK] Analysis complete! The data is ready for machine learning modeling.")

if __name__ == "__main__":
    main()