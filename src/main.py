import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Define the missing classes
class WalmartForecastAnalyzer:
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self, df):
        """Load and preprocess Walmart data for forecasting"""
        print("Loading and preprocessing data...")
        
        # Convert date and sort
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df = df.sort_values(['Store', 'Date'])
        
        # Store the original data
        self.data = df.copy()
        
        return df
    
    def analyze_time_series_properties(self, store_num=1):
        """Analyze time series properties for a specific store"""
        store_data = self.data[self.data['Store'] == store_num].copy()
        store_data = store_data.set_index('Date')
        
        print(f"\n=== Time Series Analysis for Store {store_num} ===")
        print(f"Time period: {store_data.index.min()} to {store_data.index.max()}")
        print(f"Total weeks: {len(store_data)}")
        print(f"Average weekly sales: ${store_data['Weekly_Sales'].mean():,.2f}")
        print(f"Sales std: ${store_data['Weekly_Sales'].std():,.2f}")
        print(f"Min sales: ${store_data['Weekly_Sales'].min():,.2f}")
        print(f"Max sales: ${store_data['Weekly_Sales'].max():,.2f}")
        
        return store_data
    
    def create_forecasting_dataset(self, store_num=1, seq_len=52, pred_len=4):
        """Create sequences for time series forecasting"""
        store_data = self.data[self.data['Store'] == store_num].copy()
        store_data = store_data.sort_values('Date')
        
        # Use only sales data for univariate forecasting
        sales_data = store_data['Weekly_Sales'].values
        
        # Normalize
        sales_normalized = self.scaler.fit_transform(sales_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(sales_normalized) - seq_len - pred_len + 1):
            seq = sales_normalized[i:i + seq_len]
            target = sales_normalized[i + seq_len:i + seq_len + pred_len]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets), sales_data

# Define the missing enhanced_forecasting_pipeline function
def enhanced_forecasting_pipeline(df, store_num=1):
    """Enhanced forecasting pipeline"""
    analyzer = WalmartForecastAnalyzer()
    processed_data = analyzer.load_and_preprocess_data(df)
    
    # Analyze the store
    store_data = analyzer.analyze_time_series_properties(store_num)
    
    # Create forecasting dataset
    sequences, targets, original_sales = analyzer.create_forecasting_dataset(
        store_num=store_num, seq_len=52, pred_len=4
    )
    
    # Split data
    split_idx = int(0.8 * len(sequences))
    train_seq, val_seq = sequences[:split_idx], sequences[split_idx:]
    train_targ, val_targ = targets[:split_idx], targets[split_idx:]
    
    print(f"\n=== Forecasting Dataset Created ===")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Total samples: {len(sequences)}")
    
    return {
        'train_sequences': train_seq,
        'train_targets': train_targ,
        'val_sequences': val_seq,
        'val_targets': val_targ,
        'scaler': analyzer.scaler,
        'original_sales': original_sales
    }

# Define a simple config class
class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.seq_len = 52
        self.pred_len = 4

# Fixed PatchTST model that outputs correct prediction length
class SimplePatchTST(nn.Module):
    def __init__(self, config, n_channels):
        super(SimplePatchTST, self).__init__()
        self.config = config
        self.n_channels = n_channels
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        
        # Encoder processes the input sequence
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True  # Important: set batch_first=True
            ),
            num_layers=3
        )
        
        # Projection layers
        self.input_projection = nn.Linear(n_channels, 64)
        
        # Output projection - directly to prediction length
        self.output_projection = nn.Linear(self.seq_len, self.pred_len)
        
        # Final layer to get single value per prediction step
        self.final_projection = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_channels)
        batch_size, seq_len, n_channels = x.shape
        
        # Project input
        x = self.input_projection(x)  # (batch_size, seq_len, 64)
        
        # Apply transformer encoder (batch_first=True so no need to transpose)
        encoded = self.encoder(x)  # (batch_size, seq_len, 64)
        
        # Use only the last few time steps for prediction, or use all
        # Option 1: Use all encoded features and project to prediction length
        # Transpose to (batch_size, 64, seq_len) for linear layer
        encoded_t = encoded.transpose(1, 2)  # (batch_size, 64, seq_len)
        
        # Project sequence length to prediction length
        pred_features = self.output_projection(encoded_t)  # (batch_size, 64, pred_len)
        
        # Transpose back and project to output dimension
        pred_features = pred_features.transpose(1, 2)  # (batch_size, pred_len, 64)
        output = self.final_projection(pred_features)  # (batch_size, pred_len, 1)
        
        return output

class EnhancedPatchTSTForecaster:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = None
        self.scaler = None
        
    def prepare_patchtst_data(self, forecasting_data):
        """Prepare data specifically for PatchTST model"""
        train_seq = forecasting_data['train_sequences']
        train_targ = forecasting_data['train_targets']
        val_seq = forecasting_data['val_sequences']
        val_targ = forecasting_data['val_targets']
        
        # Convert to PyTorch tensors
        train_sequences = torch.FloatTensor(train_seq).unsqueeze(-1)  # Add channel dimension
        train_targets = torch.FloatTensor(train_targ).unsqueeze(-1)
        val_sequences = torch.FloatTensor(val_seq).unsqueeze(-1)
        val_targets = torch.FloatTensor(val_targ).unsqueeze(-1)
        
        print(f"Input sequences shape: {train_sequences.shape}")
        print(f"Target sequences shape: {train_targets.shape}")
        
        # Create data loaders
        train_dataset = TensorDataset(train_sequences, train_targets)
        val_dataset = TensorDataset(val_sequences, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader):
        """Train the PatchTST model with enhanced monitoring"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_seq, batch_targ in train_loader:
                batch_seq, batch_targ = batch_seq.to(self.device), batch_targ.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_seq)
                
                # Ensure output and target have same shape
                if outputs.shape != batch_targ.shape:
                    print(f"Shape mismatch - Output: {outputs.shape}, Target: {batch_targ.shape}")
                
                loss = criterion(outputs, batch_targ)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_seq, batch_targ in val_loader:
                    batch_seq, batch_targ = batch_seq.to(self.device), batch_targ.to(self.device)
                    outputs = self.model(batch_seq)
                    val_loss += criterion(outputs, batch_targ).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.config.num_epochs}], '
                      f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        return train_losses, val_losses
    
    def forecast_future(self, model, last_sequence, steps=12):
        """Generate multi-step forecasts"""
        model.eval()
        forecasts = []
        current_sequence = last_sequence.clone()
        
        with torch.no_grad():
            for step in range(steps):
                # Predict next 4 weeks
                pred = model(current_sequence.unsqueeze(0))  # (1, pred_len, 1)
                
                # Take all predictions for this step
                step_forecasts = pred[0, :, 0].cpu().numpy()
                forecasts.extend(step_forecasts)
                
                if step < steps - 1:  # Don't update after last prediction
                    # Update sequence: remove first pred_len elements, add new predictions
                    remaining_sequence = current_sequence[self.config.pred_len:]
                    new_predictions = pred[0].detach()  # (pred_len, 1)
                    
                    # Ensure we maintain sequence length
                    if len(remaining_sequence) + len(new_predictions) == self.config.seq_len:
                        current_sequence = torch.cat([remaining_sequence, new_predictions], dim=0)
                    else:
                        # If sequence length doesn't match, use a different approach
                        current_sequence = torch.cat([
                            current_sequence[self.config.pred_len:], 
                            new_predictions
                        ], dim=0)
                        # Truncate or pad to maintain seq_len
                        if len(current_sequence) > self.config.seq_len:
                            current_sequence = current_sequence[-self.config.seq_len:]
        
        return forecasts[:steps]  # Return only the requested number of steps

def run_complete_forecasting_pipeline(df):
    """Complete forecasting pipeline integrating analysis and PatchTST"""
    
    # 1. Enhanced Analysis
    analyzer = WalmartForecastAnalyzer()
    processed_data = analyzer.load_and_preprocess_data(df)
    forecasting_data = enhanced_forecasting_pipeline(df)
    
    # 2. Prepare PatchTST Data
    config = Config()
    forecaster = EnhancedPatchTSTForecaster(config)
    
    # Initialize your PatchTST model
    n_channels = 1  # Univariate
    model = SimplePatchTST(config, n_channels)
    model = model.to(config.device)
    
    # Print model architecture
    print(f"\n=== Model Architecture ===")
    print(f"Input shape: (batch_size, {config.seq_len}, {n_channels})")
    print(f"Output shape: (batch_size, {config.pred_len}, 1)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    forecaster.model = model
    
    # 3. Prepare data loaders
    train_loader, val_loader = forecaster.prepare_patchtst_data(forecasting_data)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # 4. Train model
    print("\n=== Starting PatchTST Training ===")
    train_losses, val_losses = forecaster.train_model(train_loader, val_loader)
    
    # 5. Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 6. Generate forecasts
    last_sequence = torch.FloatTensor(forecasting_data['val_sequences'][-1]).unsqueeze(-1)
    forecasts = forecaster.forecast_future(model, last_sequence.to(config.device), steps=12)
    
    # Convert back to original scale
    forecasts_original = forecasting_data['scaler'].inverse_transform(
        np.array(forecasts).reshape(-1, 1)
    ).flatten()
    
    # 7. Plot forecasts
    plt.subplot(1, 2, 2)
    actual_sales = forecasting_data['original_sales'][-24:]  # Last 24 weeks
    forecast_weeks = range(len(actual_sales), len(actual_sales) + len(forecasts_original))
    
    plt.plot(range(len(actual_sales)), actual_sales, label='Historical', marker='o')
    plt.plot(forecast_weeks, forecasts_original, label='Forecast', marker='s', color='red')
    plt.title('12-Week Sales Forecast')
    plt.xlabel('Weeks')
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/complete_forecasting_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Forecasting Results ===")
    for i, forecast in enumerate(forecasts_original, 1):
        print(f"Week {i}: ${forecast:,.2f}")
    print(f"Average forecast: ${forecasts_original.mean():,.2f}")
    
    return {
        'model': model,
        'forecasts': forecasts_original,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

# Store Comparison Forecasting
def multi_store_forecasting(df, store_list=[1, 2]):
    """Compare forecasts across multiple stores"""
    
    store_forecasts = {}
    
    for store in store_list:
        print(f"\n=== Forecasting for Store {store} ===")
        
        # Analyze store
        analyzer = WalmartForecastAnalyzer()
        processed_data = analyzer.load_and_preprocess_data(df)
        store_data = df[df['Store'] == store].copy()
        
        # Create forecasting dataset for this store
        sequences, targets, original_sales = analyzer.create_forecasting_dataset(
            store_num=store, seq_len=52, pred_len=4
        )
        
        # Prepare data
        forecasting_data = {
            'train_sequences': sequences[:int(0.8 * len(sequences))],
            'train_targets': targets[:int(0.8 * len(targets))],
            'val_sequences': sequences[int(0.8 * len(sequences)):],
            'val_targets': targets[int(0.8 * len(targets)):],
            'scaler': analyzer.scaler,
            'original_sales': original_sales
        }
        
        print(f"Store {store}: {len(sequences)} sequences ready for training")
        
        store_stats = {
            'avg_sales': store_data['Weekly_Sales'].mean(),
            'sequences': len(sequences),
            'data_quality': 'Good' if len(store_data) >= 52 else 'Limited'
        }
        
        store_forecasts[store] = store_stats
    
    # Compare stores
    print("\n=== Store Comparison ===")
    comparison_df = pd.DataFrame(store_forecasts).T
    print(comparison_df)
    
    return store_forecasts

def enhanced_analysis_and_reporting(results, df, store_num=1):
    """Provide comprehensive analysis of forecasting results"""
    
    print("\n" + "="*60)
    print("ENHANCED FORECASTING ANALYSIS REPORT")
    print("="*60)
    
    # Get store data for comparison
    store_data = df[df['Store'] == store_num]
    historical_avg = store_data['Weekly_Sales'].mean()
    forecast_avg = results['forecasts'].mean()
    
    # Calculate performance metrics
    variance_from_historical = ((forecast_avg - historical_avg) / historical_avg) * 100
    forecast_std = results['forecasts'].std()
    historical_std = store_data['Weekly_Sales'].std()
    
    print(f"\n📊 STORE {store_num} PERFORMANCE METRICS:")
    print(f"   Historical Average: ${historical_avg:,.2f}")
    print(f"   12-Week Forecast Average: ${forecast_avg:,.2f}")
    print(f"   Forecast vs Historical: {variance_from_historical:+.2f}%")
    print(f"   Historical Volatility (std): ${historical_std:,.2f}")
    print(f"   Forecast Volatility (std): ${forecast_std:,.2f}")
    
    # Trend analysis
    forecast_trend = (results['forecasts'][-1] - results['forecasts'][0]) / results['forecasts'][0] * 100
    print(f"   Forecast Trend: {forecast_trend:+.2f}% over 12 weeks")
    
    # Confidence intervals (simplified)
    confidence_upper = forecast_avg + 1.96 * forecast_std
    confidence_lower = forecast_avg - 1.96 * forecast_std
    
    print(f"\n🎯 FORECAST CONFIDENCE INTERVALS (95%):")
    print(f"   Upper Bound: ${confidence_upper:,.2f}")
    print(f"   Lower Bound: ${confidence_lower:,.2f}")
    print(f"   Range: ${confidence_upper - confidence_lower:,.2f}")
    
    # Business insights
    print(f"\n💡 BUSINESS INSIGHTS:")
    if variance_from_historical > 5:
        print("   📈 POSITIVE: Forecast suggests growth above historical average")
    elif variance_from_historical < -5:
        print("   📉 CAUTION: Forecast below historical average - review strategy")
    else:
        print("   ↔️  STABLE: Forecast aligns with historical performance")
    
    if forecast_std < historical_std:
        print("   ✅ STABLE: Forecast shows less volatility than historical data")
    else:
        print("   ⚠️  VOLATILE: Forecast shows higher volatility than historical")
    
    # Training quality assessment
    final_train_loss = results['train_losses'][-1]
    final_val_loss = results['val_losses'][-1]
    
    print(f"\n🤖 MODEL TRAINING QUALITY:")
    print(f"   Final Training Loss: {final_train_loss:.6f}")
    print(f"   Final Validation Loss: {final_val_loss:.6f}")
    
    if final_val_loss < 0.5:
        print("   ✅ EXCELLENT: Model shows strong generalization")
    elif final_val_loss < 1.0:
        print("   👍 GOOD: Model performance is acceptable")
    else:
        print("   🔧 NEEDS IMPROVEMENT: Consider model tuning")
    
    return {
        'historical_avg': historical_avg,
        'forecast_avg': forecast_avg,
        'variance_pct': variance_from_historical,
        'confidence_upper': confidence_upper,
        'confidence_lower': confidence_lower
    }

def create_comprehensive_visualization(results, df, store_num=1):
    """Create enhanced visualization with insights"""
    
    plt.figure(figsize=(16, 12))
    
    # 1. Training Progress
    plt.subplot(3, 2, 1)
    plt.plot(results['train_losses'], label='Training Loss', linewidth=2)
    plt.plot(results['val_losses'], label='Validation Loss', linewidth=2)
    plt.title('Model Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Historical vs Forecast
    plt.subplot(3, 2, 2)
    store_data = df[df['Store'] == store_num]
    recent_sales = store_data['Weekly_Sales'].values[-24:]
    
    forecast_weeks = range(len(recent_sales), len(recent_sales) + len(results['forecasts']))
    
    plt.plot(range(len(recent_sales)), recent_sales, 
             label='Historical Sales', marker='o', linewidth=2, color='blue')
    plt.plot(forecast_weeks, results['forecasts'], 
             label='Forecast', marker='s', linewidth=2, color='red')
    plt.axhline(y=store_data['Weekly_Sales'].mean(), 
                color='green', linestyle='--', alpha=0.7, label='Historical Avg')
    plt.title('12-Week Sales Forecast vs Historical', fontsize=14, fontweight='bold')
    plt.xlabel('Weeks')
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Forecast Distribution
    plt.subplot(3, 2, 3)
    plt.hist(results['forecasts'], bins=8, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(results['forecasts'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${results["forecasts"].mean():,.0f}')
    plt.title('Forecast Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sales ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Monthly Pattern Analysis
    plt.subplot(3, 2, 4)
    store_data = df[df['Store'] == store_num].copy()
    store_data['Date'] = pd.to_datetime(store_data['Date'], format='%d-%m-%Y')
    store_data['Month'] = store_data['Date'].dt.month
    monthly_avg = store_data.groupby('Month')['Weekly_Sales'].mean()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(months, monthly_avg, alpha=0.7, color='purple')
    plt.title('Average Monthly Sales Pattern', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Average Sales ($)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Store Comparison
    plt.subplot(3, 2, 5)
    store_comparison = df.groupby('Store')['Weekly_Sales'].mean().head(5)
    plt.bar(store_comparison.index.astype(str), store_comparison.values, 
            alpha=0.7, color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.title('Top 5 Stores: Average Sales Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Store Number')
    plt.ylabel('Average Sales ($)')
    plt.grid(True, alpha=0.3)
    
    # 6. Forecast Trend
    plt.subplot(3, 2, 6)
    weeks = range(1, len(results['forecasts']) + 1)
    plt.plot(weeks, results['forecasts'], marker='o', linewidth=2, color='red')
    plt.fill_between(weeks, 
                    results['forecasts'] - results['forecasts'].std(),
                    results['forecasts'] + results['forecasts'].std(),
                    alpha=0.2, color='red')
    plt.title('12-Week Forecast Trend with Confidence', fontsize=14, fontweight='bold')
    plt.xlabel('Weeks Ahead')
    plt.ylabel('Sales ($)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_forecasting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the complete pipeline
if __name__ == "__main__":
    # Use your actual data here
    df_path = r"C:\Users\Anvita\Desktop\ML_project\Walmart.csv"    
    df = pd.read_csv(df_path)
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run single store forecasting
    results = run_complete_forecasting_pipeline(df)
    
    # Enhanced analysis and reporting
    analysis_results = enhanced_analysis_and_reporting(results, df, store_num=1)
    
    # Comprehensive visualization
    create_comprehensive_visualization(results, df, store_num=1)
    
    # Compare multiple stores
    store_comparison = multi_store_forecasting(df, store_list=[1, 2])
    
    print("\n🎉 FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
    print("📁 Results saved to 'results/' directory")
    print("📊 Comprehensive analysis generated")