import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    def test_strategy(self, signals, data):
        self.equity_curve = [self.initial_capital]
        
        for i, signal in enumerate(signals):
            if i >= len(data):
                break
                
            current_data = data.iloc[i]
            current_price = current_data['close']
            
            if signal['type'] == 'LONG':
                self.execute_long(signal, current_price, i)
            elif signal['type'] == 'SHORT':
                self.execute_short(signal, current_price, i)
            
            # Actualizar curva de equity
            self.update_equity(current_price)
                
        return self.calculate_performance()
    
    def execute_long(self, signal, current_price, idx):
        entry_price = signal['entry']
        stop_loss = signal['sl']
        take_profit1 = signal['tp1']
        take_profit2 = signal['tp2']
        
        # Calcular tamaño de posición (2% de riesgo)
        risk_amount = self.capital * 0.02
        risk_per_share = entry_price - stop_loss
        position_size = risk_amount / risk_per_share
        
        # Ejecutar entrada
        entry_cost = entry_price * position_size
        if entry_cost > self.capital:
            position_size = self.capital / entry_price
            entry_cost = self.capital
        
        self.capital -= entry_cost
        self.positions.append({
            'type': 'LONG',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit1': take_profit1,
            'take_profit2': take_profit2,
            'size': position_size,
            'entry_time': idx
        })
    
    def execute_short(self, signal, current_price, idx):
        entry_price = signal['entry']
        stop_loss = signal['sl']
        take_profit1 = signal['tp1']
        take_profit2 = signal['tp2']
        
        # Calcular tamaño de posición (2% de riesgo)
        risk_amount = self.capital * 0.02
        risk_per_share = stop_loss - entry_price
        position_size = risk_amount / risk_per_share
        
        # Ejecutar entrada
        entry_cost = entry_price * position_size
        margin_required = entry_cost * 0.5  # Asumir 50% de margen
        
        if margin_required > self.capital:
            position_size = (self.capital * 2) / entry_price
            entry_cost = entry_price * position_size
        
        self.capital -= margin_required
        self.positions.append({
            'type': 'SHORT',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit1': take_profit1,
            'take_profit2': take_profit2,
            'size': position_size,
            'entry_time': idx,
            'margin': margin_required
        })
    
    def update_equity(self, current_price):
        total_value = self.capital
        
        for pos in self.positions:
            if pos['type'] == 'LONG':
                position_value = pos['size'] * current_price
                total_value += position_value
            else:  # SHORT
                # Para posiciones cortas, el valor es (precio entrada - precio actual) * tamaño
                pnl = (pos['entry_price'] - current_price) * pos['size']
                total_value += pos['margin'] + pnl
        
        self.equity_curve.append(total_value)
    
    def calculate_performance(self):
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown(equity)
        win_rate = len([r for r in returns if r > 0]) / len(returns) * 100 if returns.any() else 0
        
        return {
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 2),
            'final_equity': round(equity[-1], 2),
            'equity_curve': equity
        }
    
    def calculate_max_drawdown(self, equity):
        peak = equity[0]
        max_dd = 0
        
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
    
    def generate_report(self):
        performance = self.calculate_performance()
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('Curva de Equity')
        plt.xlabel('Períodos')
        plt.ylabel('Capital ($)')
        plt.grid(True)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        equity_plot = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        report = f"""
        <h3>Reporte de Backtesting</h3>
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Métricas de Performance</div>
                    <div class="card-body">
                        <p><strong>Retorno Total:</strong> {performance['total_return']}%</p>
                        <p><strong>Ratio Sharpe:</strong> {performance['sharpe_ratio']}</p>
                        <p><strong>Máximo Drawdown:</strong> {performance['max_drawdown']}%</p>
                        <p><strong>Ratio de Aciertos:</strong> {performance['win_rate']}%</p>
                        <p><strong>Capital Final:</strong> ${performance['final_equity']:,.2f}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Curva de Equity</div>
                    <div class="card-body">
                        <img src="data:image/png;base64,{equity_plot}" class="img-fluid">
                    </div>
                </div>
            </div>
        </div>
        """
        
        return report

# Integración con Flask
@app.route('/backtest')
def backtest():
    # Obtener señales históricas
    with analysis_state['lock']:
        params = analysis_state['params']
        current_timeframe = params['timeframe']
        
        if current_timeframe in analysis_state['timeframe_data']:
            timeframe_data = analysis_state['timeframe_data'][current_timeframe]
            historical_signals = list(timeframe_data['historical_signals'])
        else:
            historical_signals = []
    
    # Obtener datos para backtesting (simplificado)
    # En una implementación real, necesitarías datos históricos completos
    backtester = Backtester()
    performance = backtester.test_strategy(historical_signals, pd.DataFrame())
    report = backtester.generate_report()
    
    return render_template('backtest.html', report=report)
