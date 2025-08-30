from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pytz
from enum import Enum

db = SQLAlchemy()

class SignalStatus(Enum):
    ACTIVE = 'active'
    COMPLETED = 'completed'
    EXPIRED = 'expired'
    CANCELLED = 'cancelled'

class SignalResult(Enum):
    NONE = 'none'
    SL = 'sl'
    TP1 = 'tp1'
    TP2 = 'tp2'
    TP3 = 'tp3'
    BREAK_EVEN = 'break_even'
    MANUAL_EXIT = 'manual_exit'

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    total_trades = db.Column(db.Integer, default=0)
    winning_trades = db.Column(db.Integer, default=0)
    winrate = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    
    signals = db.relationship('SavedSignal', backref='user', lazy=True)

class SavedSignal(db.Model):
    __tablename__ = 'saved_signals'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    signal_type = db.Column(db.String(10), nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    sl_price = db.Column(db.Float, nullable=False)
    tp1_price = db.Column(db.Float, nullable=False)
    tp2_price = db.Column(db.Float, nullable=False)
    tp3_price = db.Column(db.Float, nullable=False)
    timeframe = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    expiration = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default=SignalStatus.ACTIVE.value)
    result = db.Column(db.String(20), default=SignalResult.NONE.value)
    recommendation = db.Column(db.Text)
    current_price = db.Column(db.Float)
    price_updated = db.Column(db.DateTime)
    candles_since_entry = db.Column(db.Integer, default=0)
    
    # Campos para seguimiento de estado
    hit_tp1 = db.Column(db.Boolean, default=False)
    hit_tp2 = db.Column(db.Boolean, default=False)
    hit_tp3 = db.Column(db.Boolean, default=False)
    moved_sl_to_be = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'entry_price': self.entry_price,
            'sl_price': self.sl_price,
            'tp1_price': self.tp1_price,
            'tp2_price': self.tp2_price,
            'tp3_price': self.tp3_price,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'expiration': self.expiration.isoformat(),
            'status': self.status,
            'result': self.result,
            'recommendation': self.recommendation,
            'current_price': self.current_price,
            'candles_since_entry': self.candles_since_entry
        }
    
    def is_expired(self):
        return datetime.utcnow() > self.expiration
    
    def check_price_levels(self, current_price):
        """Verificar si el precio ha tocado niveles importantes"""
        if self.signal_type == 'LONG':
            # Check for SL
            if current_price <= self.sl_price:
                return SignalResult.SL, "Toco Stop Loss, esta fue una falsa señal. Lo siento"
            
            # Check for TP levels
            if current_price >= self.tp3_price and not self.hit_tp3:
                return SignalResult.TP3, "Toco Take Profit 3, Felicidades la operación fue todo un éxito"
            elif current_price >= self.tp2_price and not self.hit_tp2:
                return SignalResult.TP2, "Toco Take Profit 2, si gustas salte de la operación con buenas ganancias o coloca SL en Take Profit 1"
            elif current_price >= self.tp1_price and not self.hit_tp1:
                return SignalResult.TP1, "Toco Take Profit 1, coloca tu SL a break even"
            
            # Check if price returned to entry after hitting TP1
            if self.hit_tp1 and current_price <= self.entry_price and not self.moved_sl_to_be:
                return SignalResult.BREAK_EVEN, "Operación exitosa, toco TP1 y luego Break Even"
                
        else:  # SHORT
            # Check for SL
            if current_price >= self.sl_price:
                return SignalResult.SL, "Toco Stop Loss, esta fue una falsa señal. Lo siento"
            
            # Check for TP levels
            if current_price <= self.tp3_price and not self.hit_tp3:
                return SignalResult.TP3, "Toco Take Profit 3, Felicidades la operación fue todo un éxito"
            elif current_price <= self.tp2_price and not self.hit_tp2:
                return SignalResult.TP2, "Toco Take Profit 2, si gustas salte de la operación con buenas ganancias o coloca SL en Take Profit 1"
            elif current_price <= self.tp1_price and not self.hit_tp1:
                return SignalResult.TP1, "Toco Take Profit 1, coloca tu SL a break even"
            
            # Check if price returned to entry after hitting TP1
            if self.hit_tp1 and current_price >= self.entry_price and not self.moved_sl_to_be:
                return SignalResult.BREAK_EVEN, "Operación exitosa, toco TP1 y luego Break Even"
        
        return None, None

# Duración de señales guardadas por timeframe (en horas)
SIGNAL_DURATIONS = {
    '15m': 6,    # 6 horas (24 velas de 15m)
    '30m': 12,   # 12 horas (24 velas de 30m)
    '1h': 24,    # 24 horas (24 velas de 1h)
    '2h': 48,    # 48 horas (24 velas de 2h)
    '4h': 96,    # 96 horas (24 velas de 4h)
    '1d': 576,   # 24 días (24 velas de 1d)
    '1w': 4032   # 24 semanas (24 velas de 1w)
}

# Tiempo de vela adicional después de alertas (en horas)
EXTRA_CANDLE_DURATION = {
    '15m': 0.25,  # 15 minutos
    '30m': 0.5,   # 30 minutos
    '1h': 1,      # 1 hora
    '2h': 2,      # 2 horas
    '4h': 4,      # 4 horas
    '1d': 24,     # 1 día
    '1w': 168     # 1 semana
}

def init_app(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        
        # Create default users if they don't exist
        default_users = [
            {'username': 'Willer', 'password': '1234'},
            {'username': 'Danilo', 'password': '1234'},
            {'username': 'Damir', 'password': '1234'}
        ]
        
        for user_data in default_users:
            user = User.query.filter_by(username=user_data['username']).first()
            if not user:
                from werkzeug.security import generate_password_hash
                user = User(
                    username=user_data['username'],
                    password_hash=generate_password_hash(user_data['password'])
                )
                db.session.add(user)
        
        db.session.commit()

def calculate_winrate(user_id):
    """Calcular winrate actualizado para un usuario"""
    user = User.query.get(user_id)
    if not user:
        return 0.0
    
    if user.total_trades > 0:
        user.winrate = (user.winning_trades / user.total_trades) * 100
    else:
        user.winrate = 0.0
    
    db.session.commit()
    return user.winrate

def update_signal_status(signal, result, recommendation):
    """Actualizar el estado de una señal y ajustar estadísticas del usuario"""
    if result != signal.result:  # Solo actualizar si el resultado cambió
        user = signal.user
        
        # Si era una operación activa y ahora tiene resultado, actualizar stats
        if signal.status == SignalStatus.ACTIVE.value and result != SignalResult.NONE.value:
            user.total_trades += 1
            
            # Considerar operaciones ganadoras
            if result in [SignalResult.TP1.value, SignalResult.TP2.value, 
                         SignalResult.TP3.value, SignalResult.BREAK_EVEN.value]:
                user.winning_trades += 1
        
        signal.result = result
        signal.recommendation = recommendation
        
        # Si el resultado finaliza la operación, marcar como completada
        if result in [SignalResult.SL.value, SignalResult.TP3.value, 
                     SignalResult.MANUAL_EXIT.value, SignalResult.BREAK_EVEN.value]:
            signal.status = SignalStatus.COMPLETED.value
            # Añadir tiempo extra de vela
            extra_hours = EXTRA_CANDLE_DURATION.get(signal.timeframe, 1)
            signal.expiration = datetime.utcnow() + timedelta(hours=extra_hours)
        
        db.session.commit()
        calculate_winrate(user.id)

def check_signal_duration(signal):
    """Verificar si la señal está durando demasiado"""
    timeframe = signal.timeframe
    expected_duration = SIGNAL_DURATIONS.get(timeframe, 24)
    half_duration = expected_duration / 2
    
    time_since_creation = (datetime.utcnow() - signal.timestamp).total_seconds() / 3600  # horas
    
    if time_since_creation > half_duration and signal.result == SignalResult.NONE.value:
        return "Esta operación está durando mucho, sal de esta con discreción"
    
    return None
