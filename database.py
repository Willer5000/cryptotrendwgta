from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pytz

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    winrate = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
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
    status = db.Column(db.String(20), default='active')
    result = db.Column(db.String(20))
    recommendation = db.Column(db.Text)
    last_alert = db.Column(db.DateTime)
    
    def is_expired(self):
        return datetime.utcnow() > self.expiration
    
    def check_status(self, current_price, high_price, low_price):
        # Si ya hay un resultado final, no hacer nada
        if self.result in ['sl', 'tp3', 'expired']:
            return self.result, self.recommendation

        # Regla 1: Si toca SL
        if (self.signal_type == 'LONG' and low_price <= self.sl_price) or \
           (self.signal_type == 'SHORT' and high_price >= self.sl_price):
            self.result = 'sl'
            self.recommendation = "Toco Stop Loss, esta fue una falsa señal. Lo siento"
            # Extender expiración por 1 vela más
            self.extend_expiration(1)
            return self.result, self.recommendation

        # Regla 3: Si toca TP1
        if (self.signal_type == 'LONG' and high_price >= self.tp1_price) or \
           (self.signal_type == 'SHORT' and low_price <= self.tp1_price):
            if self.result != 'tp1':
                self.result = 'tp1'
                self.recommendation = "Toco Take Profit 1, coloca tu SL a break even"
                self.last_alert = datetime.utcnow()
            return self.result, self.recommendation

        # Regla 4: Si toca TP1 y luego break even (precio de entrada)
        if self.result == 'tp1':
            if (self.signal_type == 'LONG' and low_price <= self.entry_price) or \
               (self.signal_type == 'SHORT' and high_price >= self.entry_price):
                self.result = 'tp1_be'
                self.recommendation = "Operación exitosa, toco TP1 y luego Break Even"
                self.extend_expiration(1)
                return self.result, self.recommendation

        # Regla 5: Si toca TP1 y no toca TP2 después de un tiempo
        if self.result == 'tp1' and self.last_alert:
            # Verificar si ha pasado la mitad del tiempo de la señal
            half_time = self.timestamp + (self.expiration - self.timestamp) / 2
            if datetime.utcnow() > half_time:
                self.result = 'tp1_exit'
                self.recommendation = "No creo que toque TP2, salte con discreción con ganancias. Felicidades"
                self.extend_expiration(1)
                return self.result, self.recommendation

        # Regla 6: Si toca TP2
        if (self.signal_type == 'LONG' and high_price >= self.tp2_price) or \
           (self.signal_type == 'SHORT' and low_price <= self.tp2_price):
            if self.result != 'tp2':
                self.result = 'tp2'
                self.recommendation = "Toco Take Profit 2, si gustas salte de la operación con buenas ganancias o coloca SL en Take Profit 1"
                self.last_alert = datetime.utcnow()
            return self.result, self.recommendation

        # Regla 7: Si toca TP2 y luego TP1
        if self.result == 'tp2':
            if (self.signal_type == 'LONG' and low_price <= self.tp1_price) or \
               (self.signal_type == 'SHORT' and high_price >= self.tp1_price):
                self.result = 'tp2_tp1'
                self.recommendation = "Si no te saliste de la operación anterior y colocaste tu SL en TP1, felicidades de todas maneras tuviste una buena utilidad, caso contrario salte con discreción"
                self.extend_expiration(1)
                return self.result, self.recommendation

        # Regla 8: Si toca TP3
        if (self.signal_type == 'LONG' and high_price >= self.tp3_price) or \
           (self.signal_type == 'SHORT' and low_price <= self.tp3_price):
            self.result = 'tp3'
            self.recommendation = "Toco Take Profit 3, Felicidades la operación fue todo un éxito"
            self.extend_expiration(1)
            return self.result, self.recommendation

        # Regla 2: Si la operación no toca nada y ha pasado la mitad del tiempo
        half_time = self.timestamp + (self.expiration - self.timestamp) / 2
        if datetime.utcnow() > half_time and self.result is None:
            self.recommendation = "Esta operación está durando mucho, sal de esta con discreción"
            self.last_alert = datetime.utcnow()
            # No cambiamos el resultado, solo la recomendación
            return self.result, self.recommendation

        # Si no hay cambios
        return self.result, self.recommendation

    def extend_expiration(self, candles=1):
        # Extender la expiración por un número de velas
        timeframe_delta = {
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1w': timedelta(weeks=1)
        }.get(self.timeframe, timedelta(hours=1))
        
        self.expiration = datetime.utcnow() + candles * timeframe_delta

# Duración de señales guardadas por timeframe (en horas)
SIGNAL_DURATIONS = {
    '15m': 4,    # 4 horas (16 velas)
    '30m': 8,    # 8 horas (16 velas)
    '1h': 16,    # 16 horas (16 velas)
    '2h': 32,    # 32 horas (16 velas)
    '4h': 64,    # 64 horas (16 velas)
    '1d': 384,   # 16 días (16 velas)
    '1w': 2688   # 16 semanas (16 velas)
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
