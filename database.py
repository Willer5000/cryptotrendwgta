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
    
    def is_expired(self):
        return datetime.utcnow() > self.expiration
    
    def check_status(self, current_price):
        if self.signal_type == 'LONG':
            if current_price <= self.sl_price:
                return 'sl', "Toco Stop Loss, esta fue una falsa señal. Lo siento"
            elif current_price >= self.tp3_price:
                return 'tp3', "Toco Take Profit 3, Felicidades la operación fue todo un éxito"
            elif current_price >= self.tp2_price:
                return 'tp2', "Toco Take Profit 2, si gustas salte de la operación con buenas ganancias o coloca SL en Take Profit 1"
            elif current_price >= self.tp1_price:
                return 'tp1', "Toco Take Profit 1, coloca tu SL a break even"
        else:  # SHORT
            if current_price >= self.sl_price:
                return 'sl', "Toco Stop Loss, esta fue una falsa señal. Lo siento"
            elif current_price <= self.tp3_price:
                return 'tp3', "Toco Take Profit 3, Felicidades la operación fue todo un éxito"
            elif current_price <= self.tp2_price:
                return 'tp2', "Toco Take Profit 2, si gustas salte de la operación con buenas ganancias o coloca SL en Take Profit 1"
            elif current_price <= self.tp1_price:
                return 'tp1', "Toco Take Profit 1, coloca tu SL a break even"
        
        # Check if signal expired
        if self.is_expired():
            return 'expired', "Señal expirada sin resultado claro"
            
        return None, None

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
