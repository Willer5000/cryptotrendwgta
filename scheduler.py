from apscheduler.schedulers.blocking import BlockingScheduler
from data_fetcher import update_all_data
from analyzer import analyze_all
import os

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=15)
def scheduled_job():
    print('Iniciando actualización de datos...')
    update_all_data()
    analyze_all()
    print('Análisis completado!')

if __name__ == '__main__':
    if os.environ.get('RUN_MAIN') == 'true':
        sched.start()
