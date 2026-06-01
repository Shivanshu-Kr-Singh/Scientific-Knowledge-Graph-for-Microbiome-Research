from apscheduler.schedulers.background import BackgroundScheduler
from scheduler.jobs import daily_update, weekly_refresh, monthly_rescan

def build_scheduler():
    sched = BackgroundScheduler()

    sched.add_job(
        daily_update,
        trigger="cron",
        hour=2
    )

    sched.add_job(
        weekly_refresh,
        trigger="cron",
        day_of_week="sun",
        hour=3
    )

    sched.add_job(
        monthly_rescan,
        trigger="cron",
        day=1,
        hour=4
    )

    return sched