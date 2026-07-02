"""数据库初始化 — 创建 SQLite 表结构"""
import sqlite3

from src.config.settings import settings


def init_db():
    """创建售后工单和风控日志表"""
    db_path = settings.DATABASE_URL.replace("sqlite:///./", "")
    # 确保目录存在
    import os
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 售后工单表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS after_sale_tickets (
            ticket_id TEXT PRIMARY KEY,
            customer_request TEXT NOT NULL,
            issue_type TEXT,
            severity TEXT,
            approval_status TEXT,
            approver_comment TEXT,
            resolution TEXT,
            customer_feedback TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 风控日志表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wechat_risk_logs (
            message_id TEXT PRIMARY KEY,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
            message_type TEXT,
            risk_score REAL,
            risk_category TEXT,
            action TEXT,
            escalation_decision TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print(f"[OK] 数据库初始化完成 — {db_path}")


if __name__ == "__main__":
    init_db()
