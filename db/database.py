from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# =========================
# CONNECTION STRING (CORRECT)
# =========================
DATABASE_URL = (
    "mssql+pyodbc://sa:YourStrong%40123"
    "@127.0.0.1:1433/VideoPolygonDB"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Encrypt=yes"
    "&TrustServerCertificate=yes"
)

# =========================
# ENGINE + SESSION
# =========================
engine = create_engine(
    DATABASE_URL,
    echo=True,
    future=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

Base = declarative_base()

def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
