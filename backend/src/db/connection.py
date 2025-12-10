import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("NEON_DB_URL")

if not DATABASE_URL:
    raise ValueError("NEON_DB_URL is not set in environment variables")


# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    echo=False  # Set to True for SQL query logging
)

# Create a SessionLocal class to handle database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for declarative models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session for FastAPI endpoints
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Utility function to initialize the database tables
def init_db():
    """
    Create all database tables based on the defined models
    """
    print("Initializing database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise


if __name__ == "__main__":
    init_db()