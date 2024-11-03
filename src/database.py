import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Set up the database engine
engine = create_engine(DATABASE_URL)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Create a function to initialize the database
def init_db():
    from models import Base
    Base.metadata.create_all(engine)
