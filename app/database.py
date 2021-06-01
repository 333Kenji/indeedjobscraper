"""Database functions"""

import os
from fastapi import APIRouter, Depends
import sqlalchemy
from dotenv import load_dotenv
import databases
import asyncio
from typing import Union, Iterable
from pypika import Query, Table, CustomFunction
from pypika.terms import Field
#
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
#

Field_ = Union[Field, str]

load_dotenv()
database_url = os.getenv("DATABASE_URL")
# database = databases.Database(database_url)

router = APIRouter()



#

SQLALCHEMY_DATABASE_URL = database_url
#
engine = create_engine(SQLALCHEMY_DATABASE_URL)
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
#
Base = declarative_base()
#
#



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



#



# @router.get("/info")
# async def get_url():
#     """Verify we can connect to the database,
#     and return the database URL in this format:
#     dialect://user:password@host/dbname
#     The password will be hidden with ***
#     """
# 
#     url_without_password = repr(database.url)
#     return {"database_url": url_without_password}
# 


