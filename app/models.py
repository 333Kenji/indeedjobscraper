#import os
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, Date
from sqlalchemy.orm import relationship
from app.database import Base
from pydantic import BaseModel
from sqlalchemy.types import Date




class Record(Base):
    __tablename__ = "Record"
    id = Column(Integer, primary_key=True, index=True)
    jobtitle = Column(Text)
    company = Column(Text)
    location = Column(Text)
    summary = Column(Text)
    requirements = Column(Text)
    description = Column(Text)
    state = Column(Text)
    city = Column(Text)
    dateposted = Column(Date)
    schedule = Column(Text)
    salary = Column(Text)
    role = Column(Text)
    focus= Column(Text)




#class Record(Base):
#    __tablename__ = "Records"
#
#    id = Column(Integer, primary_key=True, index=True)
#    date = Column(Date)
#    country = Column(String(255), index=True)
#    cases = Column(Integer)
#    deaths = Column(Integer)
#    recoveries = Column(Integer)