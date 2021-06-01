from datetime import date
from pydantic import BaseModel


class Record(BaseModel):
    id: int
    jobtitle: str
    company: str
    location: str
    summary: str
    requirements: str
    description: str
    state: str
    city: str
    dateposted: date
    schedule: str
    salary: str
    role: str
    focus: str



    class Config:
        orm_mode = True




# class Record(BaseModel):
#     id: int
#     date: date
#     country: str
#     cases: int
#     deaths: int
#     recoveries: int
# 
#     class Config:
#         orm_mode = True