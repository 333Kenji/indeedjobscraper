from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# from app import db, ml, viz
#from app import database, models, schemas
from app.src import viz


#
from typing import List
import uvicorn
import pandas as pd
# from app import reddit_api
from sqlalchemy.orm import Session
#from app.database import engine, get_db, SessionLocal
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Depends, status, Request, Form, HTTPException
#
description = """
IndeedApp
An app that analyzes data from Indeed.com using ML to predict the salary\n
of listings that don't mention one.\n
1 \n
2 \n
3 \n
Use data to find a place right for you to live.
"""

app = FastAPI(
    title="Indeed App API",
    description=description,
    docs_url="/",
)
application = app
#def get_db():
#    try:
#        db = SessionLocal()
#        yield db
#    finally:
#        db.close()


#models.Base.metadata.create_all(bind=engine)

#@app.get("/")
#def main():
#    return RedirectResponse(url="/docs/")

#Recording6
#@app.get("/Record/", response_model=List[schemas.Record])
#def show_records(db: Session = Depends(get_db)):
#    records = db.query(models.Record).limit(1)
#    return records
#

# Define templates directory for Jinja2
templates = Jinja2Templates(directory='app/templates')  
# Define the directory for the static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")  
#models.Base.metadata.create_all(engine)  # create all sql tables (empty) if they don't exist



@app.get('/ex', response_class=HTMLResponse)
def index(request: Request):
    #posts = pd.read_sql('posts', engine)

    #df = pd.read_csv('data/graph_ready.csv', index_col=0)
    viz.create_cumsum_plot()
    viz.create_cumsum_plot()
    viz.create_bar()
    viz.create_usmap()
    viz.create_sunburst()
    viz.create_table()
    return templates.TemplateResponse('index.html', {
        'request': request})


#@app.get("/exe", response_class=HTMLResponse, status_code=status.HTTP_201_CREATED)
#def learn(request: Request, subreddit: str = Form('todayilearned'), db: Session = Depends(get_db)):
    #records = pd.read_sql('posts', engine)

    #return templates.TemplateRespose('index.html')


#@app.post("/exe", response_class=HTMLResponse, status_code=status.HTTP_201_CREATED)
#def learn(request: Request, subreddit: str = Form('todayilearned'), db: Session = Depends(get_db)):
#    # get the result of a request on the reddit api as json. a random post from the specified subreddit
#    res = reddit_api.get_reddit(subreddit=subreddit)
#    # get preprocessed post data as a dictionary
#    data = reddit_api.get_post_details(res)
#    # insert post data in sql table
#    data_for_post = sql_table_entities.Post(reddit_post_id=data['reddit_post_id'], title=data['title'],
#                                            score=data['score'], created=data['created'],
#                                            upvote_ratio=data['upvote_ratio'], thumbnail=data['thumbnail'],
#                                            url=data['url'],
#                                            permalink=data['permalink'], num_comments=data['num_comments'],
#                                            total_awards_received=data['total_awards_received'],
#                                            subreddit=data['subreddit'], timestamp_accessed=data['timestamp_accessed'])
#    db.add(data_for_post)
#    db.commit()
#    db.refresh(data_for_post)
#
#    # read table of posts and create the cumulative plot in the static folder, so it will be rendered later in jinja
#    posts = pd.read_sql('posts', engine)
#    reddit_api.create_cumsum_plot(posts)
#
#    # get preprocessed post comment data as a pd DataFrame
#    comments = reddit_api.get_comments_details(res)
#
#    # insert each comment in db
#    for i, row in comments.iterrows():
#        data_for_comment = models.Comment(reddit_comment_id=row['reddit_comment_id'],
#                                                      score=row['score'], body=row['body'],
#                                                      total_awards_received=row['total_awards_received'],
#                                                      owner=data_for_post)  # for 1:N relationship
#        db.add(data_for_comment)
#        db.commit()
#        db.refresh(data_for_comment)
#
#    return templates.TemplateResponse('index.html', {
#        'request': request, 'title': data['title'], 'score': data['score'], 'created': data['created'],
#        'thumbnail': data['thumbnail'], 'url': data['url'],
#        'upvote_ratio': data['upvote_ratio'], 'permalink': data['permalink'],
#        'total_awards_received': data['total_awards_received'], 'num_comments': data['num_comments'],
#        'comments': comments  # comments is a pd DataFrame. Everything else is a single value
#    })


#app.include_router(database.router, tags=["Database"])
# app.include_router(ml.router, tags=["Machine Learning"])
#app.include_router(viz.router, tags=["Visualization"])
# 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.on_event("startup")
# async def startup():
#     await db.database.connect()
# 
# 
# @app.on_event("shutdown")
# async def shutdown():
#     await db.database.disconnect()
# 

if __name__ == "__main__":
    uvicorn.run(application, debug=True)



