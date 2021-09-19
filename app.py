from flask import Flask,render_template,request,Markup
from datetime import date,timedelta
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
app=Flask(__name__)
@app.route("/")
def home():
    d=date.today()
    return render_template("index.html",date = d)
@app.route("/stock", methods=['GET','POST'])
def stock():
    code=request.form.get("stock code")
    ticker=yf.Ticker(code)
    inf=ticker.info
    new_df = pd.DataFrame().from_dict(inf, orient="index").T
    summery= new_df['longBusinessSummary'][0]
    name=new_df['shortName'][0]
    logo=new_df['logo_url'][0]
    return render_template("stock.html",code=code,date=date.today(),summery=summery,name=name,logo=logo)
@app.route("/stock-price", methods=['GET','POST'])
def stockprice():
    startdate=request.form["start date"]
    code=request.form.get("code")
    ticker=yf.Ticker(code)
    inf=ticker.info
    new_df = pd.DataFrame().from_dict(inf, orient="index").T
    summery= new_df['longBusinessSummary'][0]
    name=new_df['shortName'][0]
    logo=new_df['logo_url'][0]
    df=yf.download(code,startdate,date.today())
    df.reset_index(inplace=True)
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(3)
    plt.plot(df['Date'],df['EWA_20'])
    plt.title("Exponential Moving Average vs Date")
    svg_file = BytesIO()
    plt.savefig(svg_file, format='svg')     # save the file to io.BytesIO
    svg_file.seek(0)
    svg_data = svg_file.getvalue().decode() # retreive the saved data
    svg_data = '<svg' + svg_data.split('<svg')[1]
    return render_template("stockprice.html",code=code,date=date.today(),summery=summery,name=name,logo=logo,svg_data=svg_data)
@app.route("/indicators", methods=['GET','POST'])
def indicators():
    startdate=request.form["start date"]
    code=request.form.get("code")
    ticker=yf.Ticker(code)
    inf=ticker.info
    new_df = pd.DataFrame().from_dict(inf, orient="index").T
    summery= new_df['longBusinessSummary'][0]
    name=new_df['shortName'][0]
    logo=new_df['logo_url'][0]
    df=yf.download(code,startdate,date.today())
    df.reset_index(inplace=True)
    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(3)
    plt.scatter(df['Date'],df['Open'])
    plt.scatter(df['Date'],df['Close'])
    plt.legend(["Open", "Close"], loc ="upper right")
    plt.title("Closing and Opening Price vs Date")
    svg_file = BytesIO()
    plt.savefig(svg_file, format='svg')     # save the file to io.BytesIO
    svg_file.seek(0)
    svg_data = svg_file.getvalue().decode() # retreive the saved data
    svg_data = '<svg' + svg_data.split('<svg')[1]
    return render_template("indicators.html",code=code,date=date.today(),summery=summery,name=name,logo=logo,svg_data=svg_data)
@app.route("/forecast",methods=['GET','POST'])
def forecast():
    end_date=request.form["date"]
    x=list(map(int,end_date.split("-")))
    forecast=(date(x[0],x[1],x[2])-date.today()).days
    startdate=date.today()-timedelta(2*forecast)
    code=request.form.get("code")
    df=yf.download(code,startdate,date.today())
    df['Prediction']=df[['Adj Close']].shift(-forecast)
    df=df[['Prediction','Adj Close']]
    new_df=df[:-forecast]
    X=np.array(new_df.drop('Prediction',1))
    y=np.array(new_df['Prediction'])
    params={
        'C':[1,10,20,30,40],
        'epsilon':[0.1,0.2,0.3,0.4],
        'gamma':['scale','auto']
    }
    reg=GridSearchCV(SVR(kernel='rbf'),params,cv=5,return_train_score=False)
    reg.fit(X,y)
    predprice=reg.predict(df.drop('Prediction',1)[-forecast:])[forecast-1]
    ticker=yf.Ticker(code)
    inf=ticker.info
    new_df = pd.DataFrame().from_dict(inf, orient="index").T
    summery= new_df['longBusinessSummary'][0]
    name=new_df['shortName'][0]
    logo=new_df['logo_url'][0]
    return render_template("forecast.html",code=code,date=date.today(),summery=summery,name=name,logo=logo,predprice=round(predprice,2),enddate=end_date)
app.run(debug=True)