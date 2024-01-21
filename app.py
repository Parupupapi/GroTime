from flask import Flask, render_template, request
app = Flask(__name__)

# Your machine learning code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[30]:


ideal_plant_temp = {'Lettuce':62.5, 'Lemon Grass':80,'Tulsi Basil':75}
ideal_plant_humidity = {'Lettuce':60, 'Lemon Grass':55,'Tulsi Basil':50}
ideal_plant_sunlight = {'Lettuce':4, 'Lemon Grass':5,'Tulsi Basil':7}



# In[37]:


data = pd.read_csv('/Users/anisiva/Documents/Github/LastTry/linear regression project/weather_data.csv')
data.head()


# In[38]:


x = data.drop(['Rating','Date'],axis=1).values
y= data['Rating'].values



# In[ ]:





# In[ ]:





# In[39]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[40]:


from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(x_train,y_train)
# ... (copy the relevant parts of your code)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_rating', methods=['POST'])
def calculate_rating():
    plantname = request.form['plantname']
    idealtemp = request.form['idealtemp']
    idealhumidity = request.form['idealhumidity']
    idealsunlight = request.form['idealsunlight']

    # Your machine learning code for prediction
    ratingoutput = ml.predict([[int(idealtemp),int(idealtemp),int(idealhumidity),int(idealsunlight)]])
    output = ratingoutput[0]
    output = round(output,2)


    

   

    return render_template('result.html', plantname=plantname, output=output)

if __name__ == '__main__':
    app.run(debug=True)
