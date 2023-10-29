#!/usr/bin/env python
# coding: utf-8

# In[208]:


import numpy as np
import pandas as pd
import sklearn
credits = pd.read_csv('IMDB/credits.csv')
keywords = pd.read_csv('IMDB/keywords.csv')
links_small = pd.read_csv('IMDB/links_small.csv')
movies_metadata = pd.read_csv('IMDB/movies_metadata.csv')
ratings_small = pd.read_csv('IMDB/ratings_small.csv')


# In[209]:


links_small_tmdid = links_small[links_small['tmdbId'].notnull()]['tmdbId']
links_small2 = links_small_tmdid.astype(int)
links_small = links_small[links_small['tmdbId'].isin(links_small_tmdid)]
links_small['tmdbId']=links_small['tmdbId'].astype(int)
links =links_small[['movieId']]
links['id'] = links_small['tmdbId']


# In[210]:


pd.set_option('display.max_colwidth', None)
movie_data = movies_metadata[['id','title','genres','popularity','vote_average','vote_count']]
movie_data = movie_data[movie_data['id'].str.contains('-') == False]
movie_data['id']=movie_data['id'].astype(int)
otherdata = pd.merge(credits,keywords,on='id')
movie_data = pd.merge(movie_data,otherdata,on='id')
movie_data = movie_data[movie_data['id'].isin(links_small_tmdid)]


# In[211]:


movie_data['vote_count'] = movie_data['vote_count'].replace(0.0, np.nan)
movie_data = movie_data.dropna(axis=0, subset=['vote_count'])
str_data = movie_data[['id','title','genres','cast','crew','keywords']]


# In[212]:


def genres_keywords_cleaner(k):
    s = k.split('}')
    if len(s)==1:
      s=[]
    else:
       p = "'name': '"
       b= len(s)-1;
       for i in range(b):
          s[i] = s[i].partition(p)[2] 
          s[i] = s[i][0:( len(s[i]) - 1)]
          s[i] = str.lower(s[i].replace(" ",""))
       s = s[0:b]  
    return s; 


# In[213]:


def cast_cleaner(k):
    s = k.split('}')
    p1 = "'name': '"
    p2 ="', 'order'"
    if len(s)==1:
       s=[]
    else:   
       for i in range(len(s)-1):
         s[i] = s[i].partition(p1)[2] 
         s[i] = s[i].partition(p2)[0] 
         s[i] = str.lower(s[i].replace(" ",""))
       if(len(s)-1>3): 
         s = s[0:4] 
       else:
         s =s[0:len(s)-1]  
    return s; 


# In[214]:


def get_director(k):
    word = "'job': 'Director'"
    p1 = "'name': '"
    p2 ="', 'profile_path'"
    if word in k:
       s = k.split('}')
       for i in range(len(s)):
         if word in s[i]:
             s = s[i]
             break   
       s = s.partition(p1)[2] 
       s = s.partition(p2)[0]  
       s = str.lower(s.replace(" ",""))
       s2 = [s,s] 
       return s2;
    else:
       return []


# In[215]:


str_data['crew'] = str_data['crew'].apply(lambda x: get_director(x))
str_data['cast'] = str_data['cast'].apply(lambda x: cast_cleaner(x))
str_data['genres'] = str_data['genres'].apply(lambda x: genres_keywords_cleaner(x))
str_data['keywords'] = str_data['keywords'].apply(lambda x: genres_keywords_cleaner(x))


# In[216]:


keywords_count = str_data.apply(lambda x :pd.Series(x['keywords']),axis=1).stack().reset_index(level=1,drop=True)
keywords_count.name = 'keywords'
keywords_count = keywords_count.value_counts()
keywords_count = keywords_count[keywords_count>1]


# In[217]:


def new_keywords(x):
  w = []
  for i in x:
    if i in keywords_count:
      w.append(i)
  return w  


# In[218]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
str_data['keywords'] = str_data['keywords'].apply(new_keywords)
str_data['keywords'] = str_data['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])


# In[219]:


str_data['all_str_features'] = str_data['crew'] + str_data['cast'] + str_data['genres'] + str_data['keywords']
str_data['all_str_features'] = str_data['all_str_features'].apply(lambda x: ' '.join(x))


# In[220]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
countvectorizer = CountVectorizer()
count_transform = countvectorizer.fit_transform(str_data['all_str_features'])
sim = cosine_similarity(count_transform)


# In[221]:


str_data = str_data.reset_index()
indexes = pd.Series(str_data.index, index=str_data['title'])
def content_based_str_recommand(movie_name,n):
    movie = indexes[movie_name]
    similar_movies = list(enumerate(sim[movie]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    similar_movies = similar_movies[0:n+1]
    similar_movies = [i[0] for i in similar_movies]
    return str_data.iloc[similar_movies]


# In[222]:


def content_base(movie_name):
  df = content_based_str_recommand(movie_name,10)
  df = df[1:11]
  titles = df['title']
  return titles


# In[ ]:





# In[223]:


#collaborative_recommander


# In[224]:


ratings = ratings_small[['userId','movieId','rating']]
movie_data2 = movies_metadata[['id','title']]
movie_data2 = movie_data2[movie_data2['id'].str.contains('-') == False]
movie_data2['id']=movie_data2['id'].astype(int)
movie_data2 = movie_data2[movie_data2['id'].isin(links_small2)]


# In[225]:


links = pd.merge(links,movie_data2,on='id')
ratings = pd.merge(ratings,links,on='movieId',how='left')


# In[226]:


rate_data = ratings[['userId','rating','title']]


# In[227]:


rate_data = pd.pivot_table(rate_data,index='userId',columns='title',values='rating')


# In[228]:


movie_states = ratings.groupby('title').agg({'rating':[np.size,np.mean]})
movie_states.columns = ['rate_size','avg_rate']
popularmovies = movie_states['rate_size']>=70
popularmovies = movie_states[popularmovies]
popularmovies2 = popularmovies['avg_rate']>=3
popularmovies = popularmovies[popularmovies2]


# In[229]:


def collaborative_recommand(movie_name,data,n):
    movie = data[movie_name]
    sim_movies = data.corrwith(movie)
    sim_movies = sim_movies.dropna()
    sim_movies =pd.DataFrame(sim_movies)
    sim_movies.columns = ['similarity']
    sim_movies = popularmovies.join(sim_movies)
    sim_movies =sim_movies.sort_values(by= ['similarity'],axis=0,ascending=False)
    sim_movies = sim_movies[1:n+1]
    return sim_movies


# In[230]:


def collaborative(movie_name):
  df = collaborative_recommand(movie_name,rate_data,10)
  df2 = list(df.index.values)
  return df2


# In[231]:


#ensemble


# In[232]:


popularmovies = movie_states['rate_size']>=30
popularmovies = movie_states[popularmovies]
popularmovies2 = popularmovies['avg_rate']>=3.4
popularmovies = popularmovies[popularmovies2]


# In[233]:


def ensemble(movie_name):
  str_data2 = content_based_str_recommand(movie_name,40)
  df =  rate_data[str_data2['title']]
  similarmovies = collaborative_recommand(movie_name,df,8)
  df2 = list(similarmovies.index.values)
  return df2


# In[ ]:





# In[ ]:




