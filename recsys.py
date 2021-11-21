import numpy as np

class RecSys:
    def __init__(self,movies):
        self.movies=movies
        self.mean_scores=3
    def fit(self,data_table):
        data_with_scores=data_table[['userId','movieId','rating']].values
        temp=data_table['movieId'].unique()
        movies_dict={temp[i]:i for i in range(len(temp))}

        temp=data_table['userId'].unique()
        users_dict={temp[i]:i for i in range(len(temp))}
        r= len(set(data_table['userId'].values))
        c= len(set(data_table['movieId'].values))
        temp=np.zeros((r,c))
    
        for i in data_with_scores:
            temp[users_dict[i[0]], movies_dict[i[1]]]= i[2]
        
        self.scores_table=self.__reconstruct_result_table(temp)
        self.users_dict=users_dict
        self.movies_dict=movies_dict
    
    def predict(self,user_item_table,get_name=False):
        predictions=[]
        for i in user_item_table.values:
            if get_name:
                predictions.append([self.predict_one(i[0],i[1]), self.__get_movie_name(i[1])])
            else:
                predictions.append([self.predict_one(i[0],i[1])])
        return predictions
        
    
    def predict_one(self,user_id,item_id):
        if (user_id in self.users_dict.keys())and(item_id in self.movies_dict.keys()):
            return self.scores_table[self.users_dict[user_id], self.movies_dict[item_id]]
        else:
            return self.mean_scores
    def __get_movie_name(self,item_id):
        name= self.movies[self.movies['movieId']==item_id]['title'].tolist()
        return name
        
    def __reconstruct_result_table(self,temp):
        u, s, vh=np.linalg.svd(temp,full_matrices=True)
        sigma= np.zeros((temp.shape[0],temp.shape[1]))
        sigma[:temp.shape[0],:temp.shape[0]]=np.diag(s)
        result_table=u.dot(sigma.dot(vh))
        return result_table