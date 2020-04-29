import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



def plot_stages(reg,X_train,y_train,X_test,y_test,ax,title=""):
    test_score = np.zeros(reg.n_estimators, dtype=np.float64)
    train_score = np.zeros(reg.n_estimators, dtype=np.float64)
    
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = np.sqrt(mean_squared_error(y_test, y_pred))
        
    for i, y_pred_train in enumerate(reg.staged_predict(X_train)):    
        train_score[i] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
    min_test_score = min(test_score)
    min_test_score_stage = np.argmin(test_score)    
    learning_rate=reg.learning_rate
    max_depth = reg.max_depth
    
    ax.set_title('RMSE : '+str(title),fontsize=9)
    ax.plot(np.arange(reg.n_estimators), train_score, 'b-', label='Training Set RMSE')
    ax.plot(np.arange(reg.n_estimators), test_score, 'r-', label='Test Set RMSE')
    ax.set_xlim((0,reg.n_estimators))
    ymin , ymax = ax.get_ylim()
    xmin , xmax = ax.get_xlim()
    ax.annotate('Min RMSE : '+str(round(min_test_score,3)), xy=(min_test_score_stage+10,min_test_score+0.1), xytext=(min_test_score_stage+10,min_test_score+0.1),color = "red")
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.hlines(y=min_test_score,xmin=0,xmax=reg.n_estimators,linestyles="dashed",color="grey")
    ax.vlines(x=min_test_score_stage,ymin=0,ymax=1,linestyles="dashed",color="grey")
    ax.set_xlabel('Boosting Iterations')
    ax.set_ylabel('RMSE')
    
    
def plot_coeff_importances(reg,data_columns,ax,title=""):
    X_columns = data_columns
    
    ordering = np.argsort(reg.feature_importances_)[::-1][:50]
    importances = reg.feature_importances_[ordering]
    feature_names = X_columns[ordering]
    x = np.arange(len(feature_names))
    
    ax.set_title('Importances : '+str(title))
    ax.bar(x, importances)
    ax.set_xticks(x + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax.set_ylabel('Feature importance')