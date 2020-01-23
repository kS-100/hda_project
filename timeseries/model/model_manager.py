from sklearn.metrics import get_scorer
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
import os
from joblib import dump, load
from nbformat import current as nbf
from tabulate import tabulate

tqdm.pandas()

BASE_MODEL_PATH = "models"
NOTEBOOK_NAME = 'report_models.ipynb'
REGRESSION_SCORES = [
    "explained_variance",
    "max_error",
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "neg_mean_squared_log_error",
    "neg_median_absolute_error",
    "r2"
]

def create_model_folder(model_id):
    os.makedirs(os.path.join(BASE_MODEL_PATH,model_id))

def get_model_path(model_id, file_name):
    return os.path.join(BASE_MODEL_PATH, model_id, file_name)

def get_model_scores(model, x_train, y_train, x_test, y_test, print_values=True):
    scores = {}
    for regression_score in REGRESSION_SCORES:
        scorer = get_scorer(regression_score)
        try:
            score_train = scorer(model, x_train, y_train)
            score_test = scorer(model, x_test, y_test)
            if print_values:
                print(regression_score,"(train):",score_train)
                print(regression_score,"(test):",score_test)
                print("------------------------------------------------------------")
            scores[regression_score] = [score_train, score_test]
        except:
            print("Could not calculate score", regression_score)
            scores[regression_score] = [None, None]
    return scores

def evaluate_and_persist_model(model, x_train, y_train, x_test, y_test, model_name, features_to_show=20, show_scores=True, show_plots=True):
    # create folder for the model data. If the folder already exist an excpetion is thrown to prevent 
    # overriding the already saved data
    create_model_folder(model_name)
    
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    
    #save training and test data
    train_df = x_train.copy()
    train_df['y'] = y_train.values
    train_df['y_pred'] = y_pred_train
    train_df.to_csv(get_model_path(model_name, 'train.csv'))

    assert (train_df['y'].values == y_train.values).all()
    assert (train_df['y_pred'].values == y_pred_train).all()
    
    assert len(train_df) == len(pd.read_csv(get_model_path(model_name, 'train.csv')).dropna())
    
    
    test_df = x_test.copy()
    test_df['y'] = y_test.values
    test_df['y_pred'] = y_pred_test
    test_df.to_csv(get_model_path(model_name, 'test.csv'))
    
    assert (test_df['y'].values == y_test.values).all()
    assert (test_df['y_pred'].values == y_pred_test).all()
    
    assert len(test_df) == len(pd.read_csv(get_model_path(model_name, 'test.csv')).dropna())
    
    #save used parameters just in case
    param_df = pd.DataFrame(model.get_params(), index=[0])
    param_df.to_csv(get_model_path(model_name, 'params.csv'), index=False)
    
    #serialize the model itself
    dump(model, get_model_path(model_name, 'model.joblib'))    

    scores = get_model_scores(model, x_train , y_train, x_test, y_test, print_values=show_scores)
    df_scores = pd.DataFrame.from_dict(scores,orient = 'index')
    df_scores = df_scores.T
    df_scores.index = ['train', 'test']
    df_scores = df_scores.unstack().to_frame().sort_index(level=1).T
    df_scores.columns = df_scores.columns.map('_'.join)
    df_scores.to_csv(get_model_path(model_name, 'scores.csv'), index=False)

    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(list(range(0,len(y_test))), y_test.values, label="True")
    plt.scatter(list(range(0,len(y_test))), y_pred_test, label="Prediction")
    plt.title('Vergleich True vs. Prediction (Test)')
    plt.ylabel('IPC-measurement')
    plt.legend()
    plt.savefig(get_model_path(model_name, 'complete_comp_test.pdf'))
    if show_plots:
        plt.show()
    

    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(list(range(0,len(y_train))), y_train.values, label="True")
    plt.scatter(list(range(0,len(y_train))), y_pred_train, label="Prediction")
    plt.title('Vergleich True vs. Prediction (Train)')
    plt.ylabel('IPC-measurement')
    plt.legend()
    plt.savefig(get_model_path(model_name, 'complete_comp_train.pdf'))
    if show_plots:
        plt.show()
    

    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(list(range(0,len(y_test[50:150]))), y_test.values[50:150], label="True")
    plt.plot(list(range(0,len(y_test[50:150]))), y_pred_test[50:150], label="Prediction")
    plt.title('Vergleich True vs. Prediction (Test)')
    plt.ylabel('IPC-measurement')
    plt.legend()
    plt.savefig(get_model_path(model_name, 'zoomed_comp_test.pdf'))
    if show_plots:
        plt.show()
    

    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(list(range(0,len(y_train[50:150]))), y_train.values[50:150], label="True")
    plt.plot(list(range(0,len(y_train[50:150]))), y_pred_train[50:150], label="Prediction")
    plt.title('Vergleich True vs. Prediction (Train)')
    plt.ylabel('IPC-measurement')
    plt.legend()
    plt.savefig(get_model_path(model_name, 'zoomed_comp_train.pdf'))
    if show_plots:
        plt.show()
    
    features_to_show=min(features_to_show, len(x_train.columns))
    
    if not hasattr(model, 'feature_importances_'):
        print("Skipping feature importance visualization because model has no feature importances!")
        return model
    
    importances = model.feature_importances_
    
    std = np.std(importances,
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    
    fi_df = pd.DataFrame(columns=['column_name', 'index', 'importance'])
    for f in range(features_to_show):
        print("%d. %s. feature %d (%f)" % (f+1, x_train.columns[indices[f]], indices[f], importances[indices[f]]))
        fi_df.loc[f] = [x_train.columns[indices[f]], indices[f], importances[indices[f]]]
    fi_df.to_csv(get_model_path(model_name, 'feature_importances.csv'))
    
    features_df = pd.DataFrame({'feature': x_train.columns})
    features_df.to_csv(get_model_path(model_name, 'features.csv'))
    

    # Plot the feature importances
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features_to_show), importances[indices][:features_to_show],
           color="r", align="center")
    feature_names = [x_train.columns[indices[f]] for f in range(features_to_show)]
    plt.xticks(range(features_to_show), feature_names, rotation='vertical')
    plt.xlim([-1, features_to_show])
    plt.savefig(get_model_path(model_name, 'feature_importances.pdf'), bbox_inches = "tight")
    if show_plots:
        plt.show()
    
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features_to_show), importances[indices][:features_to_show],
           color="r", align="center")
    plt.xticks(range(features_to_show), indices, rotation='vertical')
    plt.xlim([-1, features_to_show])
    plt.savefig(get_model_path(model_name, 'feature_importances_indicies.pdf'), bbox_inches = "tight")
    if show_plots:
        plt.show()
        
    return model
    
def create_import_cell(cells):
    text = """
from IPython.display import IFrame
import pandas as pd
    """
    text = text.strip()
    cell = nbf.new_code_cell(text)
    cells.append(cell)

def create_pdf_view_cell(image_file, cells):
    text = """
IFrame('%s', width=1200, height=500)
    """ % (image_file)
    text = text.strip()
    cell = nbf.new_code_cell(text)
    cells.append(cell)

def create_text_cell(text, cells):
    cells.append(nbf.new_text_cell('markdown', text))

def create_score_view_cell(df, cells):
    text = tabulate(df, tablefmt="pipe", headers="keys")
    create_text_cell(text, cells)

def create_headline_cell(headline, cells, level=1):
    text = ('#'*level)+' '+headline
    cells.append(nbf.new_text_cell('markdown', text))

def create_model_comparison_notebook():
    nb = nbf.new_notebook()
    cells = []
    
    create_headline_cell('Model Report', cells)
    create_import_cell(cells)
    
    df_scores_all_test = pd.DataFrame()
    df_scores_all_train = pd.DataFrame()
    
    for model_dir in next(os.walk(BASE_MODEL_PATH))[1]:
        if model_dir.startswith('.'):
            continue
        create_headline_cell(model_dir, cells, level=2)
        
        create_headline_cell('Scores', cells, level=3)
        df_scores = pd.read_csv(get_model_path(model_dir, 'scores.csv'))
        df_scores.index = [model_dir]
        create_score_view_cell(df_scores, cells)
        df_scores_all_train =df_scores_all_train.append(df_scores.filter(regex=("_train")))
        df_scores_all_test =df_scores_all_test.append(df_scores.filter(regex=("_test")))
        
        create_headline_cell('Params', cells, level=3)
        df_scores = pd.read_csv(get_model_path(model_dir, 'params.csv'))
        df_scores.index = [model_dir]
        create_score_view_cell(df_scores, cells)
        
        create_headline_cell('Prediction comparison', cells, level=3)

        create_headline_cell('Train', cells, level=4)
        create_pdf_view_cell(get_model_path(model_dir, 'complete_comp_train.pdf'), cells)
        create_pdf_view_cell(get_model_path(model_dir, 'zoomed_comp_train.pdf'), cells)
        create_headline_cell('Test', cells, level=4)
        create_pdf_view_cell(get_model_path(model_dir, 'complete_comp_test.pdf'), cells)
        create_pdf_view_cell(get_model_path(model_dir, 'zoomed_comp_train.pdf'), cells)
        
    create_headline_cell('Overall score comparison', cells, level=2)
    create_headline_cell('Test', cells, level=3)
    create_score_view_cell(df_scores_all_test, cells)
    create_headline_cell('Train', cells, level=3)
    create_score_view_cell(df_scores_all_train, cells)
    
    cells = cells[:2] + cells[-5:] + cells[2:-5]
    
    nb['worksheets'].append(nbf.new_worksheet(cells=cells))

    with open(NOTEBOOK_NAME, 'w') as f:
            nbf.write(nb, f, 'ipynb')




