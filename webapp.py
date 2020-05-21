from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import numpy as np
import re
from flask import Flask, flash, render_template,url_for, request, Markup, make_response, Response
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io , os
import random
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
app.secret_key = b'daire'

# Loading of Models
WikiModel = KeyedVectors.load_word2vec_format('models/Wiki.bin', binary=True)
#GoogleModel = KeyedVectors.load_word2vec_format('models/Google.bin', binary=True)
#GloveModel = KeyedVectors.load_word2vec_format(filename, binary=False)
#FastTextModel = KeyedVectors.load_word2vec_format('models/FastText.vec')
Wiki = "Wiki"
Google = "Google"
FastText = "FastText"
Ensemble = "Ensemble"
Glove = "Glove"

# Initial home page
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

# The Creation of the lexicons
@app.route("/recommend", methods= ['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        
        # Getting the 3 hidden values stored
        query = request.values.get("query")
        str_keywords = request.values.get("keywords")
        # Number of results being shown.
        number = int(request.values.get("numb"))

        # get the selected check boxes
        removals = request.form.getlist("check")
        # convert the strings to lists
        updated_keywords = request.values.get("keep_keywords")
        # Checking if the updated keywords is empty. E.g. If it's the first set of removals
        if not updated_keywords:
            keep_keywords = []
            original_keywords = str_keywords.split(",")
            # subtract the removals from original keywords to get the ones to keep
            for keyword in original_keywords:
                if not keyword in removals:
                    keep_keywords.append( keyword )
            return render_template("recommend2.html", query=query, keep_keywords=keep_keywords, str_keywords=str_keywords, number=number)
        else:
            # subtract the removals from original keywords to get the ones to keep
            updated_keywords = re.findall(r'\w+', updated_keywords)
            keep_keywords = []
            # subtract the removals from original keywords to get the ones to keep
            for keyword in updated_keywords:
                if not keyword in removals:
                    keep_keywords.append( keyword )
            return render_template("recommend2.html", query=query, keep_keywords=keep_keywords, removals=removals, updated_keywords=updated_keywords, number=number)

    else:
        keywords = []
        # Assigning user input
        s_keywords = request.args.get("keyword", default = "")
        # Cleaning input
        queryx = s_keywords.split(',')
        query = ", ".join(queryx)
        string = ' '.join([str(elem) for elem in queryx]) 
        string = string.lower()
        # Creating Suffix list and creating list of seed word with suffixs
        test_list = ['','er', 'ing', 's', 'ed', 'or', 'en', 'y'] 
        suf_res = [string + sub for sub in test_list] 
        pattern = '|'.join(suf_res)
        # Getting the model and number of similar words user input
        model = request.args.get('method')
        number = int(request.args.get('num'))
        # Checking Model used
        if Ensemble in model:
            # Checking if Seed word is in Vocab
            if string in GoogleModel.vocab:
                # Getting 30 most similar words to seed word in that model, putting it into a dataframe and cleaning x3
                FastList = FastTextModel.most_similar(positive=string, topn=30)
                dfFast = pd.DataFrame(FastList, columns = ['Most Similar' , 'Vector Accuracy'])
                cols = list(dfFast.columns)
                cols.remove('Most Similar')
                dfFast[cols]
                # Calculating the z-score of the vector accuracy and making a new column
                for col in cols:
                    col_zscore = col + '_zscore'
                    dfFast[col_zscore] = (dfFast[col] - dfFast[col].mean())/dfFast[col].std(ddof=0)

                GoogleList = GoogleModel.most_similar(positive=string, topn=30)
                dfGoogle = pd.DataFrame(GoogleList, columns = ['Most Similar' , 'Vector Accuracy'])
                cols1 = list(dfGoogle.columns)
                cols1.remove('Most Similar')
                dfGoogle[cols1]
                for col in cols1:
                    col_zscore = col + '_zscore'
                    dfGoogle[col_zscore] = (dfGoogle[col] - dfGoogle[col].mean())/dfGoogle[col].std(ddof=0)

                GloveList = WikiModel.most_similar(positive=string, topn=30)
                dfGlove = pd.DataFrame(GloveList, columns = ['Most Similar' , 'Vector Accuracy'])
                cols2 = list(dfGlove.columns)
                cols2.remove('Most Similar')
                dfGlove[cols2]
                for col in cols2:
                    col_zscore = col + '_zscore'
                    dfGlove[col_zscore] = (dfGlove[col] - dfGlove[col].mean())/dfGlove[col].std(ddof=0)
                
                # Combining the 3 dataframes and cleaning for unwanted entries
                df2 = pd.concat([dfFast, dfGlove])
                dfNew = pd.concat([df2, dfGoogle])
                dfNew = dfNew.apply(lambda x: x.astype(str).str.lower())
                dfNew = dfNew[~dfNew['Most Similar'].str.contains("_")]
                dfNew = dfNew[~dfNew['Most Similar'].str.contains("-")]
                dfNew = dfNew[~dfNew['Most Similar'].str.contains('\d')]
                dfNew = dfNew[~dfNew['Most Similar'].str.contains(pattern)]
                dfNew['Vector Accuracy_zscore'] = dfNew['Vector Accuracy_zscore'].astype(float)
                # Sorting the data frame by highest Z-score and dropping any dublicates
                dfNew = dfNew.sort_values(by = ['Vector Accuracy_zscore'], ascending = [False])
                dfNew = dfNew.drop_duplicates(subset=['Most Similar'], keep="first")
                dfNew = dfNew.reset_index(drop=True)
                # Assigning the ordered most similar words to keywords
                keywords = (dfNew['Most Similar'])
                str_query = "".join(query)
                str_keywords = ",".join(keywords)
                return render_template("recommend.html", query=query, keywords=keywords, str_query=str_query, str_keywords=str_keywords, number=number )
            # Flash error if seed word not in vocab
            else:
                flash("invalid word entry. Please enter another word.")
                return render_template('home.html')

        # Checking what model and then getting most similar words
        elif Wiki in model:
            if string in WikiModel.vocab:
                result = WikiModel.most_similar(positive=string, topn=50)
            else:
                flash("invalid word entry. Please enter another word.")
                return render_template('home.html')
        elif Google in model:
            if string in GoogleModel.vocab:
                result = GoogleModel.most_similar(positive=string, topn=50)
            else:
                flash("invalid word entry. Please enter another word.")
                return render_template('home.html')
        elif FastText in model:
            if string in FastTextModel.vocab:
                result = FastTextModel.most_similar(positive=string, topn=50)
            else:
                flash("invalid word entry. Please enter another word.")
                return render_template('home.html')

        # Creating dataframe of words, cleaning and assigning to keywords
        df = pd.DataFrame(result, columns = ['Most Similar' , 'Vector Accuracy'])
        df = df.apply(lambda x: x.astype(str).str.lower())
        df = df[~df['Most Similar'].str.contains("_")]
        df = df[~df['Most Similar'].str.contains("-")]
        df = df[~df['Most Similar'].str.contains('\d')]
        df = df[~df['Most Similar'].str.contains(pattern)]
        df = df.reset_index(drop=True)
        keywords = (df['Most Similar'])
        str_query = "".join(query)
        str_keywords = ",".join(keywords)
        return render_template("recommend.html", query=query, keywords=keywords, str_query=str_query, str_keywords=str_keywords, number=number )

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

# Fuction to generate 2-D visualization of input seed words
def tsne_plot(title, labels, embedding_clusters, word_clusters, a):
    plt.figure(figsize=(8, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=1, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    # Encoding image in base 64
    plot_url = base64.b64encode(img.getvalue()).decode()
    return(plot_url)



@app.route('/embeddingvisualizations')
def embeddingvisualizations():
    # Getting user input words and model
    s_keywords = request.args.get("keyword", default = "")
    model = request.args.get('method')
    keys = []
    for keyword in s_keywords.split(" "):
        keys.append( keyword.strip() )
    # Model selection x3
    if Wiki in model:
        # Checking if all the words are in the vocab
        result = all(key in WikiModel.vocab for key in keys)
        if result:
            embedding_clusters = []
            word_clusters = []
            # Creating lists of most similar words for all the input words
            for word in keys:
                embeddings = []
                words = []
                for similar_word, _ in WikiModel.most_similar(word, topn=8):
                    words.append(similar_word)
                    embeddings.append(WikiModel[similar_word])
                embedding_clusters.append(embeddings)
                word_clusters.append(words)
            # Using T-Sne with perplexity 10 to create the 2D arrays
            embedding_clusters = np.array(embedding_clusters)
            n, m, k = embedding_clusters.shape
            tsne_model2d = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3500, random_state=32)
            embeddings2d = np.array(tsne_model2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
            # Calling the fuction to plot the graph and sending it to the page
            plot_url = tsne_plot('Visualization of Most Similar Words to Input Seed Words and their Relation to Each Other', keys, embeddings2d, word_clusters, 0.7)
            model_plot = Markup('<img src="data:image/png;base64,{}" width: 360px; height: 288px>'.format(plot_url))
            return render_template("embeddingvisualizations.html", model_plot=model_plot)
        else:
            flash("One or more of the input seed words were not in the models vocabulary. Please try again!")
            return render_template('visualizations.html')

    elif Google in model:
        result = all(key in GoogleModel.vocab for key in keys)
        if result:
            embedding_clusters = []
            word_clusters = []
            for word in keys:
                embeddings = []
                words = []
                for similar_word, _ in GoogleModel.most_similar(word, topn=8):
                    words.append(similar_word)
                    embeddings.append(GoogleModel[similar_word])
                embedding_clusters.append(embeddings)
                word_clusters.append(words)

            embedding_clusters = np.array(embedding_clusters)
            n, m, k = embedding_clusters.shape
            tsne_model2d = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3500, random_state=32)
            embeddings2d = np.array(tsne_model2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    
            plot_url = tsne_plot('Visualization of Most Similar Words to Input Seed Words and their Relation to Each Other', keys, embeddings2d, word_clusters, 0.7)
            model_plot = Markup('<img src="data:image/png;base64,{}" width: 360px; height: 288px>'.format(plot_url))
            return render_template("embeddingvisualizations.html", model_plot=model_plot)
        else:
            flash("One or more of the input seed words were not in the models vocabulary. Please try again!")
            return render_template('visualizations.html')

    elif Fast in model:
        result = all(key in FastModel.vocab for key in keys)
        if result:
            embedding_clusters = []
            word_clusters = []
            for word in keys:
                embeddings = []
                words = []
                for similar_word, _ in FastModel.most_similar(word, topn=8):
                    words.append(similar_word)
                    embeddings.append(FastModel[similar_word])
                embedding_clusters.append(embeddings)
                word_clusters.append(words)

            embedding_clusters = np.array(embedding_clusters)
            n, m, k = embedding_clusters.shape
            tsne_model2d = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3500, random_state=32)
            embeddings2d = np.array(tsne_model2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    
            plot_url = tsne_plot('Visualization of Most Similar Words to Input Seed Words and their Relation to Each Other', keys, embeddings2d, word_clusters, 0.7)
            model_plot = Markup('<img src="data:image/png;base64,{}" width: 360px; height: 288px>'.format(plot_url))
            return render_template("embeddingvisualizations.html", model_plot=model_plot)
        else:
            flash("One or more of the input seed words were not in the models vocabulary. Please try again!")
            return render_template('visualizations.html')



@app.route('/about')
def about():
    return render_template("about.html")
    

if __name__ == '__main__':
    app.run(debug=True)



