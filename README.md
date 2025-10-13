# a simple streamlit app

In this tutorial, we:

1. Load a data frame of coffee reviews, clean the data, and prepare it for modeling.
2. Use sentiment analysis on subsets of this data.
3. Build a web app which uses this sentiment analysis tool, and test it out locally.
4. Store data on Backblaze, and connect it to Streamlit.
5. Host the web app online using Streamlit (and GitHub).

## Prerequisites

This tutorial assumes the following:

- You are comfortable with Python coding in a Jupyter-compatible IDE like VS Code.
- You have a GitHub account, and you're familiar with using Git and .gitignore files.
- You can build and manage a pip Python environment.
- You are familiar with environment variables.

## Project Setup

> Download [this data](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset/) (i.e., the coffee_analysis.csv file) for this web app. Refer to the *utils/modeling.py* file for some preprocessing.

1. Create a new folder for your web app (give it an [appropriate](https://gravitydept.com/blog/devising-a-git-repository-naming-convention) name), and place it in your GitHub "projects" folder.
2. Initiate a Git repository in this folder. You'll want to use Git early and often to ensure that your changes are tracked, and you can always go back to (or compare with) a past version that works.
   - Note the contents of the .gitignore file. **Environment files (such as the .env file) should be strictly ignored.**
3. Initialize a pip environment using the *env.yml* file in this directory. For example, if you're using Ana/Miniconda, you'll use [this](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
   - The use of "env.yml" instead of "environment.yml" is intentional. If you prefer to use "environment.yml", you may want to add the file to your .gitignore list (see below).
4. Any time you add new packages to the environment, **update the .yml file**, and save an updated requirements file with `pip freeze > requirements.txt` (this should be run **within the environment**, so you'll need to activate it first).
   - **All packages should be installed using pip.**

**Note:** [Streamlit will only use \*one dependency file](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/app-dependencies#other-python-package-managers). If you have both a "requirements.txt" file *and* an "environment.yml" file in your repository, you will likely run into issues. The recommended option is to keep the *requirements.txt file tracked on GitHub and the "environment.yml" file ignored (and unseen by Streamlit Cloud).

### Modules

Consider creating a *scratch.ipynb* file for testing out code as you build your web app, and add it to your *.gitignore* file. When you feel ready, divide your code into separate Python files, saved in modular folders. For example, you'll notice in this repository, there is a *utils* folder. In that folder, we have a few Python files with code tested out in Jupyter first. Once it was determined the code worked in Jupyter, it was transferred into the separate Python file.

This *utils* folder, [including the empty *\_\_init__.py* file](https://stackoverflow.com/a/48804718), creates a Python [package](https://docs.python.org/3/tutorial/modules.html#packages) of multiple modules. If you open a notebook from the directory containing this folder, you'll find you can run code like `from utils.modeling import clean_data`. Using modules, and maybe the [autoreload](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html) functionality in Jupyter, keeps your notebook(s) clean and it makes debugging much easier.

## Streamlit

The app in this repository is run on [Streamlit](https://streamlit.io/).

Code can be tested locally with `streamlit run app.py` (assuming your app file name is `app.py`, of course). Once the app is running as you like, you can click the "Deploy" button on the upper right of the page. If you haven't already, you may need to [set up a Streamlit Cloud account](https://docs.streamlit.io/streamlit-community-cloud/get-started) to link your project GitHub repository.

### Environment Variables

Use [dotenv](https://github.com/theskumar/python-dotenv#getting-started) to manage environment variables. You'll need to `pip` install it, as directed in the instructions, then create a file with the name *.env* (notice the period) in your project directory to hold any keys or secrets. We'll be using dotenv to access Backblaze, below. **Make sure ".env" is added to your [.gitignore file](https://www.atlassian.com/git/tutorials/saving-changes/gitignore)**.

Once your environment variables are working locally, make sure you configure them accordingly on Streamlit using their [Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management).

### Caching

This app uses [Streamlit caching](https://docs.streamlit.io/library/advanced-features/caching) capabilities to cache (or save) objects whose loading would otherwise slow down the application. To do so, the app employs decorator functions. For more information on decorator functions in Python, I recommend the first two sections of [this article on RealPython](https://realpython.com/primer-on-python-decorators), by Geir Arne Hjelle.

In short, a decorator function adjusts its companion function by "wrapping" it in some other function. Here, the `@st.cache_data` or `@st.cache_object` decorator wraps its companion function such that the input is checked against a list of previous inputs of that function. If it's been run before, the output is pulled from the cache instead of being run again.

## Backblaze

Sign up for the [Backblaze B2 Cloud Storage](https://www.backblaze.com/b2/cloud-storage.html) service. At the time of writing this, the first 10GB of storage (with downloading up to 1GB/day) are free! For more information on Backblaze, check out their [documentation](https://www.backblaze.com/docs/cloud-storage-python-developer-quick-start-guide).

1. [Create at least one storage bucket](https://www.backblaze.com/docs/cloud-storage-create-and-manage-buckets#create-a-bucket) for your project, and at least one separate [application key](https://www.backblaze.com/docs/cloud-storage-create-and-manage-app-keys#create-an-app-key). It's recommended that you get into the habit of keeping your bucket *private*.
   - Your Application Key is **different** from the "Master" app key. You can set it to Read and Write to the storage bucket you created.
   - You can set the "Default Encryption" to be "Disabled". Enabling this just adds another level of protection which you can experiment with *if you like*. Since this also adds a level of complexity and room for error, this tutorial assumes this is "Disabled."
   - When shown your app key, make sure to save it in some file locally somewhere! You'll only be shown it once.
2. This package includes a few Backblaze helper functions in the *utils/b2.py* file.
   - For more information on these functions, take a look at the [Backblaze b2sdk documentation](https://b2-sdk-python.readthedocs.io/en/master/index.html) (this is the Python package driving the [Backblaze Native API](https://www.backblaze.com/apidocs/introduction-to-the-b2-native-api)).

## Other Resources

- You may consider using [Google Colab](https://colab.research.google.com/) to train your model and mold your project. Feel free to [use this as a template](https://colab.research.google.com/drive/1kgr3zMrC4sgBZXCx0jgVwXAIPXJgUJn_?usp=sharing) (i.e., make a copy of that notebook if you like). 
- Note that this app just uses the native Streamlit visualization functionality. For more flexibility on your visualization options, see [here](https://docs.streamlit.io/library/api-reference/charts).
- [Render](https://render.com/) and [Railway](https://railway.app/) are alternatives for hosting more robust web apps, but their free tiers can be restricting. In these cases, it's best to just pay the monthly fee.
- [Plotly Dash](https://dash.plotly.com/tutorial) is an alternative for building extensive interactive visualizations, dashboards, and more intriguing Python-based products. It requires a more robust app hosting resource like Render or Railway, and it does have a bit of a steeper learning curve.