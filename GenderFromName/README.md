1. #### Create an isolated Python environment for project

  Open CMD, install virtualenv: 

  ```
  pip install virtualenv
  ```

  CD to the path of GenderFromName project: and create environment

  ```
  virtualenv venv
  ```

  activate environment first though:

  ```
  source venv/bin/activate
  ```

  You should see a (venv) appear at the beginning of your terminal prompt indicating that you are working inside the virtualenv

2. #### Now you can start to install the package in this environment to run this project

  ```
  pip install tensorflow==2.6.2
  pip install keras==2.6.0
  pip install scikit-learn==0.24.2
  pip install plot-keras-history==1.1.36
  ```


  if other package is missing, please install accordinyly, after environment and package is installed.

3. ### you can run the model.py to train an MLP model with 80% accuracy

  ```
  python model.py
  ```

  The model will be saved in under Model folder

4. ### you can import flask framework to access an web app

  in CMD, you can do it by 

  ```
  pip install flask
  ```

  after that, you can run the app.py by 

  ```
  python app.py
  ```


  then,you can access the web, by click the link after app run
