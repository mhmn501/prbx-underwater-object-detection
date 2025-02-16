# PRBX: Deep Learning for Underwater Object Detection

## Project Setup and Execution Guide

### Prerequisites
- **Anaconda**: This project requires Anaconda to manage virtual environments and dependencies. Ensure you have Anaconda installed. You can download it from [Anaconda's official site](https://www.anaconda.com/products/individual).

### Environment Setup
Follow these steps to set up your virtual environment and install the necessary dependencies:

1. **Clone the Repository**
   - Clone the project repository to your local machine using Git:
     ```bash
     git clone https://github.com/mhmn501/imlo.git
     cd path-to-your-project
     ```

2. **Create the Anaconda Environment**
   - Create a new virtual environment using Anaconda:
     ```bash
     conda create -n myenv python=3.8
     ```
   - Activate the newly created environment:
     ```bash
     conda activate myenv
     ```

3. **Install Dependencies**
   - Ensure the `requirements.txt` or `environment.yml` file is present in your project directory.
   - If using `environment.yml` and the newly created environment, use the following command to update the environment with all dependencies:
     ```bash
     conda env update --name myenv --file environemnt.yml --prune
     ```
   - If you have a `requirements.txt`, install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```
   

### Running the Code
To run the project using terminal:

1. **Directory and Environment**
   - Ensure the `Image-Classifier-Neural-Network.py` and `latest_model.pth` file is present in your project directory.
   - Ensure the virtual environment is already activated and you are within the project directory.
     
2. **Model Testing**
   - To validate the reported classification performance (the training part is commented out by default), run the file:
     ```bash
        python Image-Classifier-Neural-Network.py
        ```
     
3. **Model Training**
   - To train the model, remove this comment in the main function:
     ```
     # run_training(model, device, NUM_EPOCHS, loaders['train'], loaders['val'], optimizer, scheduler, loss_fn, start_epoch, best_val_loss, checkpoint_path)
     ```
   - Then run it the same as model testing.


To run the project and evaluate the model using Jupyter Notebook:

1. **Launch Jupyter Notebook**
   - With the virtual environment activated, start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```

2. **Open the Notebook**
   - Navigate to the project notebook (`Image-Classifier-Neural-Network.ipynb`) in the Jupyter Notebook interface opened in your web browser.

3. **Run the Notebook**
   - Execute the cells in sequence to train or evaluate the model. Ensure that the dataset path and any configuration settings are correct as per your setup.

4. **Model Evaluation**
   - To validate the reported classification performance, run the testing sections of the notebook which load the trained model and perform classification on the test dataset.

### Important Notes
- Ensure that the path to the dataset and the paths in code (e.g., for loading the model) are correct based on your local or server setup.
- The notebook includes detailed comments explaining each step of the code, which facilitates understanding and any required adjustments.

