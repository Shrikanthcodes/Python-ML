import numpy as np
import os
from matplotlib import pyplot as plt

class PredictionData():
    
    def __init__(self, inputpath : str, index : int) -> None:
        '''
        A class responsible for file parsing, plotting and linear regression.

        Parameters
        ----------
        inputpath : str
            The input file's path. This file contains the dataset
        index : int
            The position of 
            the target variable in the dataset 

        Returns
        -------
        None.

        '''
        filein = os.path.abspath(inputpath)
        with open(filein) as f:
            self.columnnames = f.readline().strip().split(",")
            self.data = np.genfromtxt(f, delimiter = ',')
        self.index = index   
        self.target_var = self.columnnames[self.index]
        
        
    def scatter_plot(self) -> None:
        '''
        This function is responsible for plotting and displaying a sample plot for
        all features in dataset against the target variable.

        Variables:
        ----------
        figure, ax: matplotlib object
            The subplot objects

        Returns
        -------
        None.

        '''
        for i in range(len(self.columnnames)):
            if self.columnnames[i] != self.target_var:
                figure, ax = plt.subplots()
                ax.scatter(self.data[:,i], self.data[:,self.index],
                           marker="*", c="red", s=30)
                ax.set(title = "{} vs {}".format(self.columnnames[i],
                            self.target_var), xlabel= self.columnnames[i], 
                            ylabel=self.target_var)
    

    def normalize(self) -> np.ndarray:
        '''
        The data in the dataset is standardized in this function.
        (Standardizing refers to making the mean of each feature value 0
        and standard deviation = 1)
        
        Returns
        -------
        normalize_data : Numpy array
            Standardized copy of the data
        '''
        normalize_data = self.data.copy()
        normalize_data -= np.mean(normalize_data, axis=0)
        normalize_data /= np.std(normalize_data, axis=0)
        return normalize_data
    
    def linear_regression(self, normalized_np : np.ndarray) -> list:
        '''
        This function performs the linear regression models for each feature
        (column) in the dataset.

        Parameters
        ----------
        normalized_np : Numpy Array
            Standardized copy of the data

        Variables:
        -----------
        Res: List
            Residuals of each feature is saved in the list
        Coeff : List
            Coefficients of each feature is saved in this list
        Prediction_final: string
            Contains the name of the feature that provides the best prediction
        figure2, bx: matplotlib object
            The subplot objects
        

        Returns
        -------
        self.columnnames    :List
            List of features (columns) in the dataset
        Res : List
            Residuals of each feature is saved in the list
        Coeff : List
            Coefficients of each feature is saved in this list

        '''
        Res = []
        Coeff = []
        for i in range(len(self.columnnames)):
            if self.columnnames[i] != self.target_var:
                A = np.vstack([np.ones(len(normalized_np)), 
                                normalized_np[:,i]]).T
                B, R, x, _ = np.linalg.lstsq(A, 
                                normalized_np[:, self.index], rcond = None)
                figure2, bx = plt.subplots()
                bx.scatter(normalized_np[:,i], normalized_np[:,self.index],
                           marker="^", c="blue", s=30)
                bx.plot(normalized_np[:,i], B[0] + B[1]*normalized_np[:,i],
                                'r', label='Model Fit')
                bx.legend()
                bx.set(title = "Model: {} vs {}".format(self.columnnames[i],
                                self.target_var), xlabel=self.columnnames[i], 
                                ylabel=self.target_var)
                Res.append(R[0])
                Coeff.append(B)
        Minimum_R = min(Res)
        prediction_final = self.columnnames[Res.index(Minimum_R)]
        return self.columnnames, Res, Coeff, prediction_final
        
if __name__ == "__main__":
    inputpath = r"California house prices.csv"
    predictor = PredictionData(inputpath,8)
    predictor.scatter_plot()
    data_normalized = predictor.normalize()
    feature, R, B, finalfeature = predictor.linear_regression(data_normalized)